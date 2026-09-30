// SPDX-License-Identifier: Apache-2.0
//! `SqliteVecColumn` — the single sqlite-vec (`vec0`) shim, shared by every store that keeps an
//! embedding column in SQLite: the Q->A cache (`cache.rs`), the overflow chunk store
//! (`chunks.rs`), and the curated corpus (`corpus.rs`).
//!
//! Each of those three carried its own copy of the identical plumbing — the `sqlite3_vec_init`
//! auto-extension registration, the open-with-disk-fallback + WAL PRAGMA block, the
//! reconcile-the-vector-table-when-the-embedder-dim-changes dance, and the
//! `emb MATCH ?1 ORDER BY distance LIMIT ?2` KNN. Three copies is three places to fix a bug and
//! three chances to drift apart. This is the one copy; each store keeps its own tables, its own
//! scoring, and its own eviction policy on top of it.
//!
//! Deliberately a *handle*, not an owner: the stores need their own `Connection` for their own
//! non-vector tables, so every method borrows the connection rather than holding it.
//!
//! Note on SQL construction: table and column names can't be bound as SQLite parameters, so they
//! are interpolated. Every one is an internal `&'static str` constant chosen by the calling store
//! — none is reachable from user input.

use crate::embed::Embedder;
use rusqlite::ffi::sqlite3_auto_extension;
use rusqlite::Connection;
use sqlite_vec::sqlite3_vec_init;

static VEC_INIT: std::sync::Once = std::sync::Once::new();

/// The WAL/mmap PRAGMA set applied in `tuned` mode, identical across all three stores.
const TUNED_PRAGMAS: &str = "PRAGMA journal_mode = WAL;
                             PRAGMA synchronous = NORMAL;
                             PRAGMA cache_size = -20000;
                             PRAGMA temp_store = MEMORY;
                             PRAGMA mmap_size = 2147483648;";

/// Register the sqlite-vec extension once per process (it's a global auto-extension).
///
/// Must run BEFORE any `Connection::open` that needs `vec0`: `sqlite3_auto_extension` installs a
/// hook that only fires for connections opened after it. `open_conn` calls this for you.
pub fn ensure_vec_registered() {
    VEC_INIT.call_once(|| unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
    });
}

/// Encode a vector as the little-endian f32 blob `vec0` stores.
pub fn emb_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Inverse of `emb_to_blob`: decode a little-endian f32 blob back into a vector. Trailing bytes
/// that don't form a whole f32 are ignored. Used to read a stored embedding back out of a `vec0`
/// table without re-embedding the source text (see `corpus::journal_take`).
pub fn blob_to_emb(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Open a store's connection: `db_path=None` -> in-memory, and a disk path that fails to open
/// falls back to in-memory with a warning rather than taking the process down. `what` names the
/// store in that warning (e.g. `"disk cache"` -> `memory: disk cache disabled (...)`).
/// Registers sqlite-vec first, and applies the shared PRAGMA set when `tuned`.
pub fn open_conn(db_path: Option<&str>, tuned: bool, what: &str) -> Connection {
    ensure_vec_registered();
    let conn = match db_path {
        Some(path) => match Connection::open(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("memory: {what} disabled ({path}: {e})");
                Connection::open_in_memory().expect("open in-memory sqlite")
            }
        },
        None => Connection::open_in_memory().expect("open in-memory sqlite"),
    };
    if tuned {
        conn.execute_batch(TUNED_PRAGMAS).expect("apply pragmas");
    }
    conn
}

/// A handle to one `vec0` virtual table (`table`) plus the `k`/`v` meta table (`meta_table`) that
/// records which embedder dim it was built for.
pub struct SqliteVecColumn {
    table: &'static str,
    meta_table: &'static str,
    dim: usize,
}

impl SqliteVecColumn {
    pub fn new(table: &'static str, meta_table: &'static str, dim: usize) -> Self {
        Self { table, meta_table, dim }
    }

    /// The embedder dim this column was built for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Insert a vector for a fresh `rowid`. Panics on failure. Use ONLY on startup/rebuild paths
    /// (e.g. `reconcile`), where a broken vector table means the store's KNN is silently wrong and a
    /// loud stop at boot is the right response. On the per-request path use [`try_insert`] instead —
    /// under `panic = "abort"` a panic here kills the whole server for every client.
    pub fn insert(&self, conn: &Connection, rowid: i64, v: &[f32]) {
        conn.execute(
            &format!("INSERT INTO {}(rowid, emb) VALUES (?1, ?2)", self.table),
            rusqlite::params![rowid, emb_to_blob(v)],
        )
        .unwrap_or_else(|e| panic!("insert {} row: {e}", self.table));
    }

    /// Best-effort insert for the per-request path: returns `false` on failure instead
    /// of aborting the process. A missing vector makes that one chunk unsearchable until it is
    /// re-inserted — a quiet degradation, but far better than taking down every in-flight request.
    #[must_use]
    pub fn try_insert(&self, conn: &Connection, rowid: i64, v: &[f32]) -> bool {
        conn.execute(
            &format!("INSERT INTO {}(rowid, emb) VALUES (?1, ?2)", self.table),
            rusqlite::params![rowid, emb_to_blob(v)],
        )
        .is_ok()
    }

    /// Replace the vector at `rowid` (delete-then-insert; `vec0` has no UPDATE for the emb
    /// column). Best-effort — used on the cache's overwrite path.
    pub fn upsert(&self, conn: &Connection, rowid: i64, v: &[f32]) {
        self.delete(conn, rowid);
        let _ = conn.execute(
            &format!("INSERT INTO {}(rowid, emb) VALUES (?1, ?2)", self.table),
            rusqlite::params![rowid, emb_to_blob(v)],
        );
    }

    /// Best-effort delete of `rowid`'s vector.
    pub fn delete(&self, conn: &Connection, rowid: i64) {
        let _ = conn.execute(
            &format!("DELETE FROM {} WHERE rowid=?1", self.table),
            rusqlite::params![rowid],
        );
    }

    /// Read a stored vector back out without re-embedding. `None` if the row has no vector.
    pub fn embedding_at(&self, conn: &Connection, rowid: i64) -> Option<Vec<f32>> {
        conn.query_row(
            &format!("SELECT emb FROM {} WHERE rowid = ?1", self.table),
            rusqlite::params![rowid],
            |r| r.get::<_, Vec<u8>>(0),
        )
        .ok()
        .map(|b| blob_to_emb(&b))
    }

    /// Exact brute-force KNN: up to `k` `(rowid, distance)` pairs, nearest first. Distance is
    /// cosine distance (the table is built `distance_metric=cosine`), so callers wanting a
    /// similarity use `1.0 - distance`.
    pub fn knn(&self, conn: &Connection, qv: &[f32], k: usize) -> Vec<(i64, f32)> {
        let qb = emb_to_blob(qv);
        let mut stmt = conn
            .prepare(&format!(
                "SELECT rowid, distance FROM {} WHERE emb MATCH ?1 ORDER BY distance LIMIT ?2",
                self.table
            ))
            .expect("prepare knn query");
        let it = stmt
            .query_map(rusqlite::params![qb, k as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)? as f32))
            })
            .expect("query knn rows");
        it.filter_map(|x| x.ok()).collect()
    }

    /// Reconcile the vector table against the current embedder's dim.
    ///
    /// Rebuilds (drop + recreate + re-embed every row of `source_table.text_col`) when the stored
    /// dim doesn't match or the table is missing; otherwise a no-op. This is what makes swapping
    /// the `Embedder` safe: the stored vectors are, by construction, always the ones the current
    /// embedder would produce. `noun` names the rows in the rebuild log line ("entries", "chunks",
    /// "corpus rows"). Returns the number of rows re-embedded (0 when no rebuild was needed).
    pub fn reconcile(
        &self,
        conn: &Connection,
        emb: &dyn Embedder,
        source_table: &str,
        text_col: &str,
        noun: &str,
    ) -> usize {
        let dim = self.dim;
        let stored_dim: Option<usize> = conn
            .query_row(
                &format!("SELECT v FROM {} WHERE k = 'embedder_dim'", self.meta_table),
                [],
                |r| r.get::<_, String>(0),
            )
            .ok()
            .and_then(|s| s.parse().ok());

        let vec_table_exists: bool = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?1",
                rusqlite::params![self.table],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if stored_dim == Some(dim) && vec_table_exists {
            return 0;
        }

        conn.execute(&format!("DROP TABLE IF EXISTS {}", self.table), [])
            .unwrap_or_else(|e| panic!("drop {}: {e}", self.table));
        conn.execute(
            &format!(
                "CREATE VIRTUAL TABLE {} USING vec0(emb float[{dim}] distance_metric=cosine)",
                self.table
            ),
            [],
        )
        .unwrap_or_else(|e| panic!("create {}: {e}", self.table));
        conn.execute(
            &format!(
                "INSERT INTO {}(k, v) VALUES ('embedder_dim', ?1) \
                 ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                self.meta_table
            ),
            rusqlite::params![dim.to_string()],
        )
        .expect("store embedder_dim");

        let rows: Vec<(i64, String)> = {
            let mut stmt = conn
                .prepare(&format!("SELECT rowid, {text_col} FROM {source_table}"))
                .expect("prepare");
            let it = stmt
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
                .expect("query source rows");
            it.filter_map(|x| x.ok()).collect()
        };
        let n = rows.len();
        for (rowid, text) in rows {
            self.insert(conn, rowid, &emb.encode(&text));
        }
        if n > 0 {
            println!("memory: re-embedded {n} {noun} after an embedder change");
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashingEmbedder;

    /// A store-shaped fixture: a `docs(rowid, body)` table + `docs_meta`, mirroring how the real
    /// stores pair a content table with a vec0 column.
    fn fixture(dim: usize) -> (Connection, SqliteVecColumn) {
        let conn = open_conn(None, false, "test store");
        conn.execute("CREATE TABLE docs (rowid INTEGER PRIMARY KEY, body TEXT)", []).unwrap();
        conn.execute("CREATE TABLE docs_meta (k TEXT PRIMARY KEY, v TEXT)", []).unwrap();
        (conn, SqliteVecColumn::new("vec_docs", "docs_meta", dim))
    }

    fn seed(conn: &Connection, bodies: &[&str]) {
        for b in bodies {
            conn.execute("INSERT INTO docs(body) VALUES (?1)", rusqlite::params![b]).unwrap();
        }
    }

    #[test]
    fn blob_roundtrip_preserves_vector() {
        let v = vec![0.5f32, -1.25, 0.0, 3.75];
        assert_eq!(blob_to_emb(&emb_to_blob(&v)), v);
    }

    #[test]
    fn blob_decode_ignores_trailing_partial_f32() {
        let mut b = emb_to_blob(&[1.0f32, 2.0]);
        b.push(0xAB); // a stray byte that can't form a whole f32
        assert_eq!(blob_to_emb(&b), vec![1.0f32, 2.0]);
    }

    #[test]
    fn reconcile_builds_then_is_a_noop() {
        let e = HashingEmbedder::with_dim(64);
        let (conn, col) = fixture(64);
        seed(&conn, &["alpha beta", "gamma delta"]);

        // First call builds the table and embeds every row...
        assert_eq!(col.reconcile(&conn, &e, "docs", "body", "docs"), 2);
        // ...and a second call is a no-op, since the stored dim already matches.
        assert_eq!(col.reconcile(&conn, &e, "docs", "body", "docs"), 0);
    }

    #[test]
    fn reconcile_rebuilds_when_the_embedder_dim_changes() {
        let (conn, col64) = fixture(64);
        seed(&conn, &["alpha beta", "gamma delta"]);
        assert_eq!(col64.reconcile(&conn, &HashingEmbedder::with_dim(64), "docs", "body", "docs"), 2);

        // A different embedder dim must force a full re-embed, not silently keep stale vectors.
        let col32 = SqliteVecColumn::new("vec_docs", "docs_meta", 32);
        assert_eq!(col32.reconcile(&conn, &HashingEmbedder::with_dim(32), "docs", "body", "docs"), 2);
        assert_eq!(col32.embedding_at(&conn, 1).expect("row 1 has a vector").len(), 32);
    }

    #[test]
    fn knn_ranks_the_matching_row_first() {
        let e = HashingEmbedder::with_dim(64);
        let (conn, col) = fixture(64);
        seed(&conn, &["the quick brown fox", "lorem ipsum dolor sit"]);
        col.reconcile(&conn, &e, "docs", "body", "docs");

        let hits = col.knn(&conn, &e.encode("quick brown fox"), 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 1, "the lexically-matching row must be nearest");
        assert!(hits[0].1 <= hits[1].1, "knn must return nearest-first");
    }

    #[test]
    fn upsert_replaces_rather_than_duplicating() {
        let e = HashingEmbedder::with_dim(64);
        let (conn, col) = fixture(64);
        seed(&conn, &["alpha"]);
        col.reconcile(&conn, &e, "docs", "body", "docs");

        let replacement = e.encode("totally different text");
        col.upsert(&conn, 1, &replacement);

        // Exactly one vector for the rowid, and it's the new one.
        assert_eq!(col.knn(&conn, &replacement, 10).len(), 1);
        assert_eq!(col.embedding_at(&conn, 1), Some(replacement));
    }

    #[test]
    fn delete_removes_the_row_from_knn() {
        let e = HashingEmbedder::with_dim(64);
        let (conn, col) = fixture(64);
        seed(&conn, &["alpha", "beta"]);
        col.reconcile(&conn, &e, "docs", "body", "docs");

        col.delete(&conn, 1);
        assert!(col.embedding_at(&conn, 1).is_none());
        let hits = col.knn(&conn, &e.encode("alpha"), 10);
        assert!(hits.iter().all(|(rowid, _)| *rowid != 1), "deleted row must not surface in knn");
    }
}
