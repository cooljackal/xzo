// SPDX-License-Identifier: Apache-2.0
//! Persistent chunk-vector store + message splitter for the overflow layer.
//!
//! Vector storage/search goes through the shared `vecstore::SqliteVecColumn` (`vec0` table + KNN
//! + dim-reconcile); this module stores `{content_hash, raw, summary, embedding}` keyed on a
//! content fingerprint. It is kept SEPARATE from the Q->A cache — its own tables (`chunks` /
//! `vec_chunks` / `chunk_meta`) so a shared sqlite file cannot collide, and typically its own
//! sidecar DB. The per-chunk digest cache is content-hash keyed, so an unchanged chunk skips
//! re-digestion (delta-only).

use crate::embed::Embedder;
use crate::vecstore::{self, SqliteVecColumn};
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// How many of the most-recent clock ticks `evict_to` protects from the LRU. Sized to
/// cover the chunks a single generous over-window request inserts before it reads them back, so a
/// concurrent request's eviction can never delete them mid-flight. Only bites when the entry cap is
/// set small; at a healthy cap the protected rows are never the oldest, so it is a no-op.
const EVICT_GUARD_BAND: u64 = 4096;

/// A chunk produced by the splitter for the CURRENT request. `positional_id` is the per-request
/// label the model sees (`"12"` or, for an oversized-message sub-chunk, `"12.3"`); `content_hash`
/// is the stable persistent cache key (decoupled from the positional id on purpose).
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub raw: String,
    pub content_hash: String,
    pub positional_id: String,
}

/// A KNN search hit from the chunk store.
#[derive(Clone, Debug)]
pub struct ChunkHit {
    pub content_hash: String,
    pub summary: String,
    pub raw: String,
    pub score: f32,
}

/// Cosine similarity of two equal-length vectors; `0.0` if either is zero-length or degenerate.
/// Used by [`ChunkStore::search_scoped`] to score a bounded candidate set directly, which the
/// `vec0` index cannot do (it has no metadata filter — see that method's note).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Stable content fingerprint (lowercase hex FNV-1a-64) of a string. Same bytes -> same hash,
/// so an unchanged message keeps its cache key across turns.
pub fn content_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// Split the bulky-middle message contents (already flattened to text, in order) into ordered
/// chunks on message boundaries. A single message whose `est_tokens` exceeds `chunk_target_tokens`
/// is sub-split by line/paragraph boundaries into ordered sub-chunks, labeled `"N.M"`; whole-message
/// chunks are labeled `"N"`. EVERY chunk is keyed `content_hash(raw)` — the key `ChunkStore` files
/// it under. Sub-chunks used to be keyed on `{parent}\0{index}`, which the store never saw: their
/// summaries were written to no row and looked up under a key with none, so each part of a large
/// tool output was summarized again on every over-window turn, and never matched a scoped search.
/// Empty/whitespace-only messages are skipped. Positional ids are 1-based over the retained set.
pub fn split_messages(middle: &[String], chunk_target_tokens: usize) -> Vec<Chunk> {
    let target = chunk_target_tokens.max(1);
    let mut out: Vec<Chunk> = Vec::new();
    let mut pos = 0usize; // top-level chunk counter (1-based when emitted)
    for msg in middle {
        if msg.trim().is_empty() {
            continue;
        }
        pos += 1;
        if crate::inject::est_tokens(msg) <= target {
            out.push(Chunk {
                raw: msg.clone(),
                content_hash: content_hash(msg),
                positional_id: pos.to_string(),
            });
        } else {
            // Oversized message: sub-split by line/paragraph, packing lines up to `target`.
            let subs = pack_lines(msg, target);
            for (j, sub) in subs.iter().enumerate() {
                out.push(Chunk {
                    raw: sub.clone(),
                    content_hash: content_hash(sub),
                    positional_id: format!("{pos}.{}", j + 1),
                });
            }
        }
    }
    out
}

/// Pack a large message's lines into ordered sub-chunks each ~<= `target` tokens, never splitting
/// mid-line (a single over-long line becomes its own sub-chunk).
fn pack_lines(msg: &str, target: usize) -> Vec<String> {
    let mut subs: Vec<String> = Vec::new();
    let mut cur = String::new();
    for line in msg.split_inclusive('\n') {
        if !cur.is_empty() && crate::inject::est_tokens(&(cur.clone() + line)) > target {
            subs.push(std::mem::take(&mut cur));
        }
        cur.push_str(line);
    }
    if !cur.is_empty() {
        subs.push(cur);
    }
    if subs.is_empty() {
        subs.push(msg.to_string());
    }
    subs
}

/// Persistent chunk-vector store. Construction, schema, reconcile, and the CRUD/KNN methods below
/// are IMPLEMENTED BY the chunks subagent (this is the contract skeleton).
pub struct ChunkStore {
    emb: Arc<dyn Embedder>,
    conn: Connection,
    /// the `vec_chunks` embedding column (shared sqlite-vec shim)
    vec: SqliteVecColumn,
    /// content_hash -> rowid (the exact tier), like `SemanticCache.exact`.
    exact: HashMap<String, i64>,
    max_entries: usize,
    clock: u64,
    /// most-recent clock ticks protected from eviction; see [`EVICT_GUARD_BAND`].
    evict_guard: u64,
    /// eviction counter, surfaced on /stack
    pub evictions: u64,
    /// failed-insert counter: a per-request insert that errored instead of aborting
    /// the process. Surfaced on /stack so a full disk is visible rather than silent.
    pub insert_failures: u64,
    /// content-hash collisions actually observed: a hash hit whose stored raw text did
    /// not match the text being inserted. Expected to stay 0 forever — FNV-1a-64's accidental
    /// birthday probability at the shipped cap is ~7e-11 — which is exactly why a non-zero value is
    /// worth surfacing: it means either a deliberate collision or a much worse bug.
    pub hash_collisions: u64,
    /// Content hashes whose summary is being written right now, and the signal raised when one
    /// finishes. Lets a request that needs a piece the background pre-warm is already summarizing
    /// wait for that result instead of summarizing it a second time on the same model slot
    ///. Held in memory only; managed by `overflow::ensure_summary`.
    pub(crate) summarizing: HashSet<String>,
    pub(crate) summarized: Arc<tokio::sync::Notify>,
}

/// Seconds since the Unix epoch, or 0 if the clock reads before it.
///
/// Used only for retention, never for ordering — `ts` and `last_used` stay the logical
/// clock, which cannot jump backwards when the wall clock does.
fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

impl ChunkStore {
    /// Open (or create) the store. `db_path=None` -> in-memory. `tuned` applies the same WAL
    /// PRAGMAs as `SemanticCache`. Creates `chunks`/`vec_chunks`/`chunk_meta`, reconciles the
    /// vector table against the embedder dim (rebuilding from stored text on change), and loads
    /// the exact-hash tier + clock. Mirror `SemanticCache::new` + `init`.
    pub fn new(
        emb: Arc<dyn Embedder>,
        db_path: Option<&str>,
        tuned: bool,
        max_entries: usize,
    ) -> Self {
        let conn = vecstore::open_conn(db_path, tuned, "disk chunk store");
        let vec = SqliteVecColumn::new("vec_chunks", "chunk_meta", emb.dim());
        let mut c =
            Self { emb, conn, vec, exact: HashMap::new(), max_entries, clock: 0,
                   evict_guard: EVICT_GUARD_BAND, evictions: 0, insert_failures: 0,
                   hash_collisions: 0, summarizing: HashSet::new(),
                   summarized: Arc::new(tokio::sync::Notify::new()) };
        c.init(db_path);
        c
    }

    /// The embedder, so a caller can embed new chunks without holding the store (see
    /// [`Self::get_or_insert_embedded`]).
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        self.emb.clone()
    }

    /// Whether a chunk with this content hash is stored. No SQL: the exact tier is in memory.
    pub fn contains(&self, content_hash: &str) -> bool {
        self.exact.contains_key(content_hash)
    }

    /// Override the eviction guard band. Production keeps the default; tests set it to
    /// 0 to assert the raw LRU cap, or to a small value to exercise the in-flight protection cheaply.
    pub fn set_evict_guard(&mut self, band: u64) {
        self.evict_guard = band;
    }

    /// Create tables (if needed), reconcile the vector table against the current embedder's
    /// dim (re-embedding all rows' RAW text if the embedder changed), and load the exact-hash
    /// tier + clock.
    fn init(&mut self, db_path: Option<&str>) {
        self.conn
            .execute(
                "CREATE TABLE IF NOT EXISTS chunks (rowid INTEGER PRIMARY KEY, content_hash TEXT UNIQUE, \
                 raw TEXT, summary TEXT, ts INTEGER, hits INTEGER DEFAULT 0, \
                 last_used INTEGER DEFAULT 0)",
                [],
            )
            .expect("create chunks table");

        // Additive migration (SSM substrate): a raw-token-ids BLOB slot for a future frozen-SSM
        // read head. Left NULL today — the store keeps raw text only. Idempotent.
        let has_token_ids: bool = {
            let mut stmt = self.conn.prepare("PRAGMA table_info(chunks)").expect("pragma table_info");
            let cols = stmt
                .query_map([], |r| r.get::<_, String>(1))
                .expect("query columns")
                .filter_map(|x| x.ok())
                .any(|name| name == "token_ids");
            cols
        };
        if !has_token_ids {
            self.conn
                .execute("ALTER TABLE chunks ADD COLUMN token_ids BLOB", [])
                .expect("add token_ids column");
        }

        // Additive migration: wall-clock insert time, so retention can be expressed in
        // DAYS. `ts` and `last_used` are a logical counter — fine for LRU ordering, useless for "drop
        // anything older than a week", which is the control an operator actually wants over a file
        // holding their conversations. Existing rows get NULL and are treated as ageless until they
        // are next touched: a migration must not silently delete data on first open.
        let has_wall_ts: bool = {
            let mut stmt = self.conn.prepare("PRAGMA table_info(chunks)").expect("pragma table_info");
            let cols = stmt
                .query_map([], |r| r.get::<_, String>(1))
                .expect("query columns")
                .filter_map(|x| x.ok())
                .any(|name| name == "wall_ts");
            cols
        };
        if !has_wall_ts {
            self.conn
                .execute("ALTER TABLE chunks ADD COLUMN wall_ts INTEGER", [])
                .expect("add wall_ts column");
        }

        self.conn
            .execute("CREATE TABLE IF NOT EXISTS chunk_meta (k TEXT PRIMARY KEY, v TEXT)", [])
            .expect("create chunk_meta table");

        self.vec.reconcile(&self.conn, self.emb.as_ref(), "chunks", "raw", "chunks");

        // Load the clock from the high-water mark of BOTH stamps. `last_used` is bumped to
        // the current clock on every cache hit, so it routinely exceeds the row's insert `ts`;
        // resuming from `MAX(ts)` alone would restart the clock BELOW rows that were hot before the
        // restart. Those rows would then sort as the newest in the LRU and sit permanently inside
        // the eviction guard band, while freshly inserted chunks aged out first.
        let rows: Vec<(String, i64, u64)> = {
            let mut stmt = self
                .conn
                .prepare("SELECT content_hash, rowid, MAX(ts, last_used) FROM chunks")
                .expect("prepare");
            let it = stmt
                .query_map([], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)? as u64))
                })
                .expect("query chunks rows");
            it.filter_map(|x| x.ok()).collect()
        };
        let n = rows.len();
        for (hash, rowid, stamp) in rows {
            self.exact.insert(hash, rowid);
            self.clock = self.clock.max(stamp);
        }
        if let Some(path) = db_path {
            println!("memory: loaded {n} chunks from {path}");
        }
    }

    /// Number of stored chunks (exact-tier len).
    pub fn len(&self) -> usize {
        self.exact.len()
    }

    /// Look up `raw` by its content hash. If a usable (non-empty) cached digest exists, bump
    /// `last_used`/`hits` and return `(hash, Some(summary))`. Otherwise insert the row (raw,
    /// empty summary), **embed the RAW text into `vec_chunks` immediately** (so `search` works
    /// before the chunk is digested — this keeps targeted digests lazy), and return `(hash, None)`
    /// so the caller digests it. Applies the `max_entries` LRU cap after a new insert.
    pub fn get_or_insert(&mut self, raw: &str) -> (String, Option<String>) {
        self.get_or_insert_embedded(raw, None)
    }

    /// [`Self::get_or_insert`] with the embedding already computed. Embedding is the slow part of
    /// an insert (a model forward pass per chunk), so callers holding the store behind a lock embed
    /// new chunks FIRST, outside it, and pass the vectors in here — see `overflow::sync_chunks`.
    /// `None` embeds inline as before.
    pub fn get_or_insert_embedded(&mut self, raw: &str, embedding: Option<Vec<f32>>) -> (String, Option<String>) {
        let h = content_hash(raw);
        self.clock += 1;
        if let Some(&rowid) = self.exact.get(&h) {
            let _ = self.conn.execute(
                "UPDATE chunks SET hits = hits + 1, last_used = ?1 WHERE rowid = ?2",
                rusqlite::params![self.clock as i64, rowid],
            );
            // Read the stored raw alongside the summary and VERIFY it. Same row, same
            // query — the check is a string compare, not an extra round-trip.
            //
            // `content_hash` is FNV-1a-64, which is not collision-resistant. an earlier review refuted the
            // claim that a collision panics the insert and recorded the real consequence: two
            // different chunks sharing a hash means the second one is served the FIRST one's digest,
            // silently, as an authoritative note about text it never contained.
            //
            // Accidental collision is negligible — at the 50k-entry cap the birthday probability is
            // about 7e-11 — so this is not a bug being fixed, it is an assumption being made
            // checkable. The finding's actual concern is a DELIBERATE collision, which sits under
            // the cooperating-operator assumption today and becomes reachable the moment that is
            // relaxed (multi-tenant, or a store fed by untrusted input).
            //
            // Verifying beats rehashing. Switching to a cryptographic hash would rekey every stored
            // chunk and every fixture keyed on one, for a failure mode this catches at the point it
            // would do damage. On mismatch: count it, and return no summary so the caller digests
            // the text in front of it rather than trusting a note about different text.
            let row: Option<(String, String)> = self
                .conn
                .query_row(
                    "SELECT summary, raw FROM chunks WHERE rowid = ?1",
                    rusqlite::params![rowid],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
                )
                .ok();
            if let Some((_, stored_raw)) = &row {
                if stored_raw != raw {
                    self.hash_collisions += 1;
                    return (h, None);
                }
            }
            let summary = row.map(|(s, _)| s).filter(|s| !s.trim().is_empty());
            return (h, summary);
        }

        let ts = self.clock as i64;
        // Per-request path: a failed insert must NOT abort the process. Under
        // `panic = "abort"` an `.expect` here would take the whole server down for every client on
        // any transient sqlite error (e.g. a full disk). Count it and return `(h, None)` — the chunk
        // simply isn't persisted this turn; the caller degrades to no digest rather than crashing.
        if self
            .conn
            .execute(
                "INSERT INTO chunks(content_hash, raw, summary, ts, hits, last_used, wall_ts) \
                 VALUES (?1, ?2, '', ?3, 0, ?3, ?4)",
                rusqlite::params![h, raw, ts, now_secs()],
            )
            .is_err()
        {
            self.insert_failures += 1;
            return (h, None);
        }
        let rowid = self.conn.last_insert_rowid();

        let v = embedding.unwrap_or_else(|| self.emb.encode(raw));
        if !self.vec.try_insert(&self.conn, rowid, &v) {
            // The row exists but has no embedding, so it can never be retrieved. Leaving
            // it registered in `exact` would make the next `get_or_insert` report a cache hit and
            // the chunk would stay invisible forever; rolling it back means the next turn retries.
            self.insert_failures += 1;
            let _ = self.conn.execute("DELETE FROM chunks WHERE rowid=?1", rusqlite::params![rowid]);
            return (h, None);
        }
        self.exact.insert(h.clone(), rowid);

        if self.max_entries > 0 && self.len() > self.max_entries {
            let low_water = (self.max_entries * 9 / 10).max(1);
            self.evict_to(low_water);
        }

        (h, None)
    }

    /// Store the digest text for `content_hash` (`UPDATE chunks SET summary=?`). Does NOT
    /// re-embed — the embedding was set from the raw at `get_or_insert` time.
    pub fn set_summary(&mut self, content_hash: &str, summary: &str) {
        let _ = self.conn.execute(
            "UPDATE chunks SET summary=?1 WHERE content_hash=?2",
            rusqlite::params![summary, content_hash],
        );
    }

    /// KNN over chunk embeddings (embedded from the raw at insert time); returns up to `k` hits
    /// (summary + raw + cosine score), sorted by score desc. Does NOT mutate counters. Mirror
    /// `SemanticCache::candidates`.
    ///
    /// STORE-WIDE: this searches every chunk ever persisted, across all conversations that have
    /// shared this process. It is correct for single-conversation callers (benches, tests) but MUST
    /// NOT be reached from a per-request path — one request seeing another's chunks is a
    /// cross-conversation leak. Per-request code calls [`Self::search_scoped`].
    ///
    /// NAMED FOR WHAT IT DOES, NOT FOR WHAT IT IS. This was `search`, and the leak it
    /// enables recurred at three separate call sites — because the dangerous method had the
    /// obvious name and the safe one had the qualified name, so reaching for the wrong one was the
    /// path of least resistance and a doc comment was the only thing saying otherwise. A caller now
    /// has to write `unscoped` to get unscoped behaviour. As of this rename it has no caller in the
    /// server path at all; every remaining one is a bench or a test.
    pub fn search_unscoped(&mut self, query: &str, k: usize) -> Vec<ChunkHit> {
        let qv = self.emb.encode(query);
        let knn = self.vec.knn(&self.conn, &qv, k);

        let meta: HashMap<i64, (String, String, String)> = if knn.is_empty() {
            HashMap::new()
        } else {
            let placeholders = std::iter::repeat("?").take(knn.len()).collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT rowid, content_hash, raw, summary FROM chunks WHERE rowid IN ({placeholders})"
            );
            let mut stmt = self.conn.prepare(&sql).expect("prepare chunk metadata query");
            let it = stmt
                .query_map(rusqlite::params_from_iter(knn.iter().map(|(rowid, _)| *rowid)), |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                    ))
                })
                .expect("query chunk metadata rows");
            it.filter_map(|x| x.ok())
                .map(|(rowid, hash, raw, summary)| (rowid, (hash, raw, summary)))
                .collect()
        };

        let mut out: Vec<ChunkHit> = Vec::new();
        for (rowid, distance) in knn {
            let Some((content_hash, raw, summary)) = meta.get(&rowid).cloned() else { continue };
            let score = (1.0 - distance).clamp(0.0, 1.0);
            out.push(ChunkHit { content_hash, summary, raw, score });
        }
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Search restricted to `in_scope` (a set of content hashes — the CURRENT request's chunks).
    /// The only search a per-request tool loop may call: another conversation's chunk is never a
    /// candidate at all, which is how the cross-conversation leak is closed. Empty `in_scope` -> no hits.
    ///
    /// EXACT, not approximate. `in_scope` is small and known (one request's own chunks — a
    /// 200k-token conversation is ~130 of them), so this scores every candidate directly against
    /// the stored vectors instead of asking the `vec0` index for neighbors and discarding the ones
    /// that don't belong. That matters: `vec0` cannot filter by metadata, so the earlier
    /// KNN-then-filter shape returned only whatever survived a *global* top-N — and as the shared
    /// store fills with other conversations' chunks, a request's own chunks get crowded out of that
    /// pool and the tool reports "(no matches)" for text sitting right there. Scoring the scope
    /// directly costs O(|in_scope| x dim) — nothing beside one core round-trip — and cannot decay
    /// with store size.
    pub fn search_scoped(&mut self, query: &str, k: usize, in_scope: &HashSet<String>) -> Vec<ChunkHit> {
        if in_scope.is_empty() || k == 0 {
            return Vec::new();
        }
        // The exact tier already maps content_hash -> rowid in memory, so resolving the scope costs
        // no SQL. A hash with no row (never inserted, or evicted) simply isn't a candidate.
        let scope: Vec<(String, i64)> = in_scope
            .iter()
            .filter_map(|h| self.exact.get(h).map(|&rowid| (h.clone(), rowid)))
            .collect();
        if scope.is_empty() {
            return Vec::new();
        }

        // One batched read for the text of every candidate (mirrors `search`'s metadata query).
        let placeholders = std::iter::repeat("?").take(scope.len()).collect::<Vec<_>>().join(",");
        let sql = format!("SELECT rowid, raw, summary FROM chunks WHERE rowid IN ({placeholders})");
        let text: HashMap<i64, (String, String)> = {
            let Ok(mut stmt) = self.conn.prepare(&sql) else { return Vec::new() };
            let Ok(it) = stmt.query_map(
                rusqlite::params_from_iter(scope.iter().map(|(_, rowid)| *rowid)),
                |r| Ok((r.get::<_, i64>(0)?, (r.get::<_, String>(1)?, r.get::<_, String>(2)?))),
            ) else {
                return Vec::new();
            };
            it.filter_map(|x| x.ok()).collect()
        };

        let qv = self.emb.encode(query);
        let mut hits: Vec<ChunkHit> = Vec::with_capacity(scope.len());
        for (content_hash, rowid) in scope {
            // A row whose vector insert failed has no embedding and is simply not retrievable.
            let Some(v) = self.vec.embedding_at(&self.conn, rowid) else { continue };
            let Some((raw, summary)) = text.get(&rowid).cloned() else { continue };
            let score = cosine(&qv, &v).clamp(0.0, 1.0);
            hits.push(ChunkHit { content_hash, summary, raw, score });
        }
        // Tie-break on the hash so the ordering is deterministic — `in_scope` is a HashSet and its
        // iteration order is not.
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.content_hash.cmp(&b.content_hash))
        });
        hits.truncate(k);
        hits
    }

    /// Fetch `(raw, summary)` by content hash.
    pub fn get(&self, content_hash: &str) -> Option<(String, String)> {
        self.conn
            .query_row(
                "SELECT raw, summary FROM chunks WHERE content_hash=?1",
                rusqlite::params![content_hash],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )
            .ok()
    }

    /// Delete chunks older than `ttl_days`, returning how many went.
    ///
    /// WHY THIS EXISTS. The store's only bound was the LRU cap, which is a size limit and not a
    /// retention policy: a conversation stays on disk until enough other traffic pushes it out, so on
    /// a quiet instance "recent" can mean months. Conversation text and full tool outputs — file
    /// contents, command output — sit there in plaintext. An operator asking "how long do you keep
    /// this?" had no answer but "until the file fills up".
    ///
    /// Age is measured from INSERT (`wall_ts`), not from last use. Retention answers "how long may
    /// this text exist"; refreshing the clock on every recall would let a frequently hit chunk live
    /// forever, which is the opposite of what a retention limit is for.
    ///
    /// Rows written before the `wall_ts` migration hold NULL and are never expired by age. That is
    /// deliberate — a migration that deletes data the first time it runs is a data-loss bug wearing
    /// a feature's clothes. They still evict by LRU, and everything written since is stamped.
    ///
    /// No guard band here, unlike [`Self::evict_to`]. That band exists because eviction by SIZE is
    /// triggered by one request's insert and can reach a concurrent request's just-written chunks;
    /// expiry by AGE cannot, since a chunk written moments ago is not days old.
    pub fn evict_older_than(&mut self, ttl_days: u64) -> usize {
        if ttl_days == 0 {
            return 0;
        }
        let cutoff = now_secs() - (ttl_days as i64).saturating_mul(86_400);
        let rows: Vec<i64> = {
            let Ok(mut stmt) = self
                .conn
                .prepare("SELECT rowid FROM chunks WHERE wall_ts IS NOT NULL AND wall_ts < ?1")
            else {
                return 0;
            };
            let Ok(mapped) = stmt.query_map(rusqlite::params![cutoff], |r| r.get::<_, i64>(0))
            else {
                return 0;
            };
            let ids: Vec<i64> = mapped.filter_map(|x| x.ok()).collect();
            ids
        };
        if rows.is_empty() {
            return 0;
        }
        for rowid in &rows {
            let _ = self
                .conn
                .execute("DELETE FROM chunks WHERE rowid=?1", rusqlite::params![rowid]);
            self.vec.delete(&self.conn, *rowid);
        }
        let rm: std::collections::HashSet<i64> = rows.iter().copied().collect();
        self.exact.retain(|_, &mut r| !rm.contains(&r));
        self.evictions += rows.len() as u64;
        rows.len()
    }

    /// Evict least-recently-used chunks until `len() <= target`; returns count deleted and adds
    /// to `self.evictions`. Mirror the delete-from-all-three-places pattern in `cache.rs`.
    ///
    /// GUARD: rows stamped within the most recent [`EVICT_GUARD_BAND`] clock ticks are
    /// never evicted. Eviction is triggered by one request's insert, but the store is shared, so the
    /// oldest-by-`last_used` rows can belong to a DIFFERENT request that inserted them moments ago
    /// and has not yet read them back in its tool loop. The band protects any recently-touched chunk
    /// regardless of which request owns it. For a healthily-sized cap the protected (recent) rows are
    /// never among the oldest-to-delete anyway, so the guard only bites when the cap is set small —
    /// which is exactly the case that would otherwise corrupt a concurrent request's answer. When it
    /// bites, we evict what we can and leave the store above `target` rather than evict in-use rows.
    pub fn evict_to(&mut self, target: usize) -> usize {
        if self.len() <= target {
            return 0;
        }
        let to_delete = self.len() - target;
        // Ask SQLite for just the oldest `to_delete` rows rather than scanning and sorting the whole
        // table in memory on every insert past the cap — at the default 50k cap that was a
        // 50k-row scan per new chunk in steady state. Rows come back oldest-first, which is what the
        // guard-band break below relies on.
        let rows: Vec<(i64, u64)> = {
            let mut stmt = self
                .conn
                .prepare("SELECT rowid, last_used FROM chunks ORDER BY last_used ASC LIMIT ?1")
                .expect("prepare");
            let it = stmt
                .query_map(rusqlite::params![to_delete as i64], |r| {
                    Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)? as u64))
                })
                .expect("query rows");
            it.filter_map(|x| x.ok()).collect()
        };

        let protect_floor = self.clock.saturating_sub(self.evict_guard);
        let mut removed: Vec<i64> = Vec::with_capacity(to_delete);
        for (rowid, last_used) in rows.into_iter() {
            if removed.len() >= to_delete {
                break;
            }
            // Never evict a recently-touched row — it may be an in-flight request's just-inserted,
            // not-yet-read chunk. Rows are sorted oldest-first, so once we reach the protected band
            // every remaining row is protected too.
            if last_used >= protect_floor {
                break;
            }
            let _ = self.conn.execute("DELETE FROM chunks WHERE rowid=?1", rusqlite::params![rowid]);
            self.vec.delete(&self.conn, rowid);
            removed.push(rowid);
        }
        // Prune the exact-hash tier in ONE pass instead of an O(map) retain per deleted row.
        if !removed.is_empty() {
            let rm: std::collections::HashSet<i64> = removed.iter().copied().collect();
            self.exact.retain(|_, &mut r| !rm.contains(&r));
        }
        let deleted = removed.len();
        self.evictions += deleted as u64;
        deleted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashingEmbedder;

    fn temp_path(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("xzo_test_chunks_{}_{}.sqlite", name, std::process::id()))
            .to_str()
            .unwrap()
            .to_string()
    }

    fn cleanup(p: &str) {
        let _ = std::fs::remove_file(p);
        let _ = std::fs::remove_file(format!("{p}-wal"));
        let _ = std::fs::remove_file(format!("{p}-shm"));
    }

    fn store() -> ChunkStore {
        ChunkStore::new(Arc::new(HashingEmbedder::new()), None, false, 0)
    }

    /// a request must never retrieve a chunk it does not own. `search_scoped` filters
    /// inside the store, so conversation B searching a term that only conversation A stored gets
    /// nothing back — not A's text under a `?` label.
    #[test]
    fn search_scoped_hides_out_of_scope_chunks() {
        let mut s = store();
        // Conversation A stores a distinctive secret.
        let (a_hash, _) = s.get_or_insert("SECRET_TOKEN_zamboni is A's private value");
        // Conversation B owns an unrelated chunk; its in_scope set excludes A's hash.
        let (b_hash, _) = s.get_or_insert("B talks about the weather and nothing else");
        let b_scope: HashSet<String> = [b_hash].into_iter().collect();

        // B searches for A's secret. Store-wide `search` would surface it; scoped must not.
        let leaked = s.search_unscoped("SECRET_TOKEN_zamboni", 5);
        assert!(
            leaked.iter().any(|h| h.content_hash == a_hash),
            "sanity: unscoped search does surface A's chunk, so the scoped test is meaningful",
        );
        let scoped = s.search_scoped("SECRET_TOKEN_zamboni", 5, &b_scope);
        assert!(
            scoped.iter().all(|h| h.content_hash != a_hash),
            "scoped search leaked A's out-of-scope chunk into B",
        );
    }

    /// The other half: scoping must not be a trivial "return nothing". An in-scope chunk that
    /// matches the query is still returned — otherwise the leak test above passes for the wrong
    /// reason (a disabled tool leaks nothing but is also useless).
    #[test]
    fn search_scoped_still_returns_in_scope_hits() {
        let mut s = store();
        let (hash, _) = s.get_or_insert("the deploy runbook lives at ops/deploy.md step four");
        let scope: HashSet<String> = [hash.clone()].into_iter().collect();
        let hits = s.search_scoped("deploy runbook", 5, &scope);
        assert!(
            hits.iter().any(|h| h.content_hash == hash),
            "scoped search dropped an in-scope hit it should have kept",
        );
    }

    /// scoped search must not decay as the shared store fills with other conversations'
    /// chunks. The old KNN-then-filter shape asked the index for a global top-N and kept whatever
    /// survived; here the in-scope chunk is deliberately ranked BELOW hundreds of foreign chunks
    /// for the query, so a pooled search would return nothing and the tool would report
    /// "(no matches)" for text sitting in the store. An exact scoped search still finds it.
    #[test]
    fn search_scoped_survives_a_store_full_of_other_conversations() {
        let mut s = store();
        // The chunk this request owns. Its match on the query is real but weak.
        let (mine, _) = s.get_or_insert("build notes: the fallback timeout knob is tuned quarterly");
        let scope: HashSet<String> = [mine.clone()].into_iter().collect();

        // 600 foreign chunks that all match the query far more strongly than the in-scope one —
        // comfortably more than any pool the old implementation over-fetched (max(8k, 64)).
        for i in 0..600 {
            s.get_or_insert(&format!(
                "foreign conversation {i}: fallback timeout fallback timeout retry budget {i}"
            ));
        }

        let hits = s.search_scoped("fallback timeout", 5, &scope);
        assert!(
            hits.iter().any(|h| h.content_hash == mine),
            "scoped search lost the request's own chunk once the store filled with other \
             conversations — retrieval quality must not depend on store size",
        );
        assert!(
            hits.iter().all(|h| h.content_hash == mine),
            "scoped search returned a chunk outside the scope",
        );
    }

    /// An empty scope (no chunks owned) can never return anything, regardless of the query.
    #[test]
    fn search_scoped_empty_scope_returns_nothing() {
        let mut s = store();
        s.get_or_insert("some stored chunk that exists in the store");
        assert!(s.search_scoped("stored chunk", 5, &HashSet::new()).is_empty());
    }

    #[test]
    fn split_is_per_message_boundary() {
        let msgs = vec!["m1".to_string(), "m2".to_string(), "m3".to_string()];
        let chunks = split_messages(&msgs, 100000);
        assert_eq!(chunks.len(), 3);
        let ids: Vec<&str> = chunks.iter().map(|c| c.positional_id.as_str()).collect();
        assert_eq!(ids, vec!["1", "2", "3"]);
    }

    #[test]
    fn oversized_message_subsplits_stably() {
        let lines: Vec<String> = (0..12).map(|i| format!("line{i}\n")).collect();
        let a = lines.concat();

        let subs_alone = split_messages(&[a.clone()], 5);
        assert!(subs_alone.len() > 1, "expected the oversized message to sub-split");
        for (i, c) in subs_alone.iter().enumerate() {
            assert_eq!(c.positional_id, format!("1.{}", i + 1));
        }
        let hashes_alone: Vec<String> = subs_alone.iter().map(|c| c.content_hash.clone()).collect();

        let b = "an unrelated second message".to_string();
        let subs_with_b = split_messages(&[a.clone(), b], 5);
        let a_hashes_in_combined: Vec<String> = subs_with_b
            .iter()
            .take(subs_alone.len())
            .map(|c| c.content_hash.clone())
            .collect();
        assert_eq!(hashes_alone, a_hashes_in_combined);
    }

    /// The splitter's key and the store's key must agree for EVERY chunk, sub-chunks included —
    /// otherwise a summary is written to no row and the part is re-summarized on every turn.
    #[test]
    fn every_chunk_is_filed_under_the_key_it_is_looked_up_by() {
        let big: String = (0..40).map(|i| format!("log line {i}: something happened\n")).collect();
        let chunks = split_messages(&[big, "a small message".to_string()], 20);
        assert!(chunks.iter().any(|c| c.positional_id.contains('.')), "expected a sub-split");
        let mut s = store();
        for c in &chunks {
            let (h, _) = s.get_or_insert(&c.raw);
            assert_eq!(h, c.content_hash, "chunk {} is filed under a different key", c.positional_id);
            s.set_summary(&c.content_hash, "a summary");
            assert_eq!(s.get(&c.content_hash).map(|(_, sum)| sum).as_deref(), Some("a summary"));
        }
    }

    #[test]
    fn append_message_leaves_prior_chunk_hashes_unchanged() {
        let a = "alpha message".to_string();
        let b = "beta message".to_string();
        let c = "gamma message".to_string();

        let ab = split_messages(&[a.clone(), b.clone()], 100000);
        let abc = split_messages(&[a, b, c], 100000);

        let ab_hashes: Vec<&str> = ab.iter().map(|x| x.content_hash.as_str()).collect();
        let abc_prefix_hashes: Vec<&str> =
            abc.iter().take(ab.len()).map(|x| x.content_hash.as_str()).collect();
        assert_eq!(ab_hashes, abc_prefix_hashes);
    }

    #[test]
    fn edit_one_message_redigests_only_that_chunk() {
        let mut s = store();
        let a = "message a content".to_string();
        let b = "message b content".to_string();
        let c = "message c content".to_string();

        let chunks = split_messages(&[a.clone(), b.clone(), c.clone()], 100000);
        for chunk in &chunks {
            let (h, _) = s.get_or_insert(&chunk.raw);
            s.set_summary(&h, "digest");
        }

        let b2 = "message b content EDITED".to_string();
        let new_chunks = split_messages(&[a, b2, c], 100000);
        let mut misses = 0;
        for chunk in &new_chunks {
            let (_, cached) = s.get_or_insert(&chunk.raw);
            if cached.is_none() {
                misses += 1;
            } else {
                assert_eq!(cached.as_deref(), Some("digest"));
            }
        }
        assert_eq!(misses, 1, "expected exactly one chunk (the edited one) to miss the digest cache");
    }

    #[test]
    fn unchanged_chunk_hits_digest_cache() {
        let mut s = store();
        let (h, cached) = s.get_or_insert("x");
        assert!(cached.is_none());
        s.set_summary(&h, "d");
        let (_, cached2) = s.get_or_insert("x");
        assert_eq!(cached2.as_deref(), Some("d"));
    }

    #[test]
    fn changed_chunk_misses_digest_cache() {
        let mut s = store();
        let (h, _) = s.get_or_insert("x");
        s.set_summary(&h, "d");
        let (_, cached) = s.get_or_insert("y");
        assert!(cached.is_none());
    }

    #[test]
    fn search_returns_nearest_chunk() {
        let mut s = store();
        let (h1, _) = s.get_or_insert("apple banana cherry fruit salad recipe");
        s.set_summary(&h1, "fruit chunk");
        let (h2, _) = s.get_or_insert("rocket propulsion aerospace engineering thrust");
        s.set_summary(&h2, "rocket chunk");
        let (h3, _) = s.get_or_insert("medieval castle architecture stone walls");
        s.set_summary(&h3, "castle chunk");

        let hits = s.search_unscoped("rocket propulsion aerospace thrust engine", 3);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].content_hash, h2, "expected the rocket chunk to be the nearest hit");
    }

    #[test]
    fn get_chunk_returns_exact_raw_bytes() {
        let mut s = store();
        let raw = "unicode: caf\u{e9} \u{1f600}\nmulti\nline\ntext\t\ttabbed";
        let (h, _) = s.get_or_insert(raw);
        let (got_raw, _) = s.get(&h).expect("chunk should exist");
        assert_eq!(got_raw, raw);
    }

    #[test]
    fn chunk_store_evicts_at_max_entries() {
        let mut s = ChunkStore::new(Arc::new(HashingEmbedder::new()), None, false, 5);
        // Guard off: this test pins the raw LRU cap. (With the default guard, all 10 inserts fall
        // inside the protected band and none would evict — see the in-flight eviction guard test below.)
        s.set_evict_guard(0);
        for i in 0..10 {
            s.get_or_insert(&format!("distinct chunk number {i}"));
        }
        assert!(s.len() <= 5, "expected eviction to cap len at max_entries, got {}", s.len());
    }

    /// the LRU must not evict a row a concurrent request just inserted and has not yet
    /// read back. With the guard protecting the most-recent `band` ticks, chunks inserted within the
    /// band survive even when the cap is exceeded — the store grows past the cap rather than corrupt
    /// an in-flight answer.
    #[test]
    fn evict_protects_recently_inserted_chunks() {
        let mut s = ChunkStore::new(Arc::new(HashingEmbedder::new()), None, false, 5);
        s.set_evict_guard(100); // protect the last 100 ticks — all inserts here are inside it
        let mut hashes = Vec::new();
        for i in 0..10 {
            let (h, _) = s.get_or_insert(&format!("in-flight chunk number {i}"));
            hashes.push(h);
        }
        // Nothing aged past the guard, so despite exceeding the cap of 5, every chunk is still here.
        assert_eq!(s.evictions, 0, "guard should have blocked all eviction");
        for h in &hashes {
            assert!(s.get(h).is_some(), "an in-flight chunk was evicted despite the guard");
        }
    }

    /// The guard protects only the RECENT band: once rows age past it, they evict normally, so the
    /// cap still bounds the store in steady state (it is not a permanent growth leak).
    #[test]
    fn evict_reclaims_rows_older_than_the_guard() {
        let mut s = ChunkStore::new(Arc::new(HashingEmbedder::new()), None, false, 5);
        s.set_evict_guard(3); // only the last 3 ticks are protected
        for i in 0..20 {
            s.get_or_insert(&format!("aging chunk number {i}"));
        }
        // Old rows (well past the 3-tick band) were reclaimed, so the store stayed bounded and did
        // evict — while still never touching the newest few.
        assert!(s.evictions > 0, "expected aged rows to evict");
        assert!(s.len() < 20, "store should be bounded, got {}", s.len());
    }

    #[test]
    fn chunk_store_persists_across_restart() {
        let p = temp_path("persists_across_restart");
        cleanup(&p);
        {
            let mut s = ChunkStore::new(Arc::new(HashingEmbedder::new()), Some(&p), false, 0);
            let (h, _) = s.get_or_insert("persisted raw content");
            s.set_summary(&h, "persisted digest");
            assert_eq!(s.len(), 1);
        }
        let mut s2 = ChunkStore::new(Arc::new(HashingEmbedder::new()), Some(&p), false, 0);
        assert_eq!(s2.len(), 1);
        let (h2, cached) = s2.get_or_insert("persisted raw content");
        assert_eq!(cached.as_deref(), Some("persisted digest"));
        let (raw, summary) = s2.get(&h2).expect("chunk should exist after restart");
        assert_eq!(raw, "persisted raw content");
        assert_eq!(summary, "persisted digest");
        let hits = s2.search_unscoped("persisted raw content", 3);
        assert!(!hits.is_empty());
        cleanup(&p);
    }

    /// the clock must resume above every stamp in the store, not just above `MAX(ts)`.
    /// A row hit repeatedly before a restart carries `last_used` well past its insert `ts`; if the
    /// clock resumed from `ts` alone, that row would outrank everything inserted afterwards in the
    /// LRU and chunks from the new session would be evicted first.
    #[test]
    fn clock_resumes_above_last_used_not_just_ts() {
        let p = temp_path("clock_resumes_above_last_used");
        cleanup(&p);
        let hot = "a chunk that gets hit over and over";
        {
            let mut s = ChunkStore::new(Arc::new(HashingEmbedder::new()), Some(&p), false, 0);
            s.get_or_insert(hot); // ts == last_used == 1
            for _ in 0..50 {
                s.get_or_insert(hot); // bumps last_used far past ts
            }
        }

        let mut s2 = ChunkStore::new(Arc::new(HashingEmbedder::new()), Some(&p), false, 0);
        let hot_last_used: u64 = s2
            .conn
            .query_row(
                "SELECT last_used FROM chunks WHERE content_hash=?1",
                rusqlite::params![content_hash(hot)],
                |r| r.get::<_, i64>(0),
            )
            .expect("hot row should exist") as u64;
        s2.get_or_insert("a brand new chunk inserted after the restart");
        let fresh_last_used: u64 = s2
            .conn
            .query_row(
                "SELECT last_used FROM chunks WHERE content_hash=?1",
                rusqlite::params![content_hash("a brand new chunk inserted after the restart")],
                |r| r.get::<_, i64>(0),
            )
            .expect("fresh row should exist") as u64;

        assert!(
            fresh_last_used > hot_last_used,
            "a chunk inserted after the restart ({fresh_last_used}) must be newer than a row that \
             was hot before it ({hot_last_used}) — otherwise the LRU evicts the new session first",
        );
        cleanup(&p);
    }

    #[test]
    fn chunk_store_reconciles_on_dim_change() {
        let p = temp_path("reconciles_on_dim_change");
        cleanup(&p);
        {
            let mut s = ChunkStore::new(Arc::new(HashingEmbedder::with_dim(256)), Some(&p), false, 0);
            s.get_or_insert("dim change chunk content");
        }
        let mut s2 = ChunkStore::new(Arc::new(HashingEmbedder::with_dim(384)), Some(&p), false, 0);
        assert_eq!(s2.len(), 1);
        let hits = s2.search_unscoped("dim change chunk content", 3);
        assert!(!hits.is_empty(), "expected search to work post-reconcile");
        cleanup(&p);
    }

    /// Replaces `retriever_trait_delegates_to_search`, which asserted the back door existed
    ///. What matters is not that some method forwards to some other method — it is that
    /// the two doors are named for what they do and behave differently, so choosing one is a
    /// decision rather than a spelling.
    #[test]
    fn the_two_searches_are_named_for_what_they_do() {
        let mut s = store();
        let (mine, _) = s.get_or_insert("rocket propulsion aerospace engineering thrust");
        let (theirs, _) = s.get_or_insert("SECRET_TOKEN_zamboni belongs to another conversation");
        s.set_summary(&mine, "rocket chunk");

        // Unscoped sees the whole store, which is exactly why it has to say so in its name.
        let wide = s.search_unscoped("SECRET_TOKEN_zamboni", 5);
        assert!(wide.iter().any(|h| h.content_hash == theirs));

        // Scoped sees only what this request owns.
        let scope: HashSet<String> = [mine.clone()].into_iter().collect();
        let narrow = s.search_scoped("SECRET_TOKEN_zamboni", 5, &scope);
        assert!(
            narrow.iter().all(|h| h.content_hash != theirs),
            "scoped search returned a chunk outside the scope set"
        );
    }

    #[test]
    fn token_ids_column_exists_and_defaults_null() {
        let p = temp_path("token_ids_migration");
        cleanup(&p);
        {
            let mut s = ChunkStore::new(Arc::new(HashingEmbedder::new()), Some(&p), false, 0);
            s.get_or_insert("some chunk content");
        }
        // reopen (exercises the migration path on an existing db) and read token_ids
        let s2 = ChunkStore::new(Arc::new(HashingEmbedder::new()), Some(&p), false, 0);
        let val: Option<Vec<u8>> = s2
            .conn
            .query_row("SELECT token_ids FROM chunks LIMIT 1", [], |r| r.get::<_, Option<Vec<u8>>>(0))
            .expect("query token_ids");
        assert!(val.is_none(), "token_ids should default to NULL");
        cleanup(&p);
    }

    // ---------------------------------------------------------------------------------------
    // Retention
    // ---------------------------------------------------------------------------------------

    /// Default off. A retention policy that starts deleting on upgrade is worse than none, so the
    /// zero case must be a hard no-op rather than "a very long TTL".
    #[test]
    fn a_ttl_of_zero_deletes_nothing() {
        let mut s = store();
        s.get_or_insert("a chunk that must survive");
        assert_eq!(s.evict_older_than(0), 0);
        assert_eq!(s.len(), 1);
        assert_eq!(crate::config::Config::default().chunk_ttl_days, 0);
    }

    /// A fresh chunk is not expired by any positive TTL.
    #[test]
    fn a_fresh_chunk_outlives_a_one_day_ttl() {
        let mut s = store();
        s.get_or_insert("written just now");
        assert_eq!(s.evict_older_than(1), 0);
        assert_eq!(s.len(), 1);
    }

    /// The actual expiry, forced by backdating the stamp — the only way to test days without
    /// waiting days. Checks all three tiers, because a row deleted from sqlite but left in the
    /// in-memory exact map would report as a cache hit with no text behind it.
    #[test]
    fn an_old_chunk_is_expired_from_every_tier() {
        let mut s = store();
        let (h, _) = s.get_or_insert("written long ago");
        let ancient = now_secs() - 40 * 86_400;
        s.conn
            .execute("UPDATE chunks SET wall_ts = ?1", rusqlite::params![ancient])
            .expect("backdate");

        assert_eq!(s.evict_older_than(30), 1);
        assert_eq!(s.len(), 0);
        assert!(!s.exact.contains_key(&h), "the exact tier still claims to hold it");
        assert!(
            s.search_unscoped("written long ago", 5).is_empty(),
            "the vector tier still returns the expired chunk"
        );
    }

    /// The migration case, and the one that would be a data-loss bug: rows written before `wall_ts`
    /// existed hold NULL, and must never be expired by age on first open.
    #[test]
    fn rows_predating_the_migration_are_never_expired_by_age() {
        let mut s = store();
        s.get_or_insert("a row from before the column existed");
        s.conn
            .execute("UPDATE chunks SET wall_ts = NULL", [])
            .expect("simulate a pre-migration row");

        assert_eq!(s.evict_older_than(1), 0, "a NULL stamp was treated as ancient");
        assert_eq!(s.len(), 1);
    }

    // ---------------------------------------------------------------------------------------
    // Content-hash collisions
    // ---------------------------------------------------------------------------------------

    /// The failure the hash-collision case names: two different chunks sharing a content hash, where the second is served
    /// the FIRST one's digest as an authoritative note about text it never contained.
    ///
    /// Forced by writing the collision directly, because FNV-1a-64 will not produce one by accident
    /// here — the whole point is that this is unreachable in normal operation and catastrophic if it
    /// ever is reached.
    #[test]
    fn a_hash_collision_is_detected_and_never_serves_the_wrong_digest() {
        let mut s = store();
        let (h, _) = s.get_or_insert("chunk A: the port is 8931");
        s.set_summary(&h, "chunk A mentions port 8931");

        // Same hash, different text — exactly what a collision looks like from the store's side.
        s.conn
            .execute(
                "UPDATE chunks SET raw = ?1 WHERE content_hash = ?2",
                rusqlite::params!["chunk B: something else entirely", h],
            )
            .expect("forge the collision");

        let (got_h, summary) = s.get_or_insert("chunk A: the port is 8931");
        assert_eq!(got_h, h);
        assert!(
            summary.is_none(),
            "served a digest for text the chunk does not contain: {summary:?}"
        );
        assert_eq!(s.hash_collisions, 1, "the collision was not counted");
    }

    /// The ordinary path must be untouched: a genuine hit still returns its own summary, and nothing
    /// is counted. Without this, "detect collisions" could be satisfied by never serving a digest.
    #[test]
    fn a_genuine_hit_still_returns_its_summary() {
        let mut s = store();
        let (h, _) = s.get_or_insert("chunk A: the port is 8931");
        s.set_summary(&h, "chunk A mentions port 8931");

        let (got_h, summary) = s.get_or_insert("chunk A: the port is 8931");
        assert_eq!(got_h, h);
        assert_eq!(summary.as_deref(), Some("chunk A mentions port 8931"));
        assert_eq!(s.hash_collisions, 0);
    }
}
