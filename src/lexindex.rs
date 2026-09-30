// SPDX-License-Identifier: Apache-2.0
//! Keyword search for the memory layer, on SQLite FTS5.
//!
//! WHY. The memory layer carried its own keyword machinery: a hand tokenizer, hand document-frequency
//! counting, and a substring line-matcher. The knowledge library already used SQLite's FTS5 for the
//! same job. Two implementations of one idea drift — the substring matcher is how `all` came to
//! match "small" and `port` to match "report" — and the hand tokenizer re-derived things
//! FTS5 already does (Unicode case folding, diacritics). This module puts memory on FTS5 too.
//!
//! SCOPE: ONE REQUEST, IN MEMORY. The index is built per request over that request's own chunks and
//! dropped with it. Two reasons, both load-bearing:
//!
//!   1. Isolation. Word statistics are part of what ranks results. An index over the shared store
//!      would let one conversation's vocabulary shift another's ranking — no text crosses, but the
//!      behaviour does, which is the cross-conversation-leak class of bug in a quieter form.
//!   2. The walk's IDF is over the request's chunks (N = this request's chunk count). A store-wide
//!      index would silently change N from dozens to tens of thousands and re-tune every score.
//!
//! Nothing on disk changes: no migration, no fourth tier to keep in sync on eviction.
//!
//! TOKENIZER: stock `unicode61`, which splits on punctuation — so `SERVICE_PORT` is `service`,
//! `port`, and a query for "port" finds it. The identifier-preserving alternative (`tokenchars`)
//! keeps `auth-svc` whole but then "port" no longer finds `SERVICE_PORT`, and with `.` as a token
//! character every sentence-final word grows a full stop ("postgres." ≠ "postgres"). Both were
//! measured; this constant is what won.
//!
//! CJK is NOT handled well, and neither was the code this replaces: `unicode61` treats a run of
//! ideographs with no spaces as one token, so a Japanese sentence is a single term. A known limit.

use rusqlite::{params, Connection};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// The FTS5 tokenizer the memory layer uses. See the module note for why it is the stock one.
pub const TOKENIZER: &str = "unicode61 remove_diacritics 2";

/// Quote a term for an FTS5 MATCH expression. Terms come from the tokenizer, so they contain no
/// operators, but quoting is what stops a term like `or`/`near` being read as syntax.
fn quote(term: &str) -> String {
    format!("\"{}\"", term.replace('"', "\"\""))
}

/// A connection with a one-row scratch table for tokenizing arbitrary text through FTS5.
fn scratch_conn(tokenizer: &str) -> rusqlite::Result<Connection> {
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE scratch USING fts5(text, tokenize = \"{tokenizer}\");
         CREATE VIRTUAL TABLE scratch_i USING fts5vocab(scratch, 'instance');"
    ))?;
    Ok(conn)
}

/// Tokenize `text` through FTS5 on `conn`'s scratch table: distinct terms, in first-seen order.
fn scratch_tokens(conn: &Connection, text: &str) -> rusqlite::Result<Vec<String>> {
    conn.execute("INSERT INTO scratch(rowid, text) VALUES (1, ?1)", params![text])?;
    let terms: Vec<String> = {
        let mut stmt = conn.prepare_cached("SELECT term FROM scratch_i ORDER BY offset")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut seen = HashSet::new();
        rows.filter_map(|r| r.ok()).filter(|t| seen.insert(t.clone())).collect()
    };
    conn.execute("DELETE FROM scratch WHERE rowid = 1", [])?;
    Ok(terms)
}

thread_local! {
    /// One tokenizer connection per thread, reused. Opening a connection per call would dominate
    /// the cost of tokenizing a line.
    static TOKENIZE: RefCell<Option<Connection>> = const { RefCell::new(None) };
}

/// Tokenize `text` exactly as the memory layer's FTS5 index would.
///
/// Falls back to the plain alphanumeric splitter if FTS5 is unavailable, because this sits on the
/// per-request path and must not panic. That cannot happen in a normal build — SQLite is bundled
/// with FTS5 — and `fts5_is_compiled_in` pins it, so a build without it fails tests instead of
/// silently degrading.
pub fn tokens(text: &str) -> Vec<String> {
    TOKENIZE.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = scratch_conn(TOKENIZER).ok();
        }
        match slot.as_ref().map(|c| scratch_tokens(c, text)) {
            Some(Ok(t)) => t,
            _ => crate::embed::tokens(text),
        }
    })
}

/// Tokenize many lines in one round trip: one insert per non-empty line, one read, one delete,
/// inside a transaction. The line grep runs this over every line of a chunk, so doing it per line
/// would be three statements per line instead of three per chunk.
pub fn tokens_many(lines: &[&str]) -> Vec<Vec<String>> {
    let fallback = || lines.iter().map(|l| crate::embed::tokens(l)).collect();
    TOKENIZE.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = scratch_conn(TOKENIZER).ok();
        }
        let Some(conn) = slot.as_ref() else {
            return fallback();
        };
        let run = || -> rusqlite::Result<Vec<Vec<String>>> {
            let tx = conn.unchecked_transaction()?;
            {
                let mut ins = tx.prepare_cached("INSERT INTO scratch(rowid, text) VALUES (?1, ?2)")?;
                for (i, line) in lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        ins.execute(params![(i + 1) as i64, line])?;
                    }
                }
            }
            let mut out: Vec<Vec<String>> = vec![Vec::new(); lines.len()];
            {
                let mut stmt =
                    tx.prepare_cached("SELECT doc, term FROM scratch_i ORDER BY doc, offset")?;
                let rows =
                    stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
                for (doc, term) in rows.filter_map(|r| r.ok()) {
                    let v = &mut out[(doc - 1) as usize];
                    if !v.contains(&term) {
                        v.push(term);
                    }
                }
            }
            tx.execute("DELETE FROM scratch", [])?;
            tx.commit()?;
            Ok(out)
        };
        run().unwrap_or_else(|_| fallback())
    })
}

/// A request-scoped FTS5 index over one request's chunks: what the graph walk searches.
pub struct LexIndex {
    conn: Connection,
    n: usize,
    /// term -> chunk indexes containing it, ascending (also gives document frequency)
    postings: HashMap<String, Vec<usize>>,
    /// per chunk: its distinct terms
    chunk_terms: Vec<HashSet<String>>,
    /// per chunk: every line in order, with that line's terms (empty for blank lines)
    lines: Vec<Vec<(String, HashSet<String>)>>,
}

impl LexIndex {
    /// Index `chunks` (`(id, raw)`) with `tokenizer`. Errors only if FTS5 itself fails.
    pub fn build(chunks: &[(String, String)], tokenizer: &str) -> rusqlite::Result<Self> {
        let conn = scratch_conn(tokenizer)?;
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE chunk_fts USING fts5(text, tokenize = \"{tokenizer}\");
             CREATE VIRTUAL TABLE chunk_i USING fts5vocab(chunk_fts, 'instance');
             CREATE VIRTUAL TABLE line_fts USING fts5(text, tokenize = \"{tokenizer}\");
             CREATE VIRTUAL TABLE line_i USING fts5vocab(line_fts, 'instance');"
        ))?;

        // line rowid -> (chunk, position within chunk)
        let mut line_at: Vec<(usize, usize)> = Vec::new();
        let mut lines: Vec<Vec<(String, HashSet<String>)>> = Vec::with_capacity(chunks.len());
        {
            let tx = conn.unchecked_transaction()?;
            {
                let mut ins_chunk =
                    tx.prepare("INSERT INTO chunk_fts(rowid, text) VALUES (?1, ?2)")?;
                let mut ins_line = tx.prepare("INSERT INTO line_fts(rowid, text) VALUES (?1, ?2)")?;
                for (i, (_, raw)) in chunks.iter().enumerate() {
                    ins_chunk.execute(params![(i + 1) as i64, raw])?;
                    let mut these = Vec::new();
                    for (j, line) in raw.lines().enumerate() {
                        these.push((line.to_string(), HashSet::new()));
                        if !line.trim().is_empty() {
                            line_at.push((i, j));
                            ins_line.execute(params![line_at.len() as i64, line])?;
                        }
                    }
                    lines.push(these);
                }
            }
            tx.commit()?;
        }

        let mut postings: HashMap<String, Vec<usize>> = HashMap::new();
        let mut chunk_terms: Vec<HashSet<String>> = vec![HashSet::new(); chunks.len()];
        {
            let mut stmt = conn.prepare("SELECT DISTINCT term, doc FROM chunk_i")?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
            for (term, doc) in rows.filter_map(|r| r.ok()) {
                let i = (doc - 1) as usize;
                postings.entry(term.clone()).or_default().push(i);
                chunk_terms[i].insert(term);
            }
        }
        for v in postings.values_mut() {
            v.sort_unstable();
            v.dedup();
        }
        {
            let mut stmt = conn.prepare("SELECT DISTINCT term, doc FROM line_i")?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
            for (term, doc) in rows.filter_map(|r| r.ok()) {
                let (i, j) = line_at[(doc - 1) as usize];
                lines[i][j].1.insert(term);
            }
        }

        Ok(Self { conn, n: chunks.len(), postings, chunk_terms, lines })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Chunks containing `term`. A term the index has never seen is treated as ubiquitous (df = n),
    /// which ranks it last rather than first — the safe direction for an unknown.
    pub fn df(&self, term: &str) -> usize {
        self.postings.get(term).map(|v| v.len()).unwrap_or(self.n)
    }

    pub fn postings(&self, term: &str) -> &[usize] {
        self.postings.get(term).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn chunk_terms(&self, i: usize) -> &HashSet<String> {
        &self.chunk_terms[i]
    }

    /// Tokenize arbitrary text with THIS index's tokenizer, so a query and the chunks it is matched
    /// against are always split the same way.
    pub fn tokens(&self, text: &str) -> Vec<String> {
        scratch_tokens(&self.conn, text).unwrap_or_default()
    }

    /// Lines of chunk `i` sharing a term with `wanted`, joined by newline, cut to about
    /// `max_tokens`. Same contract as the walk's previous line matcher.
    pub fn lines_matching(&self, i: usize, wanted: &HashSet<String>, max_tokens: usize) -> String {
        let joined = self.lines[i]
            .iter()
            .filter(|(_, terms)| terms.iter().any(|t| wanted.contains(t)))
            .map(|(text, _)| text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        crate::tokgraph::truncate_to_chars(&joined, max_tokens.saturating_mul(4))
    }

    /// FTS5's own BM25 over whole chunks for `terms` (OR'd). Higher is better; FTS5 reports it
    /// negated, so the sign is flipped here. Chunk index order breaks nothing — the caller sorts.
    pub fn bm25(&self, terms: &[String]) -> Vec<(usize, f64)> {
        if terms.is_empty() {
            return Vec::new();
        }
        let expr = terms.iter().map(|t| quote(t)).collect::<Vec<_>>().join(" OR ");
        let Ok(mut stmt) = self
            .conn
            .prepare("SELECT rowid, bm25(chunk_fts) FROM chunk_fts WHERE chunk_fts MATCH ?1")
        else {
            return Vec::new();
        };
        let Ok(rows) = stmt.query_map(params![expr], |r| {
            Ok(((r.get::<_, i64>(0)? - 1) as usize, -r.get::<_, f64>(1)?))
        }) else {
            return Vec::new();
        };
        rows.filter_map(|r| r.ok()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// If this fails, the build has no FTS5 and `tokens` is silently falling back. Better a red
    /// test than a quiet downgrade.
    #[test]
    fn fts5_is_compiled_in() {
        assert!(scratch_conn(TOKENIZER).is_ok());
        assert_eq!(tokens("SERVICE_PORT = 8931"), vec!["service", "port", "8931"]);
    }

    /// The finding that motivated this module, from both sides: `port` must still find
    /// `SERVICE_PORT`, and must no longer be found inside `report`.
    #[test]
    fn identifiers_split_but_words_do_not_leak_into_other_words() {
        let t = tokens("the report covers SERVICE_PORT and small things");
        assert!(t.contains(&"port".to_string()), "SERVICE_PORT did not yield `port`: {t:?}");
        assert!(!t.contains(&"all".to_string()), "`small` yielded `all`: {t:?}");
    }

    #[test]
    fn case_and_diacritics_fold() {
        assert_eq!(tokens("École"), tokens("ecole"));
    }

    #[test]
    fn the_index_counts_chunks_not_occurrences() {
        let chunks = vec![
            ("1".to_string(), "postgres postgres postgres".to_string()),
            ("2".to_string(), "postgres once\nand a second line".to_string()),
            ("3".to_string(), "nothing relevant".to_string()),
        ];
        let lx = LexIndex::build(&chunks, TOKENIZER).expect("index builds");
        assert_eq!(lx.n(), 3);
        assert_eq!(lx.df("postgres"), 2);
        assert_eq!(lx.postings("postgres"), &[0, 1]);
        assert_eq!(lx.df("never-seen"), 3, "an unknown term must rank as ubiquitous, not rare");
    }

    #[test]
    fn lines_matching_returns_only_matching_lines_in_order() {
        let chunks = vec![("1".to_string(), "alpha\nthe port is 8931\n\nbeta port".to_string())];
        let lx = LexIndex::build(&chunks, TOKENIZER).expect("index builds");
        let wanted: HashSet<String> = ["port".to_string()].into_iter().collect();
        assert_eq!(lx.lines_matching(0, &wanted, 64), "the port is 8931\nbeta port");
    }

    #[test]
    fn bm25_prefers_the_chunk_that_is_about_the_term() {
        let chunks = vec![
            ("1".to_string(), "filler filler filler postgres filler filler filler".to_string()),
            ("2".to_string(), "postgres stores the state; postgres is the database".to_string()),
            ("3".to_string(), "unrelated".to_string()),
        ];
        let lx = LexIndex::build(&chunks, TOKENIZER).expect("index builds");
        let mut hits = lx.bm25(&["postgres".to_string()]);
        hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 1, "BM25 did not rank the on-topic chunk first: {hits:?}");
    }

    /// The batch path must agree with the one-at-a-time path, blank lines included.
    #[test]
    fn tokens_many_matches_tokens_line_by_line() {
        let lines = ["SERVICE_PORT = 8931", "", "   ", "the report"];
        let many = tokens_many(&lines);
        assert_eq!(many.len(), 4);
        for (i, l) in lines.iter().enumerate() {
            let one = if l.trim().is_empty() { Vec::new() } else { tokens(l) };
            assert_eq!(many[i], one, "line {i} ({l:?}) differs");
        }
    }
}
