// SPDX-License-Identifier: Apache-2.0
//! Token-following lexical expansion — the "graph" targeted-seed strategy.
//!
//! An alternative to the embedding walk (`overflow::expand_path`) for verbatim-entity multi-hop.
//! Start from the query-matched lines, tokenize them, then follow the DISTINCTIVE tokens (those
//! appearing in few chunks) to grep the other chunks — iterating up to `hops` rounds. This walks a
//! reasoning chain (`/login → auth-svc → golang → postgres`) by exact bridging tokens, which are
//! sharp where chunk-embedding similarity is blurry. All local, no LLM/core/embedding calls.
//!
//! The core `token_follow` function is a PURE function (no store, no I/O) implemented by the
//! tokgraph subagent; this file is the contract skeleton.

use std::collections::{HashMap, HashSet};

/// One line reached on the token-follow path.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphHit {
    /// positional id of the chunk this line came from
    pub id: String,
    /// the matched line(s) from that chunk (the bridging fact)
    pub snippet: String,
}

/// Words too common to carry signal. THE list — there is no second one.
///
/// Three call sites share it: `is_usable` here (the graph walk), `corpus::bm25_candidates` (pruning
/// worthless huge-postings FTS lookups), and `overflow::grep_lines`. `grep_lines` kept a private
/// 34-word copy until the shared stopword list; two lists that must agree and are not the same list will drift, and the
/// drift shows up as a retrieval difference nobody is looking for.
///
/// It matters most for `grep_lines`, which matches by SUBSTRING: a query term `all` hits "small",
/// "call" and "allocate", so a word missing from the list there is a false-positive source, not
/// merely a wasted lookup.
///
/// Every entry is an ordinary English function word. `entirely` and `exclusively` used to be here
/// too and were removed: they are not function words, they appear in no standard stopword list, and
/// they are verbatim filler from this module's own test fixtures ("built *entirely* in golang").
/// A stopword list that grew to fit the fixtures is tuning wearing a config hat.
pub const STOPWORDS: &[&str] = &[
    "the", "is", "in", "a", "an", "of", "to", "and", "or", "what", "which", "how", "do", "does",
    "are", "was", "were", "be", "for", "on", "at", "it", "this", "that", "with", "from", "by",
    "as", "i", "you", "we", "they", "there", "here", "use", "uses", "used", "their", "its",
    "all",
];

/// The tokenizer the walk follows tokens with.
///
/// NOT `embed::tokens`, which splits on every non-alphanumeric character. That tokenizer is correct
/// for what it is — it feeds the hashing embedder, where a bag of word-parts is the point — but it
/// makes this module's stated job impossible. `auth-svc` arrives as `["auth", "svc"]`, so the
/// module documented as following "verbatim entities" never sees the entity, and `widget-x` arrives
/// as `["widget"]` alone, the `x` dropped for being one character. Hyphenated service names, dotted
/// paths and file paths are exactly the sharp bridges the walk exists to follow, and they were the
/// ones being cut in half.
///
/// This emits the WHOLE unit and its parts, so nothing that used to match stops matching — the
/// change only ever adds the compound. That matters with IDF ranking downstream: `auth-svc` is rarer
/// than either half, so the compound outranks its own parts and the walk prefers the sharper bridge
/// on its own, with no rule saying so.
///
/// Separators are trimmed from the ends before splitting. Without that, prose sentences produce
/// `end.` as a distinct token — a df=1 token, therefore top-ranked, therefore followed, spending
/// budget on a full stop.
///
/// Lowercasing is `to_lowercase`, not `to_ascii_lowercase`: the ASCII form silently leaves `École`
/// and `école` as different tokens, and a bridge that fails to match its own other spelling is not
/// a bridge. (the token under-count and the English-only routing limit are the same blind spot elsewhere and remain open.)
pub fn graph_tokens(s: &str) -> Vec<String> {
    const SEPS: [char; 4] = ['_', '-', '.', '/'];
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut cur = String::new();

    let flush = |cur: &mut String, out: &mut Vec<String>, seen: &mut HashSet<String>| {
        let unit = cur.trim_matches(|c| SEPS.contains(&c));
        if !unit.is_empty() {
            let whole = unit.to_lowercase();
            if seen.insert(whole.clone()) {
                out.push(whole);
            }
            for part in unit.split(SEPS).filter(|p| !p.is_empty()) {
                let p = part.to_lowercase();
                if seen.insert(p.clone()) {
                    out.push(p);
                }
            }
        }
        cur.clear();
    };

    for ch in s.chars() {
        if ch.is_alphanumeric() || SEPS.contains(&ch) {
            cur.push(ch);
        } else if !cur.is_empty() {
            flush(&mut cur, &mut out, &mut seen);
        }
    }
    if !cur.is_empty() {
        flush(&mut cur, &mut out, &mut seen);
    }
    out
}

/// A "usable" token: long enough and not a stopword.
pub fn is_usable(tok: &str) -> bool {
    tok.len() >= 2 && !STOPWORDS.contains(&tok)
}

/// Truncate `s` to at most `max_chars` bytes, snapped back to a char boundary.
pub(crate) fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    let mut end = max_chars;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// Lines of `raw` whose token SET intersects `wanted` (exact token membership, not substring),
/// joined by "\n" and truncated to about `max_tokens_per_snippet` tokens.
fn lines_matching(raw: &str, wanted: &HashSet<String>, max_tokens_per_snippet: usize) -> String {
    let matched: Vec<&str> = raw
        .lines()
        .filter(|line| {
            graph_tokens(line)
                .into_iter()
                .any(|t| wanted.contains(&t))
        })
        .collect();
    let joined = matched.join("\n");
    truncate_to_chars(&joined, max_tokens_per_snippet.saturating_mul(4))
}

/// Follow distinctive tokens from the query-matched lines through the other chunks, up to `hops`
/// rounds, returning the connected lines in DISCOVERY order (query-matched lines first, then lines
/// reached by following bridging tokens).
///
/// - `chunks`: `(positional_id, raw)` for every chunk in the current request.
/// - `query`: the user query; its non-stopword tokens seed the walk.
/// - `max_df_frac`: distinctiveness threshold — a token appearing in MORE than this fraction of
///   chunks is too common to follow (skipped). E.g. 0.15.
/// - `hops`: max follow rounds (0 = just the query-matched lines).
/// - `max_chunks`: overall cap on chunks placed on the path.
/// - `max_tokens_per_snippet`: per-chunk snippet token budget (truncate longer).
pub fn token_follow(
    chunks: &[(String, String)],
    query: &str,
    max_df_frac: f32,
    hops: usize,
    max_chunks: usize,
    max_tokens_per_snippet: usize,
) -> Vec<GraphHit> {
    let n = chunks.len();
    if n == 0 {
        return Vec::new();
    }

    // Document frequency + postings over the distinct token set of each chunk.
    let mut df: HashMap<String, usize> = HashMap::new();
    let mut postings: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, (_, raw)) in chunks.iter().enumerate() {
        let set: HashSet<String> = graph_tokens(raw).into_iter().collect();
        for tok in &set {
            *df.entry(tok.clone()).or_insert(0) += 1;
            postings.entry(tok.clone()).or_default().push(i);
        }
    }

    // FLOOR OF 2, not 1. A bridging token is present in TWO chunks by definition —
    // being in both is what makes it a bridge. A threshold of 1 therefore admits only tokens unique
    // to a single chunk, which are precisely the ones that lead nowhere: the walk cannot leave its
    // anchors and multi-hop silently becomes plain lookup.
    //
    // `ceil(0.15 * n)` is 1 for every n <= 6, so at the shipped default the mechanism was inert
    // below seven chunks — and chunk counts are small exactly when `n_ctx` is small, which is the
    // configuration the README's own quickstart uses.
    //
    // This is a structural fix, not a tuning guess, and it is scoped to the broken regime: for
    // n >= 7 the computed threshold is already >= 2, so the floor is a no-op and retrieval at
    // realistic sizes is bit-for-bit unchanged. Choosing the fraction itself still wants a
    // measurement (see the graph-seed default follow-up); making the mechanism function at all does not.
    let distinctive_threshold = std::cmp::max(2, (max_df_frac * n as f32).ceil() as usize);
    let is_distinctive = |tok: &str, df: &HashMap<String, usize>| -> bool {
        is_usable(tok) && df.get(tok).copied().unwrap_or(0) <= distinctive_threshold
    };

    // Seed on the query's DISTINCTIVE tokens only: a query word that is common across chunks
    // (e.g. "service"/"database" in software-architecture prose) would otherwise match every
    // distractor, pollute the anchor set, and explode the frontier before the real chain is walked.
    let q_tokens: HashSet<String> = graph_tokens(query)
        .into_iter()
        .filter(|t| is_distinctive(t.as_str(), &df))
        .collect();

    let mut visited: HashSet<usize> = HashSet::new();
    let mut path: Vec<GraphHit> = Vec::new();
    let mut frontier: HashSet<String> = HashSet::new();

    // Seed round: chunks whose lines match the query tokens.
    for (i, (id, raw)) in chunks.iter().enumerate() {
        if path.len() >= max_chunks {
            break;
        }
        if visited.contains(&i) {
            continue;
        }
        let snip = lines_matching(raw, &q_tokens, max_tokens_per_snippet);
        if !snip.is_empty() {
            for tok in graph_tokens(&snip) {
                if is_distinctive(&tok, &df) {
                    frontier.insert(tok);
                }
            }
            path.push(GraphHit {
                id: id.clone(),
                snippet: snip,
            });
            visited.insert(i);
        }
    }

    // Follow rounds: chase distinctive bridging tokens through the postings list.
    let mut followed: HashSet<String> = HashSet::new();
    for _ in 0..hops {
        if frontier.is_empty() || path.len() >= max_chunks {
            break;
        }
        let mut next_frontier: HashSet<String> = HashSet::new();
        // SORTED, because the walk stops at `max_chunks` and whichever tokens are followed first
        // decide which chunks make the cut. Iterating a HashSet meant the same
        // conversation, same query and same config could seed differently between runs — in a repo
        // whose benchmarks are single-run, that is a source of unreproducible results nobody was
        // controlling for. It also has to be fixed BEFORE the graph-seed default can be measured: you cannot sweep a
        // retrieval parameter while the baseline moves on its own.
        let mut current: Vec<String> = frontier.into_iter().collect();
        current.sort_unstable();
        for tok in current {
            if followed.contains(&tok) {
                continue;
            }
            followed.insert(tok.clone());
            let wanted: HashSet<String> = std::iter::once(tok.clone()).collect();
            if let Some(idxs) = postings.get(&tok) {
                for &i in idxs {
                    if path.len() >= max_chunks {
                        break;
                    }
                    if visited.contains(&i) {
                        continue;
                    }
                    let (id, raw) = &chunks[i];
                    let snip = lines_matching(raw, &wanted, max_tokens_per_snippet);
                    if !snip.is_empty() {
                        for t2 in graph_tokens(&snip) {
                            if is_distinctive(&t2, &df) {
                                next_frontier.insert(t2);
                            }
                        }
                        path.push(GraphHit {
                            id: id.clone(),
                            snippet: snip,
                        });
                        visited.insert(i);
                    }
                }
            }
        }
        frontier = next_frontier;
    }

    path
}

/// Inverse document frequency, smoothed: `ln(1 + N/df)`. Rare tokens score high, common ones low.
///
/// The `1 +` is not cosmetic. Plain `ln(N/df)` is exactly zero for a token present in every chunk,
/// and with two or three chunks in the request that describes ordinary words — so the walk would go
/// dead precisely at small chunk counts, which is the same shape of bug as the graph-seed default's threshold being
/// inert below seven chunks. Smoothing keeps every score positive while preserving the ordering
/// (`ln` is monotone, so ranking by `ln(1 + N/df)` ranks by `N/df`), so nothing is ever unfollowable
/// for being common — it just loses to something rarer when both are on offer.
fn idf(df: usize, n: usize) -> f64 {
    (1.0 + n as f64 / df.max(1) as f64).ln()
}

/// How many tokens from one snippet may become follow candidates.
///
/// This bounds fan-out without reintroducing a frequency cutoff, and the distinction matters: a
/// cutoff refuses a token for being *common*, whereas this keeps the most *informative* few. A
/// bridge appearing in six chunks still qualifies if it is among the best its snippet offers, which
/// is exactly the case a cutoff cannot express.
const MAX_FOLLOW_PER_SNIPPET: usize = 8;

/// One chunk waiting to be visited, and the token that would justify visiting it.
struct Candidate {
    score: f64,
    idx: usize,
    /// Tokens whose lines make up the snippet if this candidate is taken.
    wanted: Vec<String>,
    depth: usize,
}

/// The graph seed, ranked instead of filtered.
///
/// WHY THIS EXISTS. `token_follow` decides what to follow with a frequency cutoff, and the sweep
/// showed that cutoff cannot win. The threshold turns out to be exactly the highest bridge frequency
/// the walk can follow — 100% at or below it, 0% above — so raising it is the only way to reach a
/// rarer bridge. But raising it also admits common tokens, which fill the chunk budget and crowd the
/// real chain out. The two failure modes pull in opposite directions and no single value escapes
/// both.
///
/// Ranking dissolves the conflict. Nothing is ever excluded; candidates are simply visited
/// best-first, so the budget is spent on the most informative chunks reachable, and a low-IDF bridge
/// is still followed — just later, if the budget lasts. There is no cliff to fall off because there
/// is no threshold.
///
/// Deterministic by construction: ties break on chunk index, never on hash order.
pub fn token_follow_idf(
    chunks: &[(String, String)],
    query: &str,
    hops: usize,
    max_chunks: usize,
    max_tokens_per_snippet: usize,
) -> Vec<GraphHit> {
    let lex = HandLexicon::build(chunks);
    walk(&lex, chunks, query, hops, max_chunks, max_tokens_per_snippet, false)
}

/// The same ranked walk, on SQLite FTS5 instead of the hand-written keyword code.
///
/// Only the keyword layer differs — tokenizing, document frequency, which lines match — so a sweep
/// comparing this with [`token_follow_idf`] measures the substrate and nothing else. With
/// `bm25_seed`, the starting chunks are ranked by FTS5's own BM25 rather than by summed IDF: that is
/// the one place the walk hands a ranking decision to the library rather than just its statistics.
///
/// Falls back to the hand walk if the index cannot be built. FTS5 is compiled into the bundled
/// SQLite, so this should never fire; it exists because this runs per request and must not panic.
pub fn token_follow_fts(
    chunks: &[(String, String)],
    query: &str,
    hops: usize,
    max_chunks: usize,
    max_tokens_per_snippet: usize,
    tokenizer: &str,
    bm25_seed: bool,
) -> Vec<GraphHit> {
    match crate::lexindex::LexIndex::build(chunks, tokenizer) {
        Ok(lex) => walk(&lex, chunks, query, hops, max_chunks, max_tokens_per_snippet, bm25_seed),
        Err(_) => token_follow_idf(chunks, query, hops, max_chunks, max_tokens_per_snippet),
    }
}

/// The keyword primitives the walk needs. Two implementations: the hand-written one the walk was
/// built on, and SQLite FTS5 (`lexindex`). Keeping the walk generic over this is what lets the two
/// be compared without the comparison also measuring a second copy of the walk.
pub trait Lexicon {
    fn n(&self) -> usize;
    fn df(&self, term: &str) -> usize;
    fn postings(&self, term: &str) -> &[usize];
    fn chunk_terms(&self, i: usize) -> &HashSet<String>;
    fn tokens(&self, text: &str) -> Vec<String>;
    fn lines_matching(&self, i: usize, wanted: &HashSet<String>, max_tokens: usize) -> String;
    /// A library ranking for the seed, if the lexicon has one.
    fn bm25(&self, _terms: &[String]) -> Vec<(usize, f64)> {
        Vec::new()
    }
}

/// The original hand-written keyword layer: `graph_tokens`, counted in Rust.
pub struct HandLexicon<'a> {
    chunks: &'a [(String, String)],
    postings: HashMap<String, Vec<usize>>,
    chunk_terms: Vec<HashSet<String>>,
}

impl<'a> HandLexicon<'a> {
    pub fn build(chunks: &'a [(String, String)]) -> Self {
        let mut postings: HashMap<String, Vec<usize>> = HashMap::new();
        let mut chunk_terms = Vec::with_capacity(chunks.len());
        for (i, (_, raw)) in chunks.iter().enumerate() {
            let set: HashSet<String> = graph_tokens(raw).into_iter().collect();
            for tok in &set {
                postings.entry(tok.clone()).or_default().push(i);
            }
            chunk_terms.push(set);
        }
        Self { chunks, postings, chunk_terms }
    }
}

impl Lexicon for HandLexicon<'_> {
    fn n(&self) -> usize {
        self.chunks.len()
    }
    fn df(&self, term: &str) -> usize {
        self.postings.get(term).map(|v| v.len()).unwrap_or(self.chunks.len())
    }
    fn postings(&self, term: &str) -> &[usize] {
        self.postings.get(term).map(|v| v.as_slice()).unwrap_or(&[])
    }
    fn chunk_terms(&self, i: usize) -> &HashSet<String> {
        &self.chunk_terms[i]
    }
    fn tokens(&self, text: &str) -> Vec<String> {
        graph_tokens(text)
    }
    fn lines_matching(&self, i: usize, wanted: &HashSet<String>, max_tokens: usize) -> String {
        lines_matching(&self.chunks[i].1, wanted, max_tokens)
    }
}

impl Lexicon for crate::lexindex::LexIndex {
    fn n(&self) -> usize {
        crate::lexindex::LexIndex::n(self)
    }
    fn df(&self, term: &str) -> usize {
        crate::lexindex::LexIndex::df(self, term)
    }
    fn postings(&self, term: &str) -> &[usize] {
        crate::lexindex::LexIndex::postings(self, term)
    }
    fn chunk_terms(&self, i: usize) -> &HashSet<String> {
        crate::lexindex::LexIndex::chunk_terms(self, i)
    }
    fn tokens(&self, text: &str) -> Vec<String> {
        crate::lexindex::LexIndex::tokens(self, text)
    }
    fn lines_matching(&self, i: usize, wanted: &HashSet<String>, max_tokens: usize) -> String {
        crate::lexindex::LexIndex::lines_matching(self, i, wanted, max_tokens)
    }
    fn bm25(&self, terms: &[String]) -> Vec<(usize, f64)> {
        crate::lexindex::LexIndex::bm25(self, terms)
    }
}

/// The best-first walk, over any [`Lexicon`]. Logic unchanged from the original IDF walk.
fn walk(
    lex: &dyn Lexicon,
    chunks: &[(String, String)],
    query: &str,
    hops: usize,
    max_chunks: usize,
    max_tokens_per_snippet: usize,
    bm25_seed: bool,
) -> Vec<GraphHit> {
    let n = lex.n();
    if n == 0 {
        return Vec::new();
    }

    let q_tokens: Vec<String> = {
        let mut v: Vec<String> = lex.tokens(query).into_iter().filter(|t| is_usable(t)).collect();
        v.sort_unstable();
        v.dedup();
        v
    };

    // Seed: every chunk sharing a query token. Scored by summed IDF, or by the library's BM25 when
    // asked — BM25 also weighs how often the term occurs and how long the chunk is.
    let mut candidates: Vec<Candidate> = Vec::new();
    let ranked = if bm25_seed { lex.bm25(&q_tokens) } else { Vec::new() };
    if bm25_seed && !ranked.is_empty() {
        for (i, score) in ranked {
            let terms = lex.chunk_terms(i);
            let hit: Vec<String> = q_tokens.iter().filter(|t| terms.contains(*t)).cloned().collect();
            if !hit.is_empty() && score > 0.0 {
                candidates.push(Candidate { score, idx: i, wanted: hit, depth: 0 });
            }
        }
    } else {
        for i in 0..n {
            let terms = lex.chunk_terms(i);
            let hit: Vec<String> = q_tokens.iter().filter(|t| terms.contains(*t)).cloned().collect();
            if hit.is_empty() {
                continue;
            }
            let score: f64 = hit.iter().map(|t| idf(lex.df(t), n)).sum();
            if score > 0.0 {
                candidates.push(Candidate { score, idx: i, wanted: hit, depth: 0 });
            }
        }
    }

    let mut visited: HashSet<usize> = HashSet::new();
    let mut path: Vec<GraphHit> = Vec::new();

    while path.len() < max_chunks {
        // Best-first: highest score wins, chunk index breaks ties so the walk is reproducible.
        let Some(best) = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| !visited.contains(&c.idx))
            .min_by(|(_, a), (_, b)| {
                b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then(a.idx.cmp(&b.idx))
            })
            .map(|(pos, _)| pos)
        else {
            break;
        };
        let cand = candidates.swap_remove(best);
        if !visited.insert(cand.idx) {
            continue;
        }

        let wanted: HashSet<String> = cand.wanted.iter().cloned().collect();
        let snip = lex.lines_matching(cand.idx, &wanted, max_tokens_per_snippet);
        if snip.is_empty() {
            continue;
        }
        path.push(GraphHit { id: chunks[cand.idx].0.clone(), snippet: snip.clone() });

        if cand.depth >= hops {
            continue;
        }
        // Expand: the most informative tokens in what we just read become the next candidates.
        let mut toks: Vec<(f64, String)> = lex
            .tokens(&snip)
            .into_iter()
            .filter(|t| is_usable(t))
            .collect::<HashSet<String>>()
            .into_iter()
            .map(|t| (idf(lex.df(&t), n), t))
            .collect();
        toks.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1))
        });
        for (score, tok) in toks.into_iter().take(MAX_FOLLOW_PER_SNIPPET) {
            for &j in lex.postings(&tok) {
                if visited.contains(&j) {
                    continue;
                }
                candidates.push(Candidate {
                    score,
                    idx: j,
                    wanted: vec![tok.clone()],
                    depth: cand.depth + 1,
                });
            }
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_chunks() -> Vec<(String, String)> {
        vec![
            (
                "1".to_string(),
                "The /login endpoint is handled by the auth-svc component.".to_string(),
            ),
            (
                "2".to_string(),
                "The auth-svc component is built entirely in golang.".to_string(),
            ),
            (
                "3".to_string(),
                "Golang subsystems persist their state to postgres exclusively.".to_string(),
            ),
            (
                "4".to_string(),
                "The weather forecast is sunny and warm today.".to_string(),
            ),
        ]
    }

    #[test]
    fn token_follow_walks_verbatim_chain() {
        let chunks = chain_chunks();
        let query = "What database does the service behind /login use?";
        let path = token_follow(&chunks, query, 0.5, 2, 8, 64);

        let ids: Vec<&str> = path.iter().map(|h| h.id.as_str()).collect();
        assert!(!ids.is_empty());
        assert_eq!(ids[0], "1", "path should start at the query-matched anchor");
        assert!(ids.contains(&"2"), "should follow auth-svc into chunk 2: {ids:?}");
        assert!(ids.contains(&"3"), "should follow golang into chunk 3: {ids:?}");
        assert!(!ids.contains(&"4"), "unrelated distractor should not be pulled in: {ids:?}");

        let hit3 = path.iter().find(|h| h.id == "3").expect("chunk 3 on path");
        assert!(
            hit3.snippet.contains("postgres"),
            "snippet for chunk 3 should mention postgres: {:?}",
            hit3.snippet
        );
    }

    fn m3_chain_chunks() -> Vec<(String, String)> {
        // The m3 fixture chain: /search → index-svc → jvm → elasticsearch. The hop2→hop3 bridge
        // token differs only in case ("jvm" vs sentence-initial "Jvm"); embed::tokens lowercases,
        // so the bridge must still connect. Isolates RETRIEVAL from the reader model: proves hop3
        // (elasticsearch) is reachable, so an m3 failure is a reading/query problem, not a miss.
        vec![
            (
                "1".to_string(),
                "The /search endpoint is handled by the index-svc component.".to_string(),
            ),
            (
                "2".to_string(),
                "The index-svc component is built entirely in jvm.".to_string(),
            ),
            (
                "3".to_string(),
                "Jvm subsystems persist their state to elasticsearch exclusively.".to_string(),
            ),
            (
                "4".to_string(),
                "The weather forecast is sunny and warm today.".to_string(),
            ),
        ]
    }

    #[test]
    fn token_follow_reaches_m3_final_hop() {
        let chunks = m3_chain_chunks();
        // The colliding category word "engine" appears in the query but in no chunk — retrieval
        // rides the "search → index-svc → jvm → elasticsearch" bridge tokens, not "engine".
        let query = "What engine does the service behind /search use?";
        let path = token_follow(&chunks, query, 0.5, 2, 8, 64);

        let ids: Vec<&str> = path.iter().map(|h| h.id.as_str()).collect();
        assert!(!ids.is_empty());
        assert_eq!(ids[0], "1", "path should start at the query-matched anchor");
        assert!(ids.contains(&"2"), "should follow index-svc into chunk 2: {ids:?}");
        assert!(
            ids.contains(&"3"),
            "should follow the jvm→Jvm (case-normalized) bridge into chunk 3: {ids:?}"
        );
        assert!(!ids.contains(&"4"), "unrelated distractor should not be pulled in: {ids:?}");

        let hit3 = path.iter().find(|h| h.id == "3").expect("chunk 3 on path");
        assert!(
            hit3.snippet.contains("elasticsearch"),
            "snippet for chunk 3 should mention elasticsearch: {:?}",
            hit3.snippet
        );
    }

    /// Every other test in this module passes `max_df_frac = 0.5`; the SHIPPED default
    /// (`XZO_OVERFLOW_GRAPH_MAX_DF`) is **0.15**, and the walk used to behave differently under it —
    /// `ceil(0.15 * n)` is 1 for any n <= 6, and a threshold of 1 admits only tokens unique to one
    /// chunk, which are exactly the ones that lead nowhere. Below seven chunks the walk could not
    /// leave its anchors, so multi-hop silently became plain lookup.
    ///
    /// The floor of 2 fixes it. This asserts the chain now walks at BOTH sizes.
    #[test]
    fn shipped_default_walks_the_chain_at_any_chunk_count() {
        let query = "What database does the service behind /login use?";

        // n = 4: the regime that used to return only the anchor.
        let small = chain_chunks();
        let ids: Vec<String> =
            token_follow(&small, query, 0.15, 2, 8, 64).iter().map(|h| h.id.clone()).collect();
        assert!(ids.contains(&"2".to_string()), "hop 2 unreachable at n=4: {ids:?}");
        assert!(ids.contains(&"3".to_string()), "hop 3 unreachable at n=4: {ids:?}");
        assert!(!ids.contains(&"4".to_string()), "the distractor was pulled in: {ids:?}");

        // n = 19: already worked, and must be unchanged — the floor is a no-op once the computed
        // threshold reaches 2 on its own.
        let mut large = chain_chunks();
        for i in 5..20 {
            large.push((i.to_string(), format!("Unrelated filler chunk number {i} about nothing.")));
        }
        let ids: Vec<String> =
            token_follow(&large, query, 0.15, 2, 8, 64).iter().map(|h| h.id.clone()).collect();
        assert!(
            ids.contains(&"2".to_string()) && ids.contains(&"3".to_string()),
            "at n=19 the chain should still walk end to end: {ids:?}",
        );
    }

    /// The floor must not change anything that already worked. For n >= 7 the computed threshold is
    /// already >= 2, so `max(2, ..)` is arithmetically a no-op — this pins that claim rather than
    /// asserting it in a comment.
    #[test]
    fn the_floor_is_a_no_op_from_seven_chunks_upward() {
        for n in 7..=40usize {
            let computed = (0.15f32 * n as f32).ceil() as usize;
            assert!(
                computed >= 2,
                "n={n}: computed threshold {computed} < 2, so the floor would change behaviour here"
            );
        }
        // ...and it genuinely does bite below that, which is the whole point.
        for n in 1..=6usize {
            assert_eq!((0.15f32 * n as f32).ceil() as usize, 1, "n={n} should compute to 1");
        }
    }

    /// the same inputs must give the same path, every time.
    ///
    /// A single run proves nothing here — a HashSet often *happens* to iterate consistently within
    /// one process. What catches the bug is a cap tight enough that the walk must choose, repeated
    /// across fresh sets whose iteration order differs run to run.
    #[test]
    fn the_walk_is_deterministic_when_the_cap_binds() {
        // One anchor naming twenty distinctive tokens, each of which leads to its own chunk. The
        // frontier therefore holds far more candidates than the cap allows, so which of them gets
        // followed first decides the result — the exact condition the deterministic walk is about.
        let mut chunks = vec![(
            "1".to_string(),
            format!(
                "The alpha manifest enumerates {}.",
                (1..=20).map(|i| format!("zeta{i:02}")).collect::<Vec<_>>().join(" ")
            ),
        )];
        for i in 1..=20 {
            chunks.push((format!("{}", i + 1), format!("Detail for zeta{i:02} is recorded here.")));
        }
        let query = "What does the alpha manifest enumerate?";

        let first: Vec<String> = token_follow(&chunks, query, 0.5, 3, 4, 64)
            .iter()
            .map(|h| h.id.clone())
            .collect();
        assert!(first.len() > 1, "the cap must actually bind for this to test anything");
        for run in 0..12 {
            let again: Vec<String> = token_follow(&chunks, query, 0.5, 3, 4, 64)
                .iter()
                .map(|h| h.id.clone())
                .collect();
            assert_eq!(again, first, "run {run} produced a different path");
        }
    }

    #[test]
    fn token_follow_hops_zero_is_anchor_only() {
        let chunks = chain_chunks();
        let query = "What database does the service behind /login use?";
        let path = token_follow(&chunks, query, 0.5, 0, 8, 64);

        let ids: Vec<&str> = path.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, vec!["1"], "hops=0 should only return the anchor: {ids:?}");
    }

    #[test]
    fn token_follow_respects_max_chunks() {
        let chunks = chain_chunks();
        let query = "What database does the service behind /login use?";
        let path = token_follow(&chunks, query, 0.5, 2, 2, 64);
        assert!(path.len() <= 2, "path should respect max_chunks: {}", path.len());
    }

    #[test]
    fn token_follow_ignores_common_tokens() {
        // Every chunk shares "system"/"the", but only chunk 1 shares the distinctive query
        // token "widget". The common tokens must never be followed, so the walk should stay
        // small instead of pulling in every chunk via "system".
        let chunks = vec![
            (
                "1".to_string(),
                "The system uses widget-x for processing.".to_string(),
            ),
            ("2".to_string(), "The system logs errors to a file.".to_string()),
            ("3".to_string(), "The system requires a restart daily.".to_string()),
            ("4".to_string(), "The system supports multiple users.".to_string()),
        ];
        let query = "What is widget-x?";
        let path = token_follow(&chunks, query, 0.5, 2, 8, 64);

        let ids: Vec<&str> = path.iter().map(|h| h.id.as_str()).collect();
        assert!(
            path.len() < chunks.len(),
            "common token 'system' must not pull in every chunk: {ids:?}"
        );
        assert!(ids.contains(&"1"), "anchor chunk should be on the path: {ids:?}");
        assert!(!ids.contains(&"2"), "chunk 2 only shares the common token: {ids:?}");
        assert!(!ids.contains(&"3"), "chunk 3 only shares the common token: {ids:?}");
        assert!(!ids.contains(&"4"), "chunk 4 only shares the common token: {ids:?}");
    }

    // ---------------------------------------------------------------------------------------
    // The ranked walk
    // ---------------------------------------------------------------------------------------

    /// A chain built so the bridge token appears in `bridge_df` chunks. This is the exact shape the
    /// cutoff cannot handle: raise the threshold enough to follow the bridge and the same threshold
    /// admits the filler, which is more numerous and takes the budget.
    fn chain_with_bridge_df(bridge_df: usize) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = Vec::new();
        v.push(("c0".into(), "the frobnicator endpoint is served by quuxsvc".into()));
        v.push(("c1".into(), "quuxsvc is written in zigzaglang".into()));
        // Chunks that mention the bridge but carry no answer -- what makes its df high.
        for i in 0..bridge_df.saturating_sub(2) {
            v.push((format!("e{i}"), "quuxsvc was discussed in the weekly review".into()));
        }
        // Filler that shares ordinary words with everything.
        for i in 0..24 {
            v.push((format!("f{i}"), format!("the service was deployed in region {i}")));
        }
        v
    }

    /// The headline claim: the ranked walk follows a bridge regardless of how common it is.
    #[test]
    fn the_ranked_walk_follows_a_bridge_at_any_frequency() {
        for df in [2usize, 3, 4, 5, 6, 8] {
            let chunks = chain_with_bridge_df(df);
            let hits = token_follow_idf(&chunks, "who serves the frobnicator endpoint", 2, 12, 64);
            let text = hits.iter().map(|h| h.snippet.as_str()).collect::<Vec<_>>().join("
");
            assert!(
                text.contains("zigzaglang"),
                "bridge_df={df}: the second hop was not reached. Path was {:?}",
                hits.iter().map(|h| &h.id).collect::<Vec<_>>()
            );
        }
    }

    /// The same fixtures through the cutoff, to pin WHY the ranked walk exists. If this ever starts
    /// passing, the cutoff got better and this comparison needs re-running rather than deleting.
    #[test]
    fn the_cutoff_loses_the_chain_once_the_bridge_is_common() {
        let chunks = chain_with_bridge_df(8);
        let hits = token_follow(&chunks, "who serves the frobnicator endpoint", 0.15, 2, 12, 64);
        let text = hits.iter().map(|h| h.snippet.as_str()).collect::<Vec<_>>().join("
");
        assert!(
            !text.contains("zigzaglang"),
            "the cutoff now reaches a df=8 bridge -- re-run the sweep before trusting either default"
        );
    }

    /// Determinism, the lesson from the earlier walk applied to the new walk. Best-first search over a HashMap-derived
    /// candidate set is exactly where hash order leaks back in, so ties must break on chunk index.
    #[test]
    fn the_ranked_walk_is_deterministic() {
        let chunks = chain_with_bridge_df(4);
        let first = token_follow_idf(&chunks, "who serves the frobnicator endpoint", 2, 6, 64);
        for _ in 0..12 {
            let again = token_follow_idf(&chunks, "who serves the frobnicator endpoint", 2, 6, 64);
            assert_eq!(first, again, "the ranked walk varied between runs");
        }
    }

    /// The budget is a cap, not a target: a query with one relevant chunk must not drag in eleven
    /// more just because the budget allows it.
    #[test]
    fn the_ranked_walk_stops_when_nothing_is_left_to_follow() {
        let chunks = vec![("c0".to_string(), "the frobnicator endpoint is served by quuxsvc".to_string())];
        let hits = token_follow_idf(&chunks, "frobnicator", 2, 12, 64);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn the_ranked_walk_handles_an_empty_corpus() {
        assert!(token_follow_idf(&[], "anything", 2, 8, 64).is_empty());
    }

    /// The property that replaces the cutoff: a common token ranks below a rare one, but stays
    /// positive so it remains followable when nothing better is on offer. A hard zero here would
    /// reintroduce "unfollowable", which is the whole thing being removed.
    #[test]
    fn a_common_token_ranks_low_but_stays_followable() {
        assert!(idf(10, 10) > 0.0, "a token in every chunk must still be followable");
        assert!(idf(1, 10) > idf(5, 10));
        assert!(idf(5, 10) > idf(10, 10));
        // The degenerate corpus: one chunk, every token in it. Must not score zero.
        assert!(idf(1, 1) > 0.0);
    }

    // ---------------------------------------------------------------------------------------
    // graph_tokens
    // ---------------------------------------------------------------------------------------

    /// The finding itself: the module claims to follow verbatim entities, and the old tokenizer
    /// destroyed them. The compound must survive AND its parts must still be there, so nothing that
    /// matched before stops matching.
    #[test]
    fn a_hyphenated_identifier_survives_as_a_unit() {
        let toks = graph_tokens("auth-svc talks to widget-x");
        assert!(toks.contains(&"auth-svc".to_string()), "the entity was split: {toks:?}");
        assert!(toks.contains(&"auth".to_string()));
        assert!(toks.contains(&"svc".to_string()));
        // The case the old tokenizer lost entirely: `x` is dropped for length, so `widget-x` and
        // `widget-y` were the same token.
        assert!(toks.contains(&"widget-x".to_string()));
    }

    #[test]
    fn dotted_and_slashed_paths_survive() {
        let toks = graph_tokens("see src/overflow.rs and config.toml");
        assert!(toks.contains(&"src/overflow.rs".to_string()), "{toks:?}");
        assert!(toks.contains(&"config.toml".to_string()));
        assert!(toks.contains(&"overflow".to_string()));
    }

    /// Trailing punctuation must not become a token. A `df=1` junk token is TOP-ranked under IDF,
    /// so without the trim the walk would spend budget following a full stop.
    #[test]
    fn sentence_punctuation_does_not_become_a_token() {
        let toks = graph_tokens("that is the end. the next thing");
        assert!(!toks.iter().any(|t| t.ends_with('.')), "punctuation leaked: {toks:?}");
        assert!(toks.contains(&"end".to_string()));
    }

    /// Non-ASCII case folding. `to_ascii_lowercase` leaves these as two different tokens, and a
    /// bridge that cannot match its own other spelling is not a bridge.
    #[test]
    fn case_folds_beyond_ascii() {
        assert_eq!(graph_tokens("École"), graph_tokens("école"));
    }

    /// Every token the old tokenizer produced is still produced. This is what makes the change
    /// additive rather than a re-tuning: recall cannot fall because a match was removed.
    #[test]
    fn nothing_the_old_tokenizer_found_is_lost() {
        for s in [
            "the /login route is served by auth-svc",
            "Hello, World_1! foo",
            "quuxsvc is written in zigzaglang",
            "src/main.rs:1443 plumbs the config",
        ] {
            let new: HashSet<String> = graph_tokens(s).into_iter().collect();
            for old in crate::embed::tokens(s) {
                assert!(new.contains(&old), "{s:?}: lost {old:?} -- new set is {new:?}");
            }
        }
    }

    /// Compounds outrank their own parts for free, because they are rarer. No rule says "prefer the
    /// whole identifier" -- IDF says it, which is the point of ranking over rules.
    #[test]
    fn a_compound_outranks_its_parts() {
        let chunks: Vec<(String, String)> = (0..10)
            .map(|i| (format!("c{i}"), format!("the auth service handled request {i}")))
            .chain(std::iter::once(("hit".to_string(), "auth-svc owns the token cache".to_string())))
            .collect();
        let hits = token_follow_idf(&chunks, "what does auth-svc own", 1, 3, 64);
        assert_eq!(hits[0].id, "hit", "the compound lost to its own parts: {hits:?}");
    }

    // ---------------------------------------------------------------------------------------
    // The stopword list
    // ---------------------------------------------------------------------------------------

    /// The list is shared, so a walk term and a grep term are filtered identically. Before the shared stopword list these
    /// two disagreed on eight words and nothing said so.
    #[test]
    fn the_walk_and_grep_agree_on_what_is_filler() {
        for w in ["the", "is", "what", "their", "all", "use"] {
            assert!(!is_usable(w), "{w:?} should be filler");
            assert!(
                crate::overflow::grep_lines("a line about the widget
", w, 64).is_empty(),
                "grep still matches on the filler word {w:?}"
            );
        }
    }

    /// The two words removed by the shared stopword list must stay removed. They are not function words -- they were the
    /// filler vocabulary of this module's own fixtures, and a stopword list fitted to the fixtures
    /// is a tuned parameter that no sweep would ever catch.
    #[test]
    fn fixture_vocabulary_is_not_treated_as_filler() {
        assert!(is_usable("entirely"));
        assert!(is_usable("exclusively"));
    }

    // ---------------------------------------------------------------------------------------
    // The walk on SQLite FTS5 (the default since the FTS5 move)
    // ---------------------------------------------------------------------------------------

    fn fts(chunks: &[(String, String)], q: &str, max_chunks: usize) -> Vec<GraphHit> {
        token_follow_fts(chunks, q, 2, max_chunks, 64, crate::lexindex::TOKENIZER, true)
    }

    /// The headline property, on the new substrate: a bridge is followed however common it is.
    #[test]
    fn the_fts_walk_follows_a_bridge_at_any_frequency() {
        for df in [2usize, 3, 4, 5, 6, 8] {
            let chunks = chain_with_bridge_df(df);
            let text = fts(&chunks, "who serves the frobnicator endpoint", 12)
                .iter()
                .map(|h| h.snippet.clone())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("zigzaglang"), "bridge_df={df}: second hop not reached");
        }
    }

    /// the deterministic walk again, re-applied rather than inherited: BM25 scores are floats and a tie must break on
    /// chunk index, never on hash order.
    #[test]
    fn the_fts_walk_is_deterministic() {
        let chunks = chain_with_bridge_df(4);
        let first = fts(&chunks, "who serves the frobnicator endpoint", 6);
        for _ in 0..12 {
            assert_eq!(first, fts(&chunks, "who serves the frobnicator endpoint", 6));
        }
    }

    #[test]
    fn the_fts_walk_handles_an_empty_corpus() {
        assert!(fts(&[], "anything", 8).is_empty());
    }

    /// The shipped defaults, pinned so neither moves without a measurement behind it.
    #[test]
    fn the_default_seed_is_fts_at_eight_chunks() {
        let cfg = crate::config::Config::default();
        assert_eq!(cfg.overflow_graph_rank, "fts");
        assert_eq!(cfg.overflow_seed_max_chunks, 8);
    }
}
