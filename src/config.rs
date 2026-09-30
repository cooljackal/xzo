// SPDX-License-Identifier: Apache-2.0
//! Runtime configuration, read from `XZO_*` environment variables.

#[derive(Clone)]
pub struct Config {
    pub host: String,
    /// Explicit opt-in to bind a non-loopback address. Off by default, and xzo refuses
    /// to start without it, because every endpoint is unauthenticated.
    pub allow_remote: bool,
    pub port: u16,
    /// served model id (what clients select)
    pub model_id: String,
    /// upstream llama-server OpenAI base (the frozen core), e.g. http://127.0.0.1:8080
    pub core_url: String,
    /// the conversation store's file (XZO_DB); None = in memory only
    pub db_path: Option<String>,
    /// model context window size
    pub n_ctx: usize,
    /// reserved tokens for generation
    pub max_tokens: usize,
    /// enable performance PRAGMAs on SQLite (WAL, NORMAL sync, cache, mmap)
    pub sqlite_tuned: bool,
    /// max rounds of xzo's own recall tools per request (search / get a stored piece)
    pub tool_max_iters: usize,

    // --- The overflow layer ---
    /// master switch: XZO_OVERFLOW = off|0|false turns xzo into a plain pass-through proxy
    pub overflow: bool,
    /// overflow fires when est prompt tokens > n_ctx * this
    pub overflow_trigger: f32,
    /// seed mode: off|topk|digest|both
    pub overflow_seed: String,
    pub overflow_seed_topk: usize,
    pub overflow_seed_max_tokens: usize,
    pub overflow_seed_min_score: f32,
    /// route override: auto|targeted|exhaustive
    pub overflow_route: String,
    /// route backend: keyword|classifier
    pub overflow_route_mode: String,
    /// |-separated substrings that force the exhaustive lane
    pub overflow_global_patterns: String,
    /// external classifier id/path (unused if empty)
    pub overflow_classifier: String,
    /// external classifier endpoint (unused if empty)
    pub overflow_classifier_url: String,
    /// per-request safety cap on chunks digested on the exhaustive path (0 = unlimited)
    pub overflow_max_chunks: usize,
    /// How many over-window requests may occupy the overflow path at once. Requests
    /// beyond this WAIT; none are rejected. `0` = unlimited.
    pub overflow_max_inflight: usize,
    /// cap on total get_chunk pulls across the targeted tool loop
    pub tool_max_chunks: usize,
    /// strip tools + force an answer on the last tool-loop iteration
    pub tool_force_answer: bool,
    /// chunk-store db path; None = in-memory (sidecar of db_path is chosen by main.rs when None)
    pub chunk_db: Option<String>,
    /// LRU cap on stored conversation chunks; 0 = unlimited. Default finite: the chunk
    /// store persists to disk by default and holds raw conversation text, so an unlimited default
    /// grows without bound. At the ~256-token chunk default this cap is roughly a few hundred MB;
    /// recently-inserted chunks are protected from eviction (the in-flight guard).
    pub chunk_max_entries: usize,
    /// Retention limit in days for stored conversation chunks. `0` = keep until the LRU
    /// cap evicts, which is the pre-0.1.1 behaviour and stays the default: a retention policy that
    /// deletes data by surprise on upgrade is worse than none.
    pub chunk_ttl_days: u64,
    /// per-chunk token target; a message larger than this is sub-split
    pub chunk_target: usize,

    // --- Overflow cost ---
    /// max concurrent digest calls in the exhaustive burst / seed (1 = serial, old behavior)
    pub overflow_digest_concurrency: usize,
    /// summarize ahead, in the background, what the eventual overflow will need (see prewarm_all)
    pub overflow_prewarm: bool,
    /// Fraction of n_ctx above which background pre-warm kicks in (0 = from turn 1). Default 0.25.
    ///
    /// The trade: start later and there is less time to catch up before the window fills; start
    /// earlier and summaries are written for conversations that never overflow, using the one model
    /// slot the user's replies also need. Measured: at 0.5 the overflow turn still
    /// had a summary left to write and waited 17–42 s; at 0.25 and at 0 it had none (4–16 s).
    /// 0.25 matches 0 without summarizing short chats that never grow.
    pub overflow_prewarm_ratio: f32,
    /// targeted-lane seed source: graph (token-follow, default) | grep | raw | summary
    pub overflow_seed_targeted: String,
    /// offer the model a `read_all` tool in the targeted loop to escalate to a full exhaustive digest
    pub overflow_escalate: bool,

    // --- Multi-hop lookup (query-anchored expansion) ---
    /// chunk-to-chunk expansion hops from the query anchors (0 = single-level, no expansion)
    pub overflow_seed_hops: usize,
    /// neighbors pulled per frontier chunk each hop
    pub overflow_seed_hop_fanout: usize,
    /// min chunk-to-chunk cosine to keep a neighbor on the path
    pub overflow_seed_hop_min_sim: f32,
    /// overall cap on chunks assembled onto the seed path
    pub overflow_seed_max_chunks: usize,
    /// token-follow ("graph" seed) distinctiveness: skip tokens appearing in > this fraction of chunks
    pub overflow_graph_max_df: f32,
    /// How the graph seed finds and ranks what to follow:
    /// - `fts` (default): the ranked walk on SQLite FTS5, with the starting chunks ranked by BM25.
    ///   Same recall as `idf` at HALF the chunk budget — 100% at 6 chunks where `idf` needs 12.
    /// - `idf`: the same walk on the hand-written keyword code. Everything measured before the FTS5
    ///   move used it, so it stays reachable; it is also the fallback if FTS5 cannot build.
    /// - `df`: the pre-0.1.1 fractional cutoff, which no value of can win.
    pub overflow_graph_rank: String,
    /// Fence recalled text so it cannot pose as instruction. **Default on** since the
    /// compaction bench ran with the fence actually wired into the agent recap. On the agent
    /// path the fence keeps the user's earlier requests in force and marks tool-result text as data;
    /// on the extraction path it marks all recalled notes as data. `XZO_OVERFLOW_FENCE_NOTES=0` = off.
    pub overflow_fence_notes: bool,

    // --- Compaction (agent and chat traffic) ---
    /// overflow strategy mode: auto|compaction|extraction (auto uses the agentic-markers gate)
    pub overflow_mode: String,
    /// token budget for the verbatim pinned tail in compaction
    pub compact_tail_tokens: usize,
    /// The tail cut moves in jumps of this many tokens instead of every turn (0 = every turn). Between
    /// jumps each turn only appends to the previous prompt, so the model server reuses what it
    /// already read instead of re-reading ~5k tokens (~3 s a turn on the 9B). The tail then
    /// holds `compact_tail_tokens - this` to `compact_tail_tokens`, never more. Measured: 40% less
    /// re-read per turn, 20/20 answers.
    pub compact_tail_step_tokens: usize,
    /// cap on the compaction recap block
    pub compact_recap_max_tokens: usize,
    /// N recent-old summaries kept hot before eviction to the cold store (Stage 2)
    pub compact_pool_slots: usize,
    /// evicted-old representation: pointer|summary (Stage 2). `pointer` (default) puts one "call
    /// search_chunks/get_chunk to recall them" line in the recap; `summary` re-summarizes all the older
    /// summaries into one, a fresh uncached core call on every over-window turn (~5 s on the 9B).
    /// Measured: pointer answers 20/20, same as summary, without that call.
    pub compact_evicted: String,
    /// offer get_chunk/search_chunks + run the intercept loop in compaction (Stage 2)
    pub compact_recall_tools: bool,
    /// tool outputs larger than this get a pointer+head stub (Stage 2)
    pub compact_externalize_min_tokens: usize,
    /// Store each compressed tool result together with the call that produced it, e.g.
    /// `[read_file path=services/metrics/config.toml]` then the output. Default ON: without it the
    /// call's arguments are dropped, and memory cannot tell look-alike outputs apart (measured on the compaction benchmark). Off exists so the benchmark can measure the difference.
    pub compact_label_tool_results: bool,

    // --- Paced-replay ("faux") streaming — re-emit the buffered answer as small SSE frames ---
    /// chars/sec pacing for replay streaming; 0 disables (single-chunk behavior)
    pub replay_stream_cps: usize,
    /// cap on total artificial delay (ms); overflow flushes the rest instantly
    pub replay_stream_max_ms: u64,
    /// max chars per replayed SSE frame (word-granularity ceiling)
    pub replay_stream_frame_chars: usize,

    // --- Token counting, streaming, chat templates ---
    /// XZO_EXACT_TOKENS: count prompt tokens exactly for the overflow gate, instead of a chars
    /// estimate that under-counts code/JSON by up to ~2x. Default on. With it on, xzo uses
    /// `XZO_TOKENIZER` if set, else the core's `/tokenize`, and REFUSES TO START when it has neither
    ///. `off` accepts the estimate knowingly.
    pub exact_tokens: bool,
    /// XZO_TOKENIZER: path to the core model's Hugging Face `tokenizer.json`. xzo then counts tokens
    /// itself — exact on any server, including ones with no `/tokenize` (Ollama, MLX, LM Studio).
    pub tokenizer: Option<String>,
    /// stream normalization: on/1/true enable; off/0/false disable
    pub stream_normalize: bool,
    /// strict chat-template message normalization: auto (reactive ladder) | off (verbatim/today) |
    /// convert | hoist | merge-user0 (force a single strategy, skipping the ladder).
    pub normalize_system: crate::normalize::Strategy,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: env("XZO_HOST", "127.0.0.1"),
            allow_remote: env("XZO_ALLOW_REMOTE", "0") == "1",
            port: env("XZO_PORT", "8000").parse().unwrap_or(8000),
            model_id: env("XZO_MODEL", "xzo-qwen3.5-9b"),
            core_url: env("XZO_CORE_URL", "http://127.0.0.1:8080"),
            db_path: match env("XZO_DB", "xzo_memory.sqlite").as_str() {
                "" | "none" | "0" => None,
                p => Some(p.to_string()),
            },
            n_ctx: env("XZO_NCTX", "8192").parse().unwrap_or(8192),
            max_tokens: env("XZO_MAX_TOKENS", "1024").parse().unwrap_or(1024),
            sqlite_tuned: !matches!(env("XZO_SQLITE_TUNED", "").to_lowercase().as_str(), "0" | "false"),
            tool_max_iters: env("XZO_TOOL_MAX_ITERS", "2").parse().unwrap_or(2),
            overflow: !matches!(env("XZO_OVERFLOW", "auto").to_lowercase().as_str(), "0" | "false" | "off"),
            overflow_trigger: env("XZO_OVERFLOW_TRIGGER", "0.9").parse().unwrap_or(0.9),
            overflow_seed: env("XZO_OVERFLOW_SEED", "both"),
            overflow_seed_topk: env("XZO_OVERFLOW_SEED_TOPK", "3").parse().unwrap_or(3),
            overflow_seed_max_tokens: env("XZO_OVERFLOW_SEED_MAX_TOKENS", "1024").parse().unwrap_or(1024),
            overflow_seed_min_score: env("XZO_OVERFLOW_SEED_MIN_SCORE", "0").parse().unwrap_or(0.0),
            overflow_route: env("XZO_OVERFLOW_ROUTE", "auto"),
            overflow_route_mode: env("XZO_OVERFLOW_ROUTE_MODE", "keyword"),
            overflow_global_patterns: env("XZO_OVERFLOW_GLOBAL_PATTERNS", "how many|how much|every|all of|list all|count|total|sum of|summarize|overall|across all|each "),
            overflow_classifier: env("XZO_OVERFLOW_CLASSIFIER", ""),
            overflow_classifier_url: env("XZO_OVERFLOW_CLASSIFIER_URL", ""),
            overflow_max_chunks: env("XZO_OVERFLOW_MAX_CHUNKS", "64").parse().unwrap_or(64),
            overflow_max_inflight: env("XZO_OVERFLOW_MAX_INFLIGHT", "4").parse().unwrap_or(4),
            tool_max_chunks: env("XZO_TOOL_MAX_CHUNKS", "16").parse().unwrap_or(16),
            tool_force_answer: !matches!(env("XZO_TOOL_FORCE_ANSWER", "true").to_lowercase().as_str(), "0" | "false" | "off"),
            chunk_db: match env("XZO_CHUNK_DB", "").as_str() { "" | "none" | "0" => None, p => Some(p.to_string()) },
            chunk_max_entries: env("XZO_CHUNK_MAX_ENTRIES", "50000").parse().unwrap_or(50000),
            chunk_ttl_days: env("XZO_CHUNK_TTL_DAYS", "0").parse().unwrap_or(0),
            chunk_target: env("XZO_CHUNK_TARGET", "1500").parse().unwrap_or(1500),
            overflow_digest_concurrency: env("XZO_OVERFLOW_DIGEST_CONCURRENCY", "4").parse().unwrap_or(4),
            overflow_prewarm: !matches!(env("XZO_OVERFLOW_PREWARM", "on").to_lowercase().as_str(), "0" | "false" | "off"),
            overflow_prewarm_ratio: env("XZO_OVERFLOW_PREWARM_RATIO", "0.25").parse().unwrap_or(0.25),
            overflow_seed_targeted: env("XZO_OVERFLOW_SEED_TARGETED", "graph"),
            overflow_escalate: !matches!(env("XZO_OVERFLOW_ESCALATE", "on").to_lowercase().as_str(), "0" | "false" | "off"),
            overflow_seed_hops: env("XZO_OVERFLOW_SEED_HOPS", "2").parse().unwrap_or(2),
            overflow_seed_hop_fanout: env("XZO_OVERFLOW_SEED_HOP_FANOUT", "2").parse().unwrap_or(2),
            overflow_seed_hop_min_sim: env("XZO_OVERFLOW_SEED_HOP_MIN_SIM", "0.15").parse().unwrap_or(0.15),
            // Back to 8, the 0.1 value. It was raised to 12 only because the hand-written walk ran out
            // of budget mid-chain (79% at 8). With BM25 choosing the starting chunks the walk reaches
            // 100% at 6, so 8 carries two chunks of margin and 12 would be four chunks of noise the
            // model has to read.
            overflow_seed_max_chunks: env("XZO_OVERFLOW_SEED_MAX_CHUNKS", "8").parse().unwrap_or(8),
            overflow_graph_max_df: env("XZO_OVERFLOW_GRAPH_MAX_DF", "0.15").parse().unwrap_or(0.15),
            overflow_graph_rank: env("XZO_OVERFLOW_GRAPH_RANK", "fts"),
            overflow_fence_notes: env("XZO_OVERFLOW_FENCE_NOTES", "1") == "1",
            overflow_mode: env("XZO_OVERFLOW_MODE", "auto"),
            compact_tail_tokens: env("XZO_COMPACT_TAIL_TOKENS", "2000").parse().unwrap_or(2000),
            compact_tail_step_tokens: env("XZO_COMPACT_TAIL_STEP_TOKENS", "1000").parse().unwrap_or(1000),
            compact_recap_max_tokens: env("XZO_COMPACT_RECAP_MAX_TOKENS", "1024").parse().unwrap_or(1024),
            compact_pool_slots: env("XZO_COMPACT_POOL_SLOTS", "8").parse().unwrap_or(8),
            compact_evicted: env("XZO_COMPACT_EVICTED", "pointer"),
            compact_recall_tools: !matches!(env("XZO_COMPACT_RECALL_TOOLS", "on").to_lowercase().as_str(), "0" | "false" | "off"),
            compact_externalize_min_tokens: env("XZO_COMPACT_EXTERNALIZE_MIN_TOKENS", "2000").parse().unwrap_or(2000),
            compact_label_tool_results: !matches!(env("XZO_COMPACT_LABEL_TOOL_RESULTS", "on").to_lowercase().as_str(), "0" | "false" | "off"),
            replay_stream_cps: env("XZO_REPLAY_STREAM_CPS", "800").parse().unwrap_or(800),
            replay_stream_max_ms: env("XZO_REPLAY_STREAM_MAX_MS", "4000").parse().unwrap_or(4000),
            replay_stream_frame_chars: env("XZO_REPLAY_STREAM_FRAME_CHARS", "24").parse().unwrap_or(24),
            exact_tokens: !matches!(env("XZO_EXACT_TOKENS", "on").to_lowercase().as_str(), "0" | "false" | "off"),
            tokenizer: match env("XZO_TOKENIZER", "").as_str() { "" => None, p => Some(p.to_string()) },
            stream_normalize: !matches!(env("XZO_STREAM_NORMALIZE", "on").to_lowercase().as_str(), "0" | "false" | "off"),
            normalize_system: crate::normalize::Strategy::from_env(&env("XZO_NORMALIZE_SYSTEM", "auto")),
        }
    }
}

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // `Config::default()` reads the whole process environment, and `set_var`/`remove_var` mutate
    // shared global state (UB under parallel access in the 2024 edition). Every test that reads a
    // default or touches env must hold this lock so the suite is serialized on env, not racy (C3).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        // Ignore poisoning: a panicking test still leaves env in a known state (each test cleans up).
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn defaults_parse_without_env() {
        let _env = env_guard();
        let cfg = Config::default();
        assert_eq!(cfg.overflow, true);
        assert_eq!(cfg.overflow_trigger, 0.9);
        assert_eq!(cfg.overflow_seed, "both");
        assert_eq!(cfg.overflow_seed_topk, 3);
        assert_eq!(cfg.chunk_target, 1500);
        assert_eq!(cfg.tool_force_answer, true);
        assert!(cfg.chunk_db.is_none());
        // The conversation store has a finite cap, so on-disk growth is bounded.
        assert_eq!(cfg.chunk_max_entries, 50000);
    }

    #[test]
    fn overflow_off_via_env() {
        let _env = env_guard();
        std::env::set_var("XZO_OVERFLOW", "off");
        let cfg = Config::default();
        assert!(!cfg.overflow);
        std::env::remove_var("XZO_OVERFLOW");
    }

    #[test]
    fn compaction_defaults() {
        let _env = env_guard();
        let cfg = Config::default();
        assert_eq!(cfg.overflow_mode, "auto");
        assert_eq!(cfg.compact_tail_tokens, 2000);
        assert_eq!(cfg.compact_tail_step_tokens, 1000);
        assert_eq!(cfg.compact_recap_max_tokens, 1024);
        assert_eq!(cfg.compact_recall_tools, true);
    }

    #[test]
    fn replay_stream_defaults() {
        let _env = env_guard();
        let cfg = Config::default();
        assert_eq!(cfg.replay_stream_cps, 800);
        assert_eq!(cfg.replay_stream_max_ms, 4000);
        assert_eq!(cfg.replay_stream_frame_chars, 24);
    }

    #[test]
    fn token_and_stream_defaults() {
        let _env = env_guard();
        let cfg = Config::default();
        assert_eq!(cfg.exact_tokens, true);
        assert!(cfg.tokenizer.is_none());
        assert_eq!(cfg.stream_normalize, true);
    }

    #[test]
    fn normalize_defaults_and_env_overrides() {
        use crate::normalize::Strategy;
        let _env = env_guard();
        std::env::remove_var("XZO_NORMALIZE_SYSTEM");
        let cfg = Config::default();
        assert_eq!(cfg.normalize_system, Strategy::Verbatim); // "auto" -> Verbatim (ladder start)

        std::env::set_var("XZO_NORMALIZE_SYSTEM", "hoist");
        assert_eq!(Config::default().normalize_system, Strategy::Hoist);
        std::env::set_var("XZO_NORMALIZE_SYSTEM", "off");
        assert_eq!(Config::default().normalize_system, Strategy::Off);
        std::env::remove_var("XZO_NORMALIZE_SYSTEM");
    }
}
