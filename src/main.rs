// SPDX-License-Identifier: Apache-2.0
//! xzo — an OpenAI-compatible proxy in front of your own model (the "core"). When a conversation
//! outgrows the core's context window, xzo files the older part away in a local store and puts back
//! only what the current turn needs; under the window it passes requests straight through.

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{extract::{DefaultBodyLimit, State}, routing::{get, post}, Json, Router};
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use xzo::inject::flatten;
use xzo::router::Route;
use xzo::{chunks, config, embed, inject, normalize, overflow, router, stats, tokcount};

struct App {
    cfg: config::Config,
    /// The conversation store: every piece of an over-window conversation, its search vector and
    /// its summary, keyed by a fingerprint of its text.
    chunk_store: Mutex<chunks::ChunkStore>,
    stats: stats::Stats,
    http: reqwest::Client,
    /// Prompt token counts by content fingerprint, so the unchanged start of a conversation is
    /// counted once and only new messages cost anything.
    token_cache: Mutex<HashMap<String, usize>>,
    /// How the overflow gate counts tokens, decided at startup (see `choose_counter`).
    counter: tokcount::Counter,
    /// Strict chat-template normalization: the winning transform per model id, learned on the first
    /// template rejection and applied up front afterwards.
    normalize_strategy_cache: Mutex<HashMap<String, normalize::Strategy>>,
    /// Bounds how many over-window requests may be in the overflow path at once. It QUEUES rather
    /// than rejects: no request fails, it only waits, so one client cannot make xzo saturate the
    /// core it fronts.
    overflow_gate: tokio::sync::Semaphore,
    /// At most ONE background summarizer at a time (see `spawn_prewarm`). Shared because the permit
    /// travels into the spawned task and is released when it ends.
    prewarm_gate: Arc<tokio::sync::Semaphore>,
}

const HELP: &str = "\
xzo 0.1 — an OpenAI-compatible proxy in front of your own model (the 'core'). It lets a conversation
keep going PAST the core's context window: when a request is over-window, xzo files the older part
into a local store and puts back only what the current question needs. Configured entirely by
environment variables.

Core:    XZO_CORE_URL (http://127.0.0.1:8080)   XZO_MODEL   XZO_HOST (127.0.0.1)   XZO_PORT (8000)
Window:  XZO_NCTX (8192) — MUST equal the core's context size (llama-server -c, Ollama num_ctx).
                           Too high overruns the core's real window; too low makes xzo step in
                           when it doesn't need to. Checked at startup when the core reports it.
Tokens:  XZO_TOKENIZER — path to the model's tokenizer.json. Required for servers without
                           /tokenize (Ollama, MLX, LM Studio); xzo refuses to start without a way
                           to count exactly. XZO_EXACT_TOKENS=off accepts an estimate instead.
Overflow: XZO_OVERFLOW (on)   XZO_OVERFLOW_TRIGGER (0.9) — step in when prompt tokens > NCTX*trigger
Storage: XZO_DB (xzo_memory.sqlite) — the conversation store, RAW conversation text in plaintext,
         kept across restarts. none = in memory only. XZO_CHUNK_MAX_ENTRIES (50000) caps it;
         XZO_CHUNK_TTL_DAYS (0) expires old pieces on /prune.
Template: XZO_NORMALIZE_SYSTEM (auto) — adapts the message layout when a strict chat template
                           rejects a client's mid-conversation system message. off = verbatim.
Trace:   XZO_TRACE_CORE (off) — off|summary|full. Print the prompt xzo actually sends the core, to
                           stderr.
Endpoints: /v1/chat/completions  /v1/models  /health  /stack (counters)  /prune

IMPORTANT — model reasoning/thinking:
  xzo does not manage the core's chain-of-thought. Many small models reason worse at agent and coding
  tasks with thinking ON, and some default it ON. Disable it AT THE CORE (llama.cpp:
  `--reasoning off`); how depends on the model family.
";

#[tokio::main]
async fn main() {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        print!("{HELP}");
        return;
    }
    let cfg = config::Config::default();
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let served = cfg.model_id.clone();
    let core = cfg.core_url.clone();

    let embedder = embed::build();
    let chunk_db = cfg.chunk_db.clone().or_else(|| cfg.db_path.clone());
    let chunk_store = chunks::ChunkStore::new(
        embedder.clone(),
        chunk_db.as_deref(),
        cfg.sqlite_tuned,
        cfg.chunk_max_entries,
    );
    println!(
        "memory: embedder={}  db={:?}  overflow={}  trigger={}",
        embedder.name(), chunk_db, cfg.overflow, cfg.overflow_trigger
    );

    let max_inflight = cfg.overflow_max_inflight;
    let http = reqwest::Client::new();
    let counter = match choose_counter(&cfg, &http).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("xzo: {e}");
            std::process::exit(2);
        }
    };
    println!("tokens: counted by {}", counter.name());
    let app = Arc::new(App {
        cfg,
        chunk_store: Mutex::new(chunk_store),
        stats: stats::Stats::new(),
        http,
        token_cache: Mutex::new(HashMap::new()),
        counter,
        normalize_strategy_cache: Mutex::new(HashMap::new()),
        prewarm_gate: Arc::new(tokio::sync::Semaphore::new(1)),
        // 0 means unlimited, expressed as a permit count nothing will exhaust rather than as a
        // branch at every call site.
        overflow_gate: tokio::sync::Semaphore::new(if max_inflight == 0 {
            tokio::sync::Semaphore::MAX_PERMITS
        } else {
            max_inflight
        }),
    });

    let router = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/stack", get(stack))
        .route("/prune", post(prune))
        // Explicit request-body cap: above realistic long-context chat payloads, well below
        // memory exhaustion.
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(app.clone());

    // Refuse to publish an unauthenticated service by accident. Checked BEFORE the core preflight:
    // a misconfigured bind should fail on its own terms.
    if let Err(e) = check_bind_exposure(&app.cfg.host, app.cfg.allow_remote) {
        eprintln!("xzo: {e}");
        std::process::exit(2);
    }
    if app.cfg.allow_remote && !is_loopback_host(&app.cfg.host) {
        eprintln!(
            "xzo: WARNING serving on {} with XZO_ALLOW_REMOTE=1 -- no endpoint is authenticated, and /prune is destructive.",
            app.cfg.host
        );
    }

    // Preflight the core, so an unreachable core or a mismatched XZO_NCTX is loud at startup
    // instead of a mysterious per-request failure.
    startup_core_check(&app.http, &app.cfg.core_url, app.cfg.n_ctx).await;

    println!("xzo :{addr}  ->  core {core}  (served id: {served})");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    axum::serve(listener, router).await.expect("serve");
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [ { "id": app.cfg.model_id, "object": "model", "owned_by": "xzo" } ]
    }))
}

async fn stack(State(app): State<Arc<App>>) -> Json<Value> {
    let mut snap = app.stats.snapshot();
    {
        let cs = app.chunk_store.lock().unwrap();
        snap["chunks"] = json!({
            "entries": cs.len(),
            "evictions": cs.evictions,
            "insert_failures": cs.insert_failures,
            // Expected to be 0 forever. A non-zero value means two different pieces shared a
            // fingerprint: a deliberate collision or a much worse bug.
            "hash_collisions": cs.hash_collisions,
        });
    }
    // Whether the background summarizer is running right now: the model server is busy on xzo's
    // own work.
    snap["prewarm_busy"] = json!(app.prewarm_gate.available_permits() == 0);
    Json(snap)
}

/// Reclaim the conversation store: expire pieces older than `XZO_CHUNK_TTL_DAYS`, then shrink to
/// the size cap (or, when unlimited, empty it). Pieces a running request just stored are protected.
async fn prune(State(app): State<Arc<App>>) -> Json<Value> {
    let (before, after, expired) = {
        let mut cs = app.chunk_store.lock().unwrap();
        let before = cs.len();
        let target = if app.cfg.chunk_max_entries > 0 {
            (app.cfg.chunk_max_entries * 9 / 10).max(1)
        } else {
            0
        };
        // Age first, then size: a piece that is both old and cold is deleted for the stated reason.
        let expired = cs.evict_older_than(app.cfg.chunk_ttl_days);
        cs.evict_to(target);
        (before, cs.len(), expired)
    };
    // `expired` is a subset of `removed`, reported separately so an operator can tell a retention
    // policy doing its job from the size cap doing its job.
    Json(json!({"chunks": {"before": before, "after": after, "removed": before - after, "expired": expired}}))
}

/// The clean query = flattened content of the last `user` message.
fn last_user(messages: &[Value]) -> String {
    for m in messages.iter().rev() {
        if m.get("role").and_then(|r| r.as_str()) == Some("user") {
            return flatten(m.get("content").unwrap_or(&Value::Null));
        }
    }
    String::new()
}

/// The messages to summarize ahead of time, split EXACTLY the way the overflow path this request
/// will take stores them — so pre-warm's summaries land under the keys that path looks up.
///
/// This is the bug pre-warm had: it always used the document-Q&A split, while agent traffic
/// overflows through compaction. Once compaction started labelling each tool result with its call
/// (`[read_file path=…]`), every tool result's text — and so its cache key — differed between the
/// two, and every pre-warmed tool-result summary was wasted.
///
/// For compaction: EVERY message, including the recent tail and the newest one. Those age into the
/// compressed middle turn by turn; summarizing them now is what leaves nothing to do when the
/// conversation crosses the window.
fn prewarm_middle(
    messages: &[Value],
    strategy: router::ContextStrategy,
    cfg: &config::Config,
) -> Vec<String> {
    match strategy {
        router::ContextStrategy::Compaction => {
            overflow::stored_texts(messages, cfg.compact_label_tool_results)
        }
        router::ContextStrategy::Extraction => extraction_middle(messages),
    }
}

/// Summarize `middle` in the background; the calling request does not wait.
///
/// ONE background summarizer at a time. If one is already running, skip rather than queue: the
/// running one, or the next turn's, catches this backlog up, because each run summarizes everything
/// still missing. Queuing would stack summaries in the model server's single slot and delay the
/// user's next reply.
fn spawn_prewarm(app: &Arc<App>, middle: Vec<String>) {
    if middle.is_empty() {
        return;
    }
    match app.prewarm_gate.clone().try_acquire_owned() {
        Ok(permit) => {
            app.stats.incr("overflow.prewarm_started");
            let bg = app.clone();
            tokio::spawn(async move {
                let n = overflow::prewarm_all(
                    &bg.http, &bg.cfg.core_url, &bg.cfg.model_id,
                    &bg.chunk_store, &bg.stats, &middle, bg.cfg.chunk_target, bg.cfg.overflow_max_chunks,
                )
                .await;
                bg.stats.add("overflow.prewarm_background_digests", n as u64);
                drop(permit);
            });
        }
        Err(_) => app.stats.incr("overflow.prewarm_busy_skips"),
    }
}

/// The EXTRACTION-lane middle (bit-for-bit the pre-2.3 behavior): every non-system message except
/// the last user turn (which is kept verbatim and carried as the `query`). Used only when the
/// strategy is `Extraction`, so the doc-Q&A path is unchanged.
fn extraction_middle(messages: &[Value]) -> Vec<String> {
    let last_user_idx = messages
        .iter()
        .rposition(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"));
    let mut middle = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "system" || Some(i) == last_user_idx {
            continue; // kept verbatim, never chunked
        }
        let text = flatten(m.get("content").unwrap_or(&Value::Null));
        if !text.trim().is_empty() {
            middle.push(text);
        }
    }
    middle
}


/// The overflow gate's prompt token count: every message's flattened content + tool_calls + the
/// tools JSON, counted by the startup-chosen [`tokcount::Counter`], plus a small fixed per-message
/// template overhead (bias slightly conservative — over-counting fires compaction a touch early,
/// which is safe; under-counting is the bug). Counts are cached by content hash so the stable prefix
/// is counted once (delta-only).
///
/// Every counter goes through the same sum. The estimate used to take a separate path that counted
/// message content only — tool calls, a large share of an agent session, were not counted at all.
async fn count_prompt_tokens(app: &App, messages: &[Value], body: &Value) -> usize {
    let mut total = tokcount::prompt_overhead(messages.len());
    for text in tokcount::prompt_texts(messages, body.get("tools")) {
        total += cached_token_count(app, &text).await;
    }
    total
}

/// Token count of `text`, cached by content hash.
async fn cached_token_count(app: &App, text: &str) -> usize {
    if text.trim().is_empty() {
        return 0;
    }
    let h = chunks::content_hash(text);
    if let Some(&n) = app.token_cache.lock().unwrap().get(&h) {
        return n;
    }
    let n = match &app.counter {
        tokcount::Counter::Local(t) => t.count(text),
        tokcount::Counter::Estimate => inject::est_tokens(text),
        // The core answered /tokenize at startup; a failure now is transient. Over-count rather
        // than guess low, and do not cache the stand-in.
        tokcount::Counter::Core => match core_token_count(&app.http, &app.cfg.core_url, text).await {
            Some(n) => n,
            None => {
                app.stats.incr("tokens.core_count_failed");
                return 2 * inject::est_tokens(text);
            }
        },
    };
    app.token_cache.lock().unwrap().insert(h, n);
    n
}

/// Pick how the overflow gate counts tokens — deterministically, or not at all.
///
/// 1. `XZO_TOKENIZER` set: the model's own tokenizer, in xzo. A file that will not load is fatal.
/// 2. `XZO_EXACT_TOKENS=off`: the chars estimate, explicitly accepted.
/// 3. The core answers `/tokenize` (llama-server, vLLM): exact, from the core.
/// 4. The core answers but has no `/tokenize` (Ollama, MLX, LM Studio): REFUSE TO START. The only
///    thing left would be the estimate, which under-counts code and JSON by up to ~2x and lets a
///    prompt overrun the window — silently, on servers that truncate instead of erroring.
/// 5. The core is not up yet: assume `/tokenize` (the default core is llama-server); a later failure
///    over-counts rather than guesses low (see `cached_token_count`).
async fn choose_counter(cfg: &config::Config, http: &reqwest::Client) -> Result<tokcount::Counter, String> {
    if let Some(path) = &cfg.tokenizer {
        return tokcount::LocalTokenizer::load(path).map(tokcount::Counter::Local);
    }
    if !cfg.exact_tokens || !cfg.overflow {
        return Ok(tokcount::Counter::Estimate);
    }
    if core_token_count(http, &cfg.core_url, "ping").await.is_some() || !core_reachable(http, &cfg.core_url).await {
        return Ok(tokcount::Counter::Core);
    }
    Err(format!(
        "the model server at {} has no /tokenize, so xzo cannot count prompt tokens exactly. Set \
         XZO_TOKENIZER to the model's tokenizer.json (from its Hugging Face repo), or \
         XZO_EXACT_TOKENS=off to accept an estimate that can under-count and overrun the window.",
        cfg.core_url
    ))
}

/// Is the core up at all? Asks the OpenAI-standard `/v1/models`, which every compatible server has
/// — unlike `/tokenize`, which only some do, and which the reachability check used to rely on.
async fn core_reachable(http: &reqwest::Client, core_url: &str) -> bool {
    http.get(format!("{core_url}/v1/models"))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// POST `{core}/tokenize {"content": text}` and return the token count; `None` on any failure.
async fn core_token_count(http: &reqwest::Client, core_url: &str, text: &str) -> Option<usize> {
    let url = format!("{core_url}/tokenize");
    let resp = http.post(&url).json(&json!({ "content": text })).send().await.ok()?;
    let v: Value = resp.json().await.ok()?;
    v.get("tokens").and_then(|t| t.as_array()).map(|a| a.len())
}

/// Best-effort fetch of the core's real context length. Tries the common llama.cpp `/props` shape
/// (`default_generation_settings.n_ctx`, then a top-level `n_ctx`). Returns `None` when the core
/// doesn't expose it — the caller then reports the window as UNVERIFIED rather than inventing a
/// check it can't make. Family-agnostic on purpose: only warns when a number is actually present.
async fn core_context_window(http: &reqwest::Client, core_url: &str) -> Option<usize> {
    let url = format!("{core_url}/props");
    let resp = http.get(&url).send().await.ok()?;
    let v: Value = resp.json().await.ok()?;
    v.pointer("/default_generation_settings/n_ctx")
        .or_else(|| v.get("n_ctx"))
        .and_then(|n| n.as_u64())
        .map(|n| n as usize)
}

/// Is this host string one that only accepts connections from this machine?
///
/// Anything that is not loopback exposes an unauthenticated service, so the question has to be
/// decided by parsing rather than by pattern-matching a couple of familiar strings. `0.0.0.0` and
/// `::` are the obvious cases; a concrete LAN address is the easy one to type by accident.
///
/// An unresolvable name is treated as NOT loopback. That is the safe direction: refusing to start is
/// recoverable in one env var, while guessing wrong the other way silently publishes the store.
fn is_loopback_host(host: &str) -> bool {
    let h = host.trim().trim_start_matches('[').trim_end_matches(']');
    if h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match h.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => false,
    }
}

/// Refuse to publish an unauthenticated service by accident.
///
/// WHAT IS EXPOSED, precisely, and why a default is not enough on its own. No endpoint authenticates.
/// `/v1/chat/completions` can read back stored conversation text; `/stack` reports operational
/// counters; **`/prune` is destructive and takes no credential at all** — one unauthenticated POST
/// discards the memory file's contents. The only thing standing between that and the network is
/// `XZO_HOST` defaulting to `127.0.0.1`, and a default is a suggestion: changing one env var to
/// `0.0.0.0`, which is the ordinary way to make a container reachable, publishes all of it.
///
/// So the boundary is made explicit instead of assumed. Binding anything but loopback now requires
/// `XZO_ALLOW_REMOTE=1`, and the operator who sets it is told exactly what they are opening. This
/// does not add security — it stops the exposure from being silent, which is the difference between
/// an accepted risk and an unnoticed one.
///
/// The repo's standing assumption is unchanged and this does not relitigate it: ingest cannot be
/// secured at this layer and xzo assumes a cooperating operator. That assumption is about the
/// CONTENT a trusted operator sends, and it never implied the service should be reachable by anyone
/// who can route to the port.
fn check_bind_exposure(host: &str, allow_remote: bool) -> Result<(), String> {
    if is_loopback_host(host) || allow_remote {
        return Ok(());
    }
    // Built line by line rather than as one multi-line literal: a literal here inherits the source
    // indentation into the message, and this text is read by someone whose server just refused to
    // start.
    let lines = [
        format!("refusing to bind XZO_HOST={host}: xzo has NO authentication on any endpoint."),
        String::new(),
        "Binding a non-loopback address publishes, to anyone who can reach the port:".into(),
        "  - /v1/chat/completions  can read back stored conversation text".into(),
        "  - /prune                DESTRUCTIVE, and takes no credential".into(),
        "  - /stack                operational counters".into(),
        String::new(),
        "If that is what you intend -- behind your own auth proxy, or on a trusted network --".into(),
        "set XZO_ALLOW_REMOTE=1. Otherwise leave XZO_HOST at 127.0.0.1 (the default).".into(),
    ];
    Err(lines.join("\n"))
}

/// Take a slot in the overflow path, waiting if they are all busy.
///
/// The permit is held for the whole overflow run and released when the returned guard drops, so the
/// bound is on CONCURRENCY rather than on a rate. That is the right shape here: the scarce resource
/// is the core, and what exhausts it is simultaneous work, not requests per second.
///
/// Waiting is counted and the wait time recorded, because a queue nobody can see is
/// indistinguishable from a slow core — the same reason the over-window prompt check counts its overruns instead of trusting
/// the assembly to fit.
///
/// The semaphore is never closed, so `acquire` cannot fail; the `expect` documents that rather than
/// guarding against it.
async fn acquire_overflow_slot(app: &Arc<App>) -> tokio::sync::SemaphorePermit<'_> {
    if app.overflow_gate.available_permits() == 0 {
        app.stats.add("overflow.queued", 1);
    }
    let t0 = std::time::Instant::now();
    let permit = app.overflow_gate.acquire().await.expect("overflow gate is never closed");
    let waited = t0.elapsed().as_millis() as u64;
    if waited > 0 {
        app.stats.add("overflow.queue_wait_ms", waited);
    }
    permit
}

/// Startup preflight: the proxy otherwise binds and looks healthy while the core is
/// unreachable or `XZO_NCTX` silently disagrees with the core's real `-c`. Runs once before serving.
/// Never aborts — the core may come up later — but makes the state loud instead of silent.
async fn startup_core_check(http: &reqwest::Client, core_url: &str, n_ctx: usize) {
    // Reachability via the OpenAI-standard /v1/models. It used to probe /tokenize, which reported
    // every server without it (Ollama, MLX, LM Studio) as down.
    match core_reachable(http, core_url).await {
        false => {
            eprintln!(
                "core: NOT reachable at {core_url} — set XZO_CORE_URL to your model server, or start \
                 it. Requests will fail until the core responds."
            );
        }
        true => {
            match core_context_window(http, core_url).await {
                Some(win) if win != n_ctx => eprintln!(
                    "core: reachable, but XZO_NCTX={n_ctx} DISAGREES with the core's context window \
                     ({win}). Set XZO_NCTX={win} to match the core's -c, or over-window prompts will \
                     be mishandled."
                ),
                Some(win) => println!("core: reachable; context window {win} matches XZO_NCTX."),
                None => println!(
                    "core: reachable; context window UNVERIFIED (core exposes no n_ctx). Ensure \
                     XZO_NCTX={n_ctx} equals the server's context size yourself (llama-server -c; \
                     Ollama num_ctx — Ollama silently cuts the START of a longer prompt, where the \
                     system prompt is, instead of rejecting it)."
                ),
            }
        }
    }
}

/// Build the per-request `OverflowParams` from config (shared by the extraction and compaction
/// paths).
fn overflow_params(cfg: &config::Config) -> overflow::OverflowParams {
    overflow::OverflowParams {
        n_ctx: cfg.n_ctx,
        max_tokens: cfg.max_tokens as u64,
        trigger: cfg.overflow_trigger,
        chunk_target: cfg.chunk_target,
        seed: cfg.overflow_seed.clone(),
        seed_topk: cfg.overflow_seed_topk,
        seed_max_tokens: cfg.overflow_seed_max_tokens,
        seed_min_score: cfg.overflow_seed_min_score,
        max_chunks: cfg.overflow_max_chunks,
        tool_max_iters: cfg.tool_max_iters,
        tool_max_chunks: cfg.tool_max_chunks,
        force_answer: cfg.tool_force_answer,
        digest_concurrency: cfg.overflow_digest_concurrency,
        seed_targeted: cfg.overflow_seed_targeted.clone(),
        seed_hops: cfg.overflow_seed_hops,
        seed_hop_fanout: cfg.overflow_seed_hop_fanout,
        seed_hop_min_sim: cfg.overflow_seed_hop_min_sim,
        seed_max_chunks: cfg.overflow_seed_max_chunks,
        graph_max_df: cfg.overflow_graph_max_df,
        graph_rank: cfg.overflow_graph_rank.clone(),
        fence_notes: cfg.overflow_fence_notes,
        escalate: cfg.overflow_escalate,
    }
}

/// Build the per-request `CompactParams` from config.
fn compact_params(cfg: &config::Config) -> overflow::CompactParams {
    overflow::CompactParams {
        recap_max_tokens: cfg.compact_recap_max_tokens,
        pool_slots: cfg.compact_pool_slots,
        evicted: cfg.compact_evicted.clone(),
        recall_tools: cfg.compact_recall_tools,
        externalize_min_tokens: cfg.compact_externalize_min_tokens,
        tool_max_iters: cfg.tool_max_iters,
        seed_topk: cfg.overflow_seed_topk,
        seed_max_tokens: cfg.overflow_seed_max_tokens,
    }
}

/// Ask an external classifier (if configured) for the route lane; `None` -> fall back to keyword.
async fn classifier_verdict(app: &App, query: &str) -> Option<Route> {
    let cfg = &app.cfg;
    if !cfg.overflow_route_mode.eq_ignore_ascii_case("classifier") || cfg.overflow_classifier_url.is_empty() {
        return None;
    }
    let body = json!({
        "model": cfg.overflow_classifier,
        "messages": [{"role": "user", "content": format!(
            "Classify this request as 'targeted' (a specific lookup) or 'exhaustive' \
             (count / aggregate / summarize-all). Answer with one word.\n\n{query}")}],
        "max_tokens": 8, "temperature": 0,
    });
    let resp = app.http.post(&cfg.overflow_classifier_url).json(&body).send().await.ok()?;
    let v: Value = resp.json().await.ok()?;
    let text = v.pointer("/choices/0/message/content").and_then(|c| c.as_str()).unwrap_or("");
    router::parse_classifier_verdict(text)
}

/// Build the HTTP response for a buffered answer. Errors / non-stream requests -> JSON. For a
/// `stream:true` client, xzo has the FULLY-processed answer already (post recall loop + write-gate),
/// so it re-emits it as PACED-REPLAY ("faux") streaming: many small `chat.completion.chunk` delta
/// frames with a short inter-frame delay, a "typing" effect — no true token streaming, no
/// architectural change. Text is split on word boundaries (reassembles exactly); pacing is
/// `XZO_REPLAY_STREAM_CPS` chars/sec, capped by `XZO_REPLAY_STREAM_MAX_MS` total (then flushed
/// instantly). `cps=0` restores the legacy single-chunk emission. A tool_calls answer (e.g. a
/// compaction passthrough of a client tool call) is emitted as one well-formed delta (no pacing).
fn respond(out: Value, stream: bool, cfg: &config::Config, stats: &stats::Stats) -> Response {
    use std::time::Duration;
    if out.get("error").is_some() {
        return Json(out).into_response();
    }
    if !stream {
        return Json(out).into_response();
    }

    let model = cfg.model_id.clone();
    let msg = out.pointer("/choices/0/message").cloned().unwrap_or(Value::Null);
    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
    let tool_arr = msg.get("tool_calls").and_then(|t| t.as_array()).cloned().unwrap_or_default();
    let has_tool_calls = !tool_arr.is_empty();

    let chunk_event = |delta: Value, finish: Value| {
        Event::default().data(
            json!({"id": "xzo-chat", "object": "chat.completion.chunk", "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
                .to_string(),
        )
    };

    let mut frames: Vec<(Event, Duration)> = Vec::new();
    if !cfg.stream_normalize {
        // Normalization OFF (XZO_STREAM_NORMALIZE=off): minimal single-chunk emission of the buffered
        // answer — NO clean multi-frame reframing, NO pacing. This is the A/B "off" arm; the clean
        // role->content->finish->[DONE] framing guarantee is deliberately not applied. (The true
        // raw-passthrough arm — never buffering — is a live-only diagnostic in bin/sse_capture.)
        stats.incr("stream.unnormalized");
        let delta = if has_tool_calls {
            json!({"role": "assistant", "tool_calls": tool_arr})
        } else {
            json!({"role": "assistant", "content": content})
        };
        let finish = if has_tool_calls { json!("tool_calls") } else { json!("stop") };
        frames.push((chunk_event(delta, Value::Null), Duration::ZERO));
        frames.push((chunk_event(json!({}), finish), Duration::ZERO));
    } else if has_tool_calls {
        // Non-text (tool_calls) answer: one delta carrying the (index-tagged) tool_calls, then stop.
        stats.incr("stream.toolcalls");
        let indexed: Vec<Value> = tool_arr
            .iter()
            .enumerate()
            .map(|(i, tc)| {
                let mut t = tc.clone();
                if t.get("index").is_none() {
                    t["index"] = json!(i);
                }
                t
            })
            .collect();
        frames.push((chunk_event(json!({"role": "assistant", "tool_calls": indexed}), Value::Null), Duration::ZERO));
        frames.push((chunk_event(json!({}), json!("tool_calls")), Duration::ZERO));
    } else if cfg.replay_stream_cps == 0 || content.is_empty() {
        // Disabled (cps=0) or empty answer: legacy single-chunk emission.
        stats.incr("stream.single");
        frames.push((chunk_event(json!({"role": "assistant", "content": content}), Value::Null), Duration::ZERO));
        frames.push((chunk_event(json!({}), json!("stop")), Duration::ZERO));
    } else {
        // Paced replay: an immediate role frame, then word-boundary content frames paced to
        // ~cps chars/sec (total artificial delay capped by max_ms), then stop.
        stats.incr("stream.replayed");
        frames.push((chunk_event(json!({"role": "assistant"}), Value::Null), Duration::ZERO));
        let pieces = inject::split_for_replay(&content, cfg.replay_stream_frame_chars.max(1));
        stats.add("stream.frames", pieces.len() as u64);
        let mut budget_ms = cfg.replay_stream_max_ms;
        for p in pieces {
            let ms = (p.chars().count() as f64 * 1000.0 / cfg.replay_stream_cps as f64) as u64;
            let ms = ms.min(budget_ms);
            budget_ms = budget_ms.saturating_sub(ms);
            frames.push((chunk_event(json!({"content": p}), Value::Null), Duration::from_millis(ms)));
        }
        frames.push((chunk_event(json!({}), json!("stop")), Duration::ZERO));
    }
    frames.push((Event::default().data("[DONE]"), Duration::ZERO));

    // Pace the frames: sleep the frame's delay (non-blocking) before yielding it.
    let s = futures::stream::iter(frames).then(|(e, d)| async move {
        if !d.is_zero() {
            tokio::time::sleep(d).await;
        }
        Ok::<Event, std::convert::Infallible>(e)
    });
    Sse::new(s).into_response()
}

async fn chat(State(app): State<Arc<App>>, Json(mut body): Json<Value>) -> Response {
    app.stats.incr("core.requests");
    let messages: Vec<Value> = body
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();
    let query = last_user(&messages);
    let stream_req = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

    // --- The overflow gate. Over-window requests are routed by traffic type: Compaction (agent and
    //     chat traffic) keeps the recent turns verbatim and replaces the older ones with a recap;
    //     Extraction (document Q&A) answers from notes. Under-window requests pass through, and
    //     the background summarizer works ahead. ---
    if app.cfg.overflow && !query.is_empty() {
        let c = overflow::classify_stepped(
            &messages,
            app.cfg.compact_tail_tokens,
            app.cfg.compact_tail_step_tokens,
            app.cfg.compact_label_tool_results,
        );
        let request_has_tools = body
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);
        // The gate decision uses the exact count; `c.total` (an estimate) only sizes the split.
        let total = count_prompt_tokens(&app, &messages, &body).await;
        app.stats.add("overflow.gate_tokens", total as u64);
        let over = overflow::should_overflow(total, app.cfg.n_ctx, app.cfg.overflow_trigger);
        let strategy = router::strategy(
            &app.cfg.overflow_mode,
            request_has_tools,
            c.has_tool_calls,
            c.has_tool_messages,
        );

        if over && strategy == router::ContextStrategy::Compaction {
            app.stats.incr("compaction.requests");
            if c.middle.is_empty() {
                // The system prompt and the latest turn alone are over the window: there is no
                // older part to compress, and only a larger core context helps.
                app.stats.incr("compaction.turn1_system_over_window");
                eprintln!(
                    "compaction: nothing to compress but the prompt is over-window ({total} tok > {}*{}); \
                     raise the core's context size and XZO_NCTX.",
                    app.cfg.n_ctx, app.cfg.overflow_trigger
                );
            }
            let tail_tokens: usize = c
                .tail
                .iter()
                .map(|m| inject::est_tokens(&flatten(m.get("content").unwrap_or(&Value::Null))))
                .sum();
            app.stats.add("compaction.tail_tokens_kept", tail_tokens as u64);
            app.stats.add("compaction.tail_msgs_kept", c.tail.len() as u64);

            let params = overflow_params(&app.cfg);
            let cparams = compact_params(&app.cfg);
            let client_tools = body.get("tools").cloned().unwrap_or(Value::Null);
            let _gate = acquire_overflow_slot(&app).await;
            let out = overflow::run_compaction(
                &app.http, &app.cfg.core_url, &app.cfg.model_id,
                &app.chunk_store, &app.stats,
                &c.system, &c.middle, &c.tail, &client_tools, &query, &params, &cparams,
                app.cfg.normalize_system, &app.normalize_strategy_cache,
            )
            .await;
            // Keep summarizing ahead after the overflow too: each later turn pushes the oldest
            // verbatim message into the summarized part, and it should already be summarized.
            if app.cfg.overflow_prewarm {
                spawn_prewarm(&app, prewarm_middle(&messages, strategy, &app.cfg));
            }
            return respond(out, stream_req, &app.cfg, &app.stats);
        } else if over {
            let middle = extraction_middle(&messages);
            if !middle.is_empty() {
                let verdict = classifier_verdict(&app, &query).await;
                let route = router::route(
                    &query,
                    &app.cfg.overflow_route,
                    &app.cfg.overflow_route_mode,
                    &app.cfg.overflow_global_patterns,
                    verdict,
                );
                // The "how many / list all" patterns are English phrases, so a query in another
                // language falls through to a plain lookup without a decision being made. Counted,
                // because the answer that follows looks exactly like a right one.
                if verdict.is_none() && router::keyword_patterns_cannot_match(&query) {
                    app.stats.add("router.no_keyword_signal", 1);
                }
                let params = overflow_params(&app.cfg);
                let _gate = acquire_overflow_slot(&app).await;
                let out = overflow::run_overflow(
                    &app.http,
                    &app.cfg.core_url,
                    &app.cfg.model_id,
                    &app.chunk_store,
                    &app.stats,
                    &middle,
                    &query,
                    route,
                    &params,
                )
                .await;
                return respond(out, stream_req, &app.cfg, &app.stats);
            }
        } else if app.cfg.overflow_prewarm
            && total as f32 > app.cfg.overflow_prewarm_ratio * app.cfg.n_ctx as f32
        {
            // Under the window: summarize ahead, in the background, so the first over-window turn
            // finds the work done. The reply does not wait for this.
            spawn_prewarm(&app, prewarm_middle(&messages, strategy, &app.cfg));
        }
    }

    // --- Under the window: pass through to the core. ---
    body["model"] = json!(app.cfg.model_id);
    // xzo is a buffered proxy, so the core call is never streamed; if the client asked to stream,
    // `respond` re-emits the buffered answer as server-sent events.
    body["stream"] = json!(false);
    // Strict chat templates: forward verbatim, and on a template rejection normalize the message
    // layout and retry, caching the winning strategy per model. Any other error body is returned
    // unchanged.
    let out = match normalize::post_normalized(
        &app.http, &app.cfg.core_url, &app.cfg.model_id, &body,
        app.cfg.normalize_system, &app.normalize_strategy_cache, &app.stats,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => return Json(json!({"error": {"message": e, "type": "upstream"}})).into_response(),
    };
    respond(out, stream_req, &app.cfg, &app.stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xzo::overflow::classify;

    // ---------------------------------------------------------------------------------------
    // Token counting is chosen deterministically at startup, or xzo refuses to start
    // ---------------------------------------------------------------------------------------

    /// A fake model server: always `/v1/models`; `/tokenize` only when asked for.
    async fn fake_core(with_tokenize: bool) -> String {
        let mut app = Router::new()
            .route("/v1/models", get(|| async { Json(json!({"object": "list", "data": []})) }));
        if with_tokenize {
            app = app.route("/tokenize", post(|| async { Json(json!({"tokens": [1, 2, 3]})) }));
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    fn cfg_for(core_url: &str) -> config::Config {
        let mut c = config::Config::default();
        c.core_url = core_url.to_string();
        c.overflow = true;
        c.exact_tokens = true;
        c.tokenizer = None;
        c
    }

    #[tokio::test]
    async fn a_core_with_tokenize_is_counted_by_the_core() {
        let url = fake_core(true).await;
        let c = choose_counter(&cfg_for(&url), &reqwest::Client::new()).await.unwrap();
        assert!(matches!(c, tokcount::Counter::Core));
    }

    /// Ollama, MLX, LM Studio: up, but no /tokenize. The estimate would under-count; refuse instead.
    #[tokio::test]
    async fn a_core_without_tokenize_refuses_to_start() {
        let url = fake_core(false).await;
        let e = choose_counter(&cfg_for(&url), &reqwest::Client::new()).await.err().expect("must refuse");
        assert!(e.contains("XZO_TOKENIZER"), "the error must say how to fix it: {e}");
    }

    #[tokio::test]
    async fn the_estimate_is_only_used_when_chosen() {
        let url = fake_core(false).await;
        let mut cfg = cfg_for(&url);
        cfg.exact_tokens = false;
        let c = choose_counter(&cfg, &reqwest::Client::new()).await.unwrap();
        assert!(matches!(c, tokcount::Counter::Estimate));
    }

    #[tokio::test]
    async fn a_tokenizer_that_will_not_load_is_fatal() {
        let url = fake_core(true).await;
        let mut cfg = cfg_for(&url);
        cfg.tokenizer = Some("does/not/exist/tokenizer.json".into());
        assert!(choose_counter(&cfg, &reqwest::Client::new()).await.is_err());
    }

    /// Reachability no longer depends on /tokenize: a server without it is still "up".
    #[tokio::test]
    async fn a_core_without_tokenize_is_still_reachable() {
        let url = fake_core(false).await;
        assert!(core_reachable(&reqwest::Client::new(), &url).await);
        assert!(!core_reachable(&reqwest::Client::new(), "http://127.0.0.1:9").await);
    }

    #[test]
    fn flatten_text_and_multimodal() {
        assert_eq!(flatten(&json!("hello")), "hello");
        let mm = json!([
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
        ]);
        assert_eq!(flatten(&mm), "describe this\n[image omitted: text-only core]");
        assert_eq!(flatten(&Value::Null), "");
    }

    #[test]
    fn last_user_takes_final_user_turn() {
        let msgs = vec![
            json!({"role": "system", "content": "be terse"}),
            json!({"role": "user", "content": "first"}),
            json!({"role": "assistant", "content": "ok"}),
            json!({"role": "user", "content": "second"}),
        ];
        assert_eq!(last_user(&msgs), "second");
        assert_eq!(last_user(&[]), "");
    }

    #[test]
    fn system_prefix_and_last_user_never_chunked() {
        let msgs = vec![
            json!({"role": "system", "content": "SYS"}),
            json!({"role": "user", "content": "earlier question"}),
            json!({"role": "assistant", "content": "an earlier answer"}),
            json!({"role": "user", "content": "the final question"}),
        ];
        // The EXTRACTION middle keeps the pre-2.3 semantics: system + last user excluded.
        let middle = extraction_middle(&msgs);
        assert!(classify(&msgs, 2000).total > 0);
        assert!(middle.iter().all(|m| m.as_str() != "SYS"));
        assert!(middle.iter().all(|m| m.as_str() != "the final question"));
        assert!(middle.iter().any(|m| m.as_str() == "earlier question"));
        assert!(middle.iter().any(|m| m.as_str() == "an earlier answer"));
    }

    #[test]
    fn under_window_skips_overflow() {
        // a tiny prompt is well under n_ctx * trigger -> the gate is false (pass-through path taken)
        let msgs = vec![json!({"role": "user", "content": "hi there"})];
        assert!(!overflow::should_overflow(classify(&msgs, 2000).total, 8192, 0.9));
    }

    #[test]
    fn classify_partitions_system_middle_tail() {
        let msgs = vec![
            json!({"role": "system", "content": "SYS"}),
            json!({"role": "user", "content": "old q"}),
            json!({"role": "assistant", "content": "old a"}),
            json!({"role": "user", "content": "the current question"}),
        ];
        let c = classify(&msgs, 2000); // generous budget
        assert_eq!(c.system.len(), 1);
        assert_eq!(c.system[0]["content"], json!("SYS"));
        // the current user turn is always the last element of the pinned tail
        assert_eq!(c.tail.last().unwrap()["content"], json!("the current question"));
        assert!(!c.has_tool_calls && !c.has_tool_messages);
    }

    #[test]
    fn classify_tiny_budget_still_keeps_current_user() {
        let msgs = vec![
            json!({"role": "user", "content": "first old turn with lots of words here"}),
            json!({"role": "assistant", "content": "a fairly long assistant reply as well"}),
            json!({"role": "user", "content": "current"}),
        ];
        let c = classify(&msgs, 1); // budget of ~1 token
        // the current turn is kept even though it exceeds the budget; older turns fall to the middle
        assert_eq!(c.tail.len(), 1);
        assert_eq!(c.tail[0]["content"], json!("current"));
        assert!(c.middle.iter().any(|m| m.contains("first old turn")));
    }

    #[test]
    fn classify_snaps_tool_group_into_tail() {
        // A large assistant tool_calls message + its tool result must be pinned whole; the tail
        // must never START with a dangling role:"tool" result.
        let msgs = vec![
            json!({"role": "user", "content": "older user turn padding padding padding padding"}),
            json!({"role": "assistant", "content": "assistant reasoning about the tool call goes here now",
                   "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "the tool result"}),
            json!({"role": "user", "content": "current"}),
        ];
        let c = classify(&msgs, 6); // budget lands the raw boundary on the tool result
        assert!(c.has_tool_calls && c.has_tool_messages);
        // snap pulled the assistant parent in, so the tail starts with the assistant, not the tool
        assert_eq!(c.tail[0].get("role").and_then(|r| r.as_str()), Some("assistant"));
        assert_eq!(c.tail.len(), 3); // assistant + tool + current
        assert!(c.middle.iter().any(|m| m.contains("older user turn")));
    }

    // ---------------------------------------------------------------------------------------
    // Bind exposure
    // ---------------------------------------------------------------------------------------

    #[test]
    fn loopback_addresses_are_recognised() {
        for h in ["127.0.0.1", "localhost", "LOCALHOST", "::1", "[::1]", "127.0.0.5", " 127.0.0.1 "]
        {
            assert!(is_loopback_host(h), "{h:?} should be loopback");
        }
    }

    /// The wildcards are the whole point: `0.0.0.0` is the ordinary way to make a container
    /// reachable, and it is exactly the change that publishes an unauthenticated service.
    #[test]
    fn wildcard_and_concrete_addresses_are_not_loopback() {
        for h in ["0.0.0.0", "::", "192.168.1.10", "10.0.0.4", "example.com", ""] {
            assert!(!is_loopback_host(h), "{h:?} must not count as loopback");
        }
    }

    /// Default config starts. This is the test that fails if someone "fixes" the guard by making it
    /// stricter than intended -- the common path must stay frictionless.
    #[test]
    fn the_default_host_starts_without_an_opt_in() {
        let cfg = crate::config::Config::default();
        assert!(check_bind_exposure(&cfg.host, cfg.allow_remote).is_ok());
        assert!(!cfg.allow_remote, "remote binding must not be opt-out");
    }

    /// Refused without the opt-in, allowed with it -- and the refusal has to SAY what is exposed,
    /// naming the destructive endpoint. An error that just says "refused" teaches the reader to set
    /// the flag without learning why.
    #[test]
    fn a_non_loopback_bind_is_refused_until_opted_in() {
        let err = check_bind_exposure("0.0.0.0", false).expect_err("must refuse");
        assert!(err.contains("/prune"));
        assert!(err.to_lowercase().contains("destructive"));
        assert!(err.contains("XZO_ALLOW_REMOTE=1"));
        assert!(check_bind_exposure("0.0.0.0", true).is_ok());
    }

    // ---------------------------------------------------------------------------------------
    // Overflow concurrency gate
    // ---------------------------------------------------------------------------------------

    /// The bound must actually bind: with N permits, only N holders can be inside at once, and the
    /// N+1th waits rather than being refused.
    #[tokio::test]
    async fn the_gate_admits_only_its_permit_count() {
        let gate = tokio::sync::Semaphore::new(2);
        let a = gate.acquire().await.expect("first");
        let b = gate.acquire().await.expect("second");
        assert_eq!(gate.available_permits(), 0);
        // A third would block, so assert non-blockingly that it cannot be had right now.
        assert!(gate.try_acquire().is_err(), "the gate admitted more than its permit count");
        drop(a);
        assert!(gate.try_acquire().is_ok(), "a released slot was not reusable");
        drop(b);
    }

    /// `0` means unlimited, and it is expressed as a permit count nothing exhausts rather than as a
    /// branch. This pins that the translation actually yields an unbounded-in-practice gate.
    #[test]
    fn zero_means_unlimited() {
        let permits = if 0 == 0 { tokio::sync::Semaphore::MAX_PERMITS } else { 0 };
        let gate = tokio::sync::Semaphore::new(permits);
        assert!(gate.available_permits() > 1_000_000);
    }

    /// The default queues rather than rejects, which is what makes shipping a non-zero default safe:
    /// no request that used to succeed can now fail, it can only wait.
    #[test]
    fn the_default_is_a_small_positive_bound() {
        let cfg = crate::config::Config::default();
        assert_eq!(cfg.overflow_max_inflight, 4);
    }

    // ---------------------------------------------------------------------------------------
    // Background pre-warm must summarize what compaction will actually look up
    // ---------------------------------------------------------------------------------------

    /// An agent session: system, a task, several tool rounds, then the question that overflows.
    fn agent_session(rounds: usize) -> Vec<Value> {
        let mut m = vec![
            json!({"role": "system", "content": "agent"}),
            json!({"role": "user", "content": "why is staging failing?"}),
        ];
        for i in 0..rounds {
            let id = format!("c{i}");
            m.push(json!({"role": "assistant", "content": null, "tool_calls": [{"id": id, "type": "function",
                "function": {"name": "read_file", "arguments": format!("{{\"path\":\"src/f{i}.rs\"}}")}}]}));
            m.push(json!({"role": "tool", "tool_call_id": id,
                "content": format!("fn f{i}() {{}}\n// a fair amount of file content number {i}\n")}));
        }
        m
    }

    fn hashes(texts: &[String]) -> std::collections::HashSet<String> {
        chunks::split_messages(texts, 1500).into_iter().map(|c| c.content_hash).collect()
    }

    /// THE property: everything the overflow turn will need was summarized on the turn before it.
    /// Uses a tail budget small enough that nearly everything is in the compressed middle.
    #[test]
    fn prewarm_covers_everything_compaction_will_need() {
        let cfg = config::Config::default();
        let before = agent_session(6); // the last turn under the window
        let mut overflow_turn = before.clone();
        overflow_turn.push(json!({"role": "user", "content": "what does f2 do?"}));

        let need = hashes(&overflow::classify_with(&overflow_turn, 1, true).middle);
        let have = hashes(&prewarm_middle(&before, router::ContextStrategy::Compaction, &cfg));
        assert!(!need.is_empty());
        let missing: Vec<_> = need.difference(&have).collect();
        assert!(missing.is_empty(), "the overflow turn would still summarize {} piece(s)", missing.len());
    }

    /// And AFTER the overflow: what one over-window turn pre-warms covers what the next turn will
    /// need, so the message that slides out of the tail is already summarized. Real tail budget,
    /// tool results big enough that only the last few stay verbatim.
    #[test]
    fn prewarm_after_overflow_covers_the_next_turn() {
        let cfg = config::Config::default();
        let mut turn = agent_session(12);
        for m in turn.iter_mut().filter(|m| m["role"] == "tool") {
            let body = m["content"].as_str().unwrap().repeat(60);
            m["content"] = json!(body);
        }
        let mut next = turn.clone();
        next.push(json!({"role": "assistant", "content": null, "tool_calls": [{"id": "c99", "type": "function",
            "function": {"name": "read_file", "arguments": "{\"path\":\"src/g.rs\"}"}}]}));
        next.push(json!({"role": "tool", "tool_call_id": "c99", "content": "fn g() {}\n".repeat(300)}));

        let this_middle = overflow::classify_with(&turn, cfg.compact_tail_tokens, true).middle;
        let need = hashes(&overflow::classify_with(&next, cfg.compact_tail_tokens, true).middle);
        let have = hashes(&prewarm_middle(&turn, router::ContextStrategy::Compaction, &cfg));
        assert!(need.len() > hashes(&this_middle).len(), "the next turn must push something out of the tail");
        let missing: Vec<_> = need.difference(&have).collect();
        assert!(missing.is_empty(), "the next turn would still summarize {} piece(s)", missing.len());
    }

    /// The bug this replaces, pinned: the old pre-warm used the document split, whose tool results
    /// carry no call label — so none of its tool-result summaries matched what compaction looks up.
    #[test]
    fn the_old_document_split_missed_every_tool_result() {
        let before = agent_session(6);
        let mut overflow_turn = before.clone();
        overflow_turn.push(json!({"role": "user", "content": "what does f2 do?"}));

        let need = hashes(&overflow::classify_with(&overflow_turn, 1, true).middle);
        let old = hashes(&extraction_middle(&before));
        let tool_results = overflow::classify_with(&overflow_turn, 1, true)
            .middle
            .iter()
            .filter(|t| t.starts_with("[read_file"))
            .count();
        assert_eq!(tool_results, 6);
        // It missed MORE than the tool results. The document split also drops the latest user
        // message, treating it as the question being asked — and in an agent session that is often
        // the original task, many tool rounds back, still sitting in the compressed history. This
        // test was first written to expect 1 of 7 covered; the real figure is 0.
        assert_eq!(
            need.intersection(&old).count(),
            0,
            "the old pre-warm was expected to cover none of what the overflow turn needs"
        );
    }

    /// Pre-warm includes the NEWEST message. `classify` keeps it in the tail because for an
    /// overflowing request it is the question; for pre-warm it has simply arrived.
    #[test]
    fn prewarm_includes_the_newest_message() {
        let cfg = config::Config::default();
        let s = agent_session(2);
        let texts = prewarm_middle(&s, router::ContextStrategy::Compaction, &cfg);
        assert!(texts.last().unwrap().contains("file content number 1"), "{texts:?}");
    }

    /// Document traffic is unchanged: it still pre-warms with the split its own path uses.
    #[test]
    fn document_traffic_prewarms_with_the_document_split() {
        let cfg = config::Config::default();
        let s = agent_session(3);
        assert_eq!(
            prewarm_middle(&s, router::ContextStrategy::Extraction, &cfg),
            extraction_middle(&s)
        );
    }

    #[test]
    fn prewarm_starts_at_a_quarter_of_the_window() {
        let cfg = config::Config::default();
        assert!(cfg.overflow_prewarm);
        assert_eq!(cfg.overflow_prewarm_ratio, 0.25);
    }
}
