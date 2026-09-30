// SPDX-License-Identifier: Apache-2.0
//! Message normalization strategies: pure helpers for adapting chat message structure
//! to different template requirements. The `chat` handler picks a strategy per
//! `Config.normalize` and uses these to adjust system/user/assistant message positioning.

use crate::stats::Stats;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;

/// How system messages are normalized within a message sequence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strategy {
    /// Messages passed through unchanged (the ladder's starting rung; also used for "auto").
    Verbatim,
    /// Non-leading system messages converted to user messages with `[system-note]` prefix.
    Convert,
    /// All system messages hoisted to position 0, joined by `"\n\n"`.
    Hoist,
    /// All system messages merged into the first user message.
    MergeUser0,
    /// Normalization disabled; return input unchanged.
    Off,
}

impl Strategy {
    /// Parse a strategy from an environment variable string (case-insensitive, trimmed).
    /// "auto" maps to Verbatim (the ladder's starting rung for retry chains).
    /// Unknown values default to Verbatim.
    pub fn from_env(s: &str) -> Strategy {
        match s.trim().to_lowercase().as_str() {
            "auto" => Strategy::Verbatim,
            "off" => Strategy::Off,
            "convert" => Strategy::Convert,
            "hoist" => Strategy::Hoist,
            "merge-user0" => Strategy::MergeUser0,
            _ => Strategy::Verbatim,
        }
    }

    /// Return the normalized name used for stats keys (e.g., `normalize.strategy_convert`).
    pub fn name(&self) -> &'static str {
        match self {
            Strategy::Verbatim => "verbatim",
            Strategy::Convert => "convert",
            Strategy::Hoist => "hoist",
            Strategy::MergeUser0 => "merge-user0",
            Strategy::Off => "off",
        }
    }
}

/// Read a message's text content. Always goes through `inject::flatten`, so the OpenAI
/// content-parts array form (`[{"type":"text","text":"…"}]`) is handled rather than silently
/// read as an empty string — the transforms below REWRITE content, so a bad read here destroys
/// the message instead of merely mis-measuring it.
fn text_of(m: &Value) -> String {
    crate::inject::flatten(m.get("content").unwrap_or(&Value::Null))
}

/// Transform: non-leading system messages become user messages.
/// For each message at index > 0 with role "system", change its role to "user"
/// and prepend "[system-note]\n" to the content. All other messages (index 0 system,
/// all non-system) pass through unchanged. Order and count are preserved.
pub fn convert(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .enumerate()
        .map(|(idx, m)| {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if idx > 0 && role == "system" {
                let content = text_of(m);
                let mut new_msg = m.clone();
                new_msg["role"] = json!("user");
                new_msg["content"] = json!(format!("[system-note]\n{}", content));
                new_msg
            } else {
                m.clone()
            }
        })
        .collect()
}

/// Transform: hoist all system messages to position 0.
/// Collect all system message contents in order, remove them from the stream,
/// and insert one merged system message at position 0 with content joined by "\n\n".
/// Non-system messages preserve their relative order.
/// If no system messages exist, return input unchanged.
pub fn hoist(messages: &[Value]) -> Vec<Value> {
    let mut system_contents = Vec::new();
    let mut non_system = Vec::new();

    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "system" {
            system_contents.push(text_of(m));
        } else {
            non_system.push(m.clone());
        }
    }

    if system_contents.is_empty() {
        return messages.to_vec();
    }

    let joined = system_contents.join("\n\n");
    let mut result = vec![json!({ "role": "system", "content": joined })];
    result.extend(non_system);
    result
}

/// Transform: merge all system messages into the first user message.
/// Collect all system message contents in order (joined by "\n\n"), remove them from the stream,
/// and prepend that text (followed by "\n\n") to the content of the first user message.
/// Edge case: if no user message exists, insert the merged content as a new user message at position 0.
/// If no system messages exist, return input unchanged.
pub fn merge_user0(messages: &[Value]) -> Vec<Value> {
    let mut system_contents = Vec::new();
    let mut non_system = Vec::new();

    for m in messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "system" {
            system_contents.push(text_of(m));
        } else {
            non_system.push(m.clone());
        }
    }

    if system_contents.is_empty() {
        return messages.to_vec();
    }

    let joined = system_contents.join("\n\n");
    let prefix = format!("{}\n\n", joined);

    // Find the first user message and merge into it.
    let mut found_user = false;
    let result: Vec<Value> = non_system
        .into_iter()
        .map(|m| {
            if !found_user {
                let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
                if role == "user" {
                    found_user = true;
                    let content = text_of(&m);
                    let mut new_msg = m.clone();
                    new_msg["content"] = json!(format!("{}{}", prefix, content));
                    return new_msg;
                }
            }
            m
        })
        .collect();

    if found_user {
        result
    } else {
        // No user message found; insert the system content as a new user message at position 0.
        let mut new_result = vec![json!({ "role": "user", "content": joined })];
        new_result.extend(result);
        new_result
    }
}

/// Detect if a 400 error from upstream is a chat-template / message-structure rejection
/// that we should retry-normalize, vs a genuine bad request.
/// Returns true only when status is 400 AND the lowercased body contains any of:
/// "system message must be at the beginning", "must alternate", "template", "raise_exception", "jinja".
pub fn is_template_400(status: u16, body: &str) -> bool {
    if status != 400 {
        return false;
    }
    let lower = body.to_lowercase();
    lower.contains("system message must be at the beginning")
        || lower.contains("must alternate")
        || lower.contains("template")
        || lower.contains("raise_exception")
        || lower.contains("jinja")
}

/// The deterministic transform ladder, tried in order on a template/structure 400.
const LADDER: [Strategy; 3] = [Strategy::Convert, Strategy::Hoist, Strategy::MergeUser0];

/// Apply a strategy to a message array (pure). Verbatim/Off return the input unchanged.
fn apply(strategy: Strategy, messages: &[Value]) -> Vec<Value> {
    match strategy {
        Strategy::Convert => convert(messages),
        Strategy::Hoist => hoist(messages),
        Strategy::MergeUser0 => merge_user0(messages),
        Strategy::Verbatim | Strategy::Off => messages.to_vec(),
    }
}

/// Clone the request body and swap in a new `messages` array (leaving model/tools/etc. intact).
fn with_messages(body: &Value, messages: Vec<Value>) -> Value {
    let mut wire = body.clone();
    wire["messages"] = json!(messages);
    wire
}

/// One POST to the core's chat-completions endpoint. Returns `(status, raw_body_text)`.
/// Unlike the pre-patch call sites, this inspects the HTTP status so a template-400 can be
/// distinguished from a success (today's code parsed the 400 error body as if it were an answer).
async fn post_once(http: &reqwest::Client, core_url: &str, body: &Value) -> Result<(u16, String), String> {
    let url = format!("{core_url}/v1/chat/completions");
    // Compaction and passthrough both land here, so this is the second (and last) seam every
    // outbound call crosses. Tracing here also shows each rung of the normalize
    // ladder separately, which is the only way to see WHICH rung a strict core accepted.
    crate::trace::request("core", body);
    let resp = http.post(&url).json(body).send().await.map_err(|e| format!("core unreachable: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp.text().await.map_err(|e| format!("bad core response: {e}"))?;
    if crate::trace::enabled() {
        match serde_json::from_str::<Value>(&text) {
            Ok(v) => crate::trace::response("core", &v),
            Err(_) => eprintln!("  <- core [core] status {status}, unparseable body"),
        }
    }
    Ok((status, text))
}

fn parse(text: &str) -> Result<Value, String> {
    serde_json::from_str(text).map_err(|e| format!("bad core response: {e}"))
}

/// POST the request to the core, normalizing the message structure reactively so strict chat
/// templates (e.g. Qwen3.5's "system must be first") accept a conversation that a lenient client
/// forwarded with a non-leading system message. Family-agnostic: the core drives escalation.
///
/// - `forced == Off`: exact passthrough (today's behavior) — no ladder.
/// - `forced == Convert|Hoist|Merge`: apply that one strategy proactively, skip the ladder.
/// - `forced == Verbatim` (the `auto` default): try the per-model cached strategy (verbatim on a
///   cold cache); on a template-400 walk the ladder, retrying after each rung until one is
///   accepted, then cache the winner per model so later turns apply it proactively.
///
/// Returns the parsed core JSON. A genuine (non-template) error body is returned **as-is** — never
/// worse than today. `Err` is only a transport/parse failure. Operates on a clone of `body`, so the
/// caller's canonical message array is never mutated (important for the recall loop, which appends
/// tool results across iterations).
pub async fn post_normalized(
    http: &reqwest::Client,
    core_url: &str,
    model: &str,
    body: &Value,
    forced: Strategy,
    cache: &Mutex<HashMap<String, Strategy>>,
    stats: &Stats,
) -> Result<Value, String> {
    let messages: Vec<Value> = body.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();

    // Off: byte-for-byte passthrough (the A/B "off" arm).
    if forced == Strategy::Off {
        let (_status, text) = post_once(http, core_url, body).await?;
        return parse(&text);
    }

    // Forced single strategy: apply it proactively and skip the ladder.
    if forced != Strategy::Verbatim {
        let wire = with_messages(body, apply(forced, &messages));
        let (_status, text) = post_once(http, core_url, &wire).await?;
        stats.incr("normalize.applied");
        stats.incr(&format!("normalize.strategy_{}", forced.name()));
        return parse(&text);
    }

    // Auto: reactive ladder with a per-model strategy cache.
    let cached = cache.lock().unwrap().get(model).copied().unwrap_or(Strategy::Verbatim);
    let wire = with_messages(body, apply(cached, &messages));
    let (status, text) = post_once(http, core_url, &wire).await?;
    if !is_template_400(status, &text) {
        // Success, or a non-template error we must return unchanged. Count a proactive fix only if
        // the cache actually applied a transform (the lenient-core common case applies nothing).
        if cached != Strategy::Verbatim {
            stats.incr("normalize.applied");
            stats.incr(&format!("normalize.strategy_{}", cached.name()));
        }
        return parse(&text);
    }
    // Keep the original 400 to return unchanged if the whole ladder fails to satisfy the core.
    let original = text;

    for rung in LADDER {
        if rung == cached {
            continue; // already tried proactively above
        }
        let wire = with_messages(body, apply(rung, &messages));
        let (st, t) = post_once(http, core_url, &wire).await?;
        if !is_template_400(st, &t) {
            cache.lock().unwrap().insert(model.to_string(), rung);
            stats.incr("normalize.applied");
            stats.incr(&format!("normalize.strategy_{}", rung.name()));
            return parse(&t);
        }
    }

    // Ladder exhausted: an exotic template none of the three satisfy. The optional LLM
    // structural-patch fallback (XZO_NORMALIZE_LLM_FALLBACK, deferred) would hook in here. For now,
    // return the core's original 400 unchanged — never worse than today.
    stats.incr("normalize.ladder_exhausted");
    parse(&original)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OpenAI content can be an array of typed parts, and many clients send it that
    /// way. The transforms REWRITE content, so reading it with a bare `as_str()` did not merely
    /// mis-measure the message — it replaced the text with an empty string and forwarded the
    /// request anyway. Each transform loses something different, so each gets its own case.
    #[test]
    fn convert_keeps_array_content_of_a_system_note() {
        let msgs = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "system", "content": [{"type": "text", "text": "be terse"}]}),
        ];
        let out = convert(&msgs);
        let c = out[1]["content"].as_str().unwrap();
        assert!(c.contains("be terse"), "convert dropped array content: {c:?}");
        assert_eq!(out[1]["role"], "user");
    }

    #[test]
    fn hoist_keeps_array_content_of_the_system_prompt() {
        let msgs = vec![
            json!({"role": "system", "content": [{"type": "text", "text": "you are a linter"}]}),
            json!({"role": "user", "content": "check this"}),
        ];
        let out = hoist(&msgs);
        let sys = out[0]["content"].as_str().unwrap();
        assert!(sys.contains("you are a linter"), "hoist emptied the system prompt: {sys:?}");
    }

    #[test]
    fn merge_user0_keeps_the_users_actual_question() {
        // The worst case: merge_user0 rewrites the first USER message too, so a bad read here
        // deleted the question and left only the system prefix.
        let msgs = vec![
            json!({"role": "system", "content": [{"type": "text", "text": "be terse"}]}),
            json!({"role": "user", "content": [{"type": "text", "text": "what is the port?"}]}),
        ];
        let out = merge_user0(&msgs);
        let u = out[0]["content"].as_str().unwrap();
        assert!(u.contains("be terse"), "merge_user0 dropped the system content: {u:?}");
        assert!(u.contains("what is the port?"), "merge_user0 deleted the question: {u:?}");
    }

    #[test]
    fn test_strategy_from_env_auto() {
        assert_eq!(Strategy::from_env("auto"), Strategy::Verbatim);
        assert_eq!(Strategy::from_env("AUTO"), Strategy::Verbatim);
        assert_eq!(Strategy::from_env("  auto  "), Strategy::Verbatim);
    }

    #[test]
    fn test_strategy_from_env_off() {
        assert_eq!(Strategy::from_env("off"), Strategy::Off);
        assert_eq!(Strategy::from_env("OFF"), Strategy::Off);
    }

    #[test]
    fn test_strategy_from_env_convert() {
        assert_eq!(Strategy::from_env("convert"), Strategy::Convert);
        assert_eq!(Strategy::from_env("CONVERT"), Strategy::Convert);
    }

    #[test]
    fn test_strategy_from_env_hoist() {
        assert_eq!(Strategy::from_env("hoist"), Strategy::Hoist);
        assert_eq!(Strategy::from_env("HOIST"), Strategy::Hoist);
    }

    #[test]
    fn test_strategy_from_env_merge_user0() {
        assert_eq!(Strategy::from_env("merge-user0"), Strategy::MergeUser0);
        assert_eq!(Strategy::from_env("MERGE-USER0"), Strategy::MergeUser0);
    }

    #[test]
    fn test_strategy_from_env_unknown_defaults_to_verbatim() {
        assert_eq!(Strategy::from_env("unknown"), Strategy::Verbatim);
        assert_eq!(Strategy::from_env(""), Strategy::Verbatim);
        assert_eq!(Strategy::from_env("   "), Strategy::Verbatim);
    }

    #[test]
    fn test_strategy_name_round_trips() {
        let strategies = vec![
            Strategy::Verbatim,
            Strategy::Convert,
            Strategy::Hoist,
            Strategy::MergeUser0,
            Strategy::Off,
        ];
        for strategy in strategies {
            let name = strategy.name();
            // "verbatim" isn't an explicit from_env arm; it falls through to the Verbatim
            // default, so the round-trip still holds for every variant.
            let parsed = Strategy::from_env(name);
            assert_eq!(parsed, strategy, "name {} should round-trip", name);
        }
    }

    #[test]
    fn test_convert_non_leading_system_becomes_user() {
        let messages = vec![
            json!({"role": "system", "content": "You are helpful"}),
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "system", "content": "Be concise"}),
            json!({"role": "user", "content": "Goodbye"}),
        ];
        let result = convert(&messages);

        assert_eq!(result.len(), 4, "length preserved");
        assert_eq!(result[0]["role"], "system", "index 0 system untouched");
        assert_eq!(result[0]["content"], "You are helpful", "index 0 content unchanged");
        assert_eq!(result[1]["role"], "user", "index 1 unchanged");
        assert_eq!(result[1]["content"], "Hello", "index 1 content unchanged");
        assert_eq!(result[2]["role"], "user", "index 2 system converted to user");
        assert_eq!(result[2]["content"], "[system-note]\nBe concise", "index 2 content prefixed");
        assert_eq!(result[3]["role"], "user", "index 3 unchanged");
        assert_eq!(result[3]["content"], "Goodbye", "index 3 content unchanged");
    }

    #[test]
    fn test_convert_no_leading_system() {
        let messages = vec![
            json!({"role": "user", "content": "First"}),
            json!({"role": "system", "content": "Middle system"}),
            json!({"role": "user", "content": "Last"}),
        ];
        let result = convert(&messages);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[1]["role"], "user", "middle system converted");
        assert_eq!(result[1]["content"], "[system-note]\nMiddle system");
        assert_eq!(result[2]["role"], "user");
    }

    #[test]
    fn test_hoist_all_system_messages() {
        let messages = vec![
            json!({"role": "system", "content": "A"}),
            json!({"role": "user", "content": "Q1"}),
            json!({"role": "system", "content": "B"}),
            json!({"role": "user", "content": "Q2"}),
        ];
        let result = hoist(&messages);

        assert_eq!(result.len(), 3, "3 non-system messages + 1 merged system");
        assert_eq!(result[0]["role"], "system", "merged system at position 0");
        assert_eq!(result[0]["content"], "A\n\nB", "system contents joined by newlines");
        assert_eq!(result[1]["role"], "user");
        assert_eq!(result[1]["content"], "Q1", "relative order of non-system preserved");
        assert_eq!(result[2]["role"], "user");
        assert_eq!(result[2]["content"], "Q2");
    }

    #[test]
    fn test_hoist_no_system_messages() {
        let messages = vec![
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi"}),
        ];
        let result = hoist(&messages);

        assert_eq!(result.len(), 2);
        assert_eq!(result, messages, "unchanged when no system messages");
    }

    #[test]
    fn test_merge_user0_basic() {
        let messages = vec![
            json!({"role": "system", "content": "S"}),
            json!({"role": "user", "content": "Q"}),
            json!({"role": "assistant", "content": "A"}),
        ];
        let result = merge_user0(&messages);

        assert_eq!(result.len(), 2, "system removed, user updated, assistant unchanged");
        assert_eq!(
            result[0]["role"], "user",
            "first message is the merged user"
        );
        assert_eq!(
            result[0]["content"], "S\n\nQ",
            "user content prefixed with system content"
        );
        assert_eq!(result[1]["role"], "assistant");
        assert_eq!(result[1]["content"], "A");
    }

    #[test]
    fn test_merge_user0_no_user_edge_case() {
        let messages = vec![
            json!({"role": "system", "content": "Only system"}),
            json!({"role": "assistant", "content": "Response"}),
        ];
        let result = merge_user0(&messages);

        assert_eq!(result.len(), 2, "system becomes user, assistant unchanged");
        assert_eq!(result[0]["role"], "user", "system content becomes a user message");
        assert_eq!(result[0]["content"], "Only system");
        assert_eq!(result[1]["role"], "assistant");
    }

    #[test]
    fn test_merge_user0_no_system_messages() {
        let messages = vec![
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi"}),
        ];
        let result = merge_user0(&messages);

        assert_eq!(result.len(), 2);
        assert_eq!(result, messages, "unchanged when no system messages");
    }

    #[test]
    fn test_is_template_400_true_for_system_at_beginning() {
        assert!(is_template_400(400, "System message must be at the beginning"));
        assert!(is_template_400(400, "system message must be at the beginning"));
        assert!(is_template_400(400, "Error: System message must be at the beginning."));
    }

    #[test]
    fn test_is_template_400_true_for_must_alternate() {
        assert!(is_template_400(400, "messages must alternate between roles"));
    }

    #[test]
    fn test_is_template_400_true_for_template() {
        assert!(is_template_400(400, "Invalid template format"));
        assert!(is_template_400(400, "Template error in chat"));
    }

    #[test]
    fn test_is_template_400_true_for_raise_exception() {
        assert!(is_template_400(400, "raise_exception called"));
    }

    #[test]
    fn test_is_template_400_true_for_jinja() {
        assert!(is_template_400(400, "Jinja template error"));
    }

    #[test]
    fn test_is_template_400_false_for_wrong_status() {
        assert!(!is_template_400(200, "system message must be at the beginning"));
        assert!(!is_template_400(500, "template error"));
        assert!(!is_template_400(401, "template error"));
    }

    #[test]
    fn test_is_template_400_false_for_unrelated_400() {
        assert!(!is_template_400(400, "invalid 'max_tokens'"));
        assert!(!is_template_400(400, "unauthorized"));
        assert!(!is_template_400(400, "rate limited"));
    }

    // ---- Ladder driver: mock-core integration ----
    // A tiny in-process axum core that enforces a chosen template rule, so we can assert the ladder
    // picks the working rung, caches it per model, and applies it proactively next turn — without a
    // real GPU core.

    use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Clone, Copy)]
    enum Rule {
        /// Strict single-leading-system (Qwen3.5-style): reject any non-leading system message.
        SystemFirst,
        /// Mistral-style: single leading system AND no two consecutive same-role messages.
        AlternationStrict,
        /// A genuine bad request (non-template 400) — must be returned as-is, no ladder.
        GenuineBadRequest,
    }

    #[derive(Clone)]
    struct MockState {
        rule: Rule,
        hits: Arc<AtomicUsize>,
    }

    async fn mock_handler(State(st): State<MockState>, Json(body): Json<Value>) -> axum::response::Response {
        st.hits.fetch_add(1, Ordering::SeqCst);
        let msgs = body.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
        let roles: Vec<String> = msgs
            .iter()
            .map(|m| m.get("role").and_then(|r| r.as_str()).unwrap_or("").to_string())
            .collect();
        let no_mid_system = roles.iter().enumerate().all(|(i, r)| !(r == "system" && i > 0));
        let alternates = roles.windows(2).all(|w| w[0] != w[1]);
        let (ok, err) = match st.rule {
            Rule::SystemFirst => (no_mid_system, "Jinja Exception: System message must be at the beginning."),
            Rule::AlternationStrict => (
                no_mid_system && alternates,
                "Conversation roles must alternate user/assistant/user/assistant/...",
            ),
            Rule::GenuineBadRequest => (false, "invalid value for 'max_tokens'"),
        };
        if ok {
            Json(json!({"choices":[{"message":{"role":"assistant","content":"ok"}}]})).into_response()
        } else {
            (StatusCode::BAD_REQUEST, Json(json!({"error":{"message": err}}))).into_response()
        }
    }

    async fn spawn_mock(rule: Rule) -> (String, Arc<AtomicUsize>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let state = MockState { rule, hits: hits.clone() };
        let app = Router::new().route("/v1/chat/completions", post(mock_handler)).with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        (format!("http://{addr}"), hits)
    }

    fn reminder_convo() -> Value {
        // A non-leading system (the injected <system-reminder>) mid-conversation.
        json!({"model":"m","messages":[
            {"role":"system","content":"You are a coding agent."},
            {"role":"user","content":"Review repo"},
            {"role":"assistant","content":"", "tool_calls":[]},
            {"role":"tool","content":"files: ..."},
            {"role":"system","content":"<system-reminder>cwd is /work"},
            {"role":"user","content":"continue"},
        ]})
    }

    #[tokio::test]
    async fn ladder_selects_convert_for_system_first_and_caches() {
        let (core_url, hits) = spawn_mock(Rule::SystemFirst).await;
        let http = reqwest::Client::new();
        let cache = Mutex::new(HashMap::new());
        let stats = Stats::new();
        let body = reminder_convo();

        // Turn 1: verbatim -> 400, ladder rung 1 (convert) accepted.
        let out = post_normalized(&http, &core_url, "qwen", &body, Strategy::Verbatim, &cache, &stats)
            .await
            .unwrap();
        assert_eq!(out.pointer("/choices/0/message/content").unwrap(), "ok");
        assert_eq!(cache.lock().unwrap().get("qwen").copied(), Some(Strategy::Convert));
        let after1 = hits.load(Ordering::SeqCst);
        assert_eq!(after1, 2, "verbatim(400) + convert(200)");

        // Turn 2: cached convert applied proactively -> a single accepted POST.
        let out2 = post_normalized(&http, &core_url, "qwen", &body, Strategy::Verbatim, &cache, &stats)
            .await
            .unwrap();
        assert_eq!(out2.pointer("/choices/0/message/content").unwrap(), "ok");
        assert_eq!(hits.load(Ordering::SeqCst) - after1, 1, "proactive: one POST, no failed round-trip");
    }

    #[tokio::test]
    async fn ladder_escalates_to_hoist_for_alternation_strict() {
        let (core_url, _hits) = spawn_mock(Rule::AlternationStrict).await;
        let http = reqwest::Client::new();
        let cache = Mutex::new(HashMap::new());
        let stats = Stats::new();
        let body = reminder_convo();

        // verbatim(400, mid-system) -> convert(400, creates consecutive users) -> hoist(200).
        let out = post_normalized(&http, &core_url, "mistral", &body, Strategy::Verbatim, &cache, &stats)
            .await
            .unwrap();
        assert_eq!(out.pointer("/choices/0/message/content").unwrap(), "ok");
        assert_eq!(cache.lock().unwrap().get("mistral").copied(), Some(Strategy::Hoist));
    }

    #[tokio::test]
    async fn genuine_400_returned_as_is_without_ladder() {
        let (core_url, hits) = spawn_mock(Rule::GenuineBadRequest).await;
        let http = reqwest::Client::new();
        let cache = Mutex::new(HashMap::new());
        let stats = Stats::new();
        let body = reminder_convo();

        let out = post_normalized(&http, &core_url, "qwen", &body, Strategy::Verbatim, &cache, &stats)
            .await
            .unwrap();
        // The core's genuine 400 body is returned unchanged; no ladder retries, no cache entry.
        assert!(out.pointer("/error/message").unwrap().as_str().unwrap().contains("max_tokens"));
        assert_eq!(hits.load(Ordering::SeqCst), 1, "no ladder retries on a non-template 400");
        assert!(cache.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn off_strategy_is_exact_passthrough() {
        let (core_url, hits) = spawn_mock(Rule::SystemFirst).await;
        let http = reqwest::Client::new();
        let cache = Mutex::new(HashMap::new());
        let stats = Stats::new();
        let body = reminder_convo();

        // Off: never normalize; the mid-system 400 comes straight back (today's behavior).
        let out = post_normalized(&http, &core_url, "qwen", &body, Strategy::Off, &cache, &stats)
            .await
            .unwrap();
        assert!(out.pointer("/error/message").is_some());
        assert_eq!(hits.load(Ordering::SeqCst), 1, "one POST, no ladder");
        assert!(cache.lock().unwrap().is_empty());
    }
}
