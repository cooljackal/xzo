// SPDX-License-Identifier: Apache-2.0
//! Show the prompt xzo actually sends the core (`XZO_TRACE_CORE`).
//!
//! WHY. xzo rewrites the conversation — chunking the middle away, splicing a recap in,
//! injecting retrieved snippets — and then never shows the result. From the client you see your own
//! messages; from the core's logs you see a prompt with no explanation of where it came from. The
//! interesting part happens in between and was invisible from both ends.
//!
//! Walkthrough 1 needed a Python proxy to see it. That was a gap in this binary, not a reason to
//! ship a proxy, and this closes it: `XZO_TRACE_CORE=summary` is enough to follow every decision the
//! memory layer makes.
//!
//! Diagnostics read the environment through a `OnceLock` rather than the config struct on purpose —
//! every outbound call funnels through two functions that have no access to `App`, and threading a
//! flag through them would be a lot of plumbing for a debug switch.

use crate::inject::est_tokens;
use serde_json::Value;
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    /// No output. The default, and free — the check is one atomic load.
    Off,
    /// One line per message: role, size, and a preview. Enough to see what the model was given.
    Summary,
    /// The complete request and response bodies, pretty-printed.
    Full,
}

impl Level {
    pub fn parse(s: &str) -> Level {
        match s.trim().to_lowercase().as_str() {
            "summary" | "on" | "1" | "true" | "yes" => Level::Summary,
            "full" | "verbose" => Level::Full,
            _ => Level::Off,
        }
    }
}

fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| Level::parse(&std::env::var("XZO_TRACE_CORE").unwrap_or_default()))
}

pub fn enabled() -> bool {
    level() != Level::Off
}

/// One-line preview with newlines made visible, so a multi-line system prompt stays one row.
fn preview(text: &str, max: usize) -> String {
    let flat = text.replace('\n', "\\n");
    let n = flat.chars().count();
    if n <= max {
        flat
    } else {
        let head: String = flat.chars().take(max).collect();
        format!("{head} …(+{} chars)", n - max)
    }
}

/// Flatten string-or-parts content for display only. Never used to build a prompt.
fn content_text(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| {
                p.get("text").and_then(|t| t.as_str()).unwrap_or("<non-text part>").to_string()
            })
            .collect::<Vec<_>>()
            .join(" | "),
        _ => String::new(),
    }
}

/// Print the outbound request. `what` names the caller ("overflow", "compaction", "passthrough")
/// so interleaved calls from the tool loops stay attributable.
pub fn request(what: &str, body: &Value) {
    let lvl = level();
    if lvl == Level::Off {
        return;
    }
    let msgs = body.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
    let tools = body.get("tools").and_then(|t| t.as_array()).cloned().unwrap_or_default();
    let total: usize = msgs.iter().map(|m| est_tokens(&content_text(m))).sum();

    eprintln!("\n{}", "=".repeat(96));
    eprint!("xzo -> core [{what}]  |  {} messages, ~{total} est tokens", msgs.len());
    if !tools.is_empty() {
        eprint!(", {} tools offered", tools.len());
    }
    eprintln!("\n{}", "=".repeat(96));

    for (i, m) in msgs.iter().enumerate() {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("?");
        let mut line = format!("[{i}] {role:<9}");
        if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
            let names: Vec<&str> = tcs
                .iter()
                .filter_map(|tc| tc.pointer("/function/name").and_then(|n| n.as_str()))
                .collect();
            line.push_str(&format!("TOOL_CALLS {names:?} "));
        }
        let body_text = content_text(m);
        let cap = if lvl == Level::Full { usize::MAX } else { 220 };
        eprintln!("{line}{}", preview(&body_text, cap));
    }
    if !tools.is_empty() {
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(|n| n.as_str()))
            .collect();
        eprintln!("     tools: {names:?}");
    }
    if lvl == Level::Full {
        eprintln!("--- full request body ---");
        eprintln!("{}", serde_json::to_string_pretty(body).unwrap_or_default());
    }
}

/// Print what came back: a tool call, or the answer.
pub fn response(what: &str, out: &Value) {
    let lvl = level();
    if lvl == Level::Off {
        return;
    }
    if let Some(err) = out.get("error") {
        eprintln!("  <- core [{what}] ERROR {}", preview(&err.to_string(), 300));
        return;
    }
    let msg = out.pointer("/choices/0/message").cloned().unwrap_or(Value::Null);
    if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        let calls: Vec<String> = tcs
            .iter()
            .map(|tc| {
                format!(
                    "{}({})",
                    tc.pointer("/function/name").and_then(|n| n.as_str()).unwrap_or("?"),
                    tc.pointer("/function/arguments").and_then(|a| a.as_str()).unwrap_or("")
                )
            })
            .collect();
        eprintln!("  <- core [{what}] TOOL CALL: {calls:?}");
    } else {
        let text = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let cap = if lvl == Level::Full { usize::MAX } else { 300 };
        eprintln!("  <- core [{what}] {}", preview(text, cap));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_levels_and_defaults_off() {
        assert_eq!(Level::parse("summary"), Level::Summary);
        assert_eq!(Level::parse("ON"), Level::Summary);
        assert_eq!(Level::parse("full"), Level::Full);
        assert_eq!(Level::parse(""), Level::Off);
        assert_eq!(Level::parse("off"), Level::Off);
        assert_eq!(Level::parse("nonsense"), Level::Off);
    }

    #[test]
    fn preview_marks_truncation_and_flattens_newlines() {
        assert_eq!(preview("a\nb", 10), "a\\nb");
        let long = "x".repeat(50);
        let p = preview(&long, 10);
        assert!(p.starts_with("xxxxxxxxxx "), "{p}");
        assert!(p.contains("+40 chars"), "{p}");
    }

    #[test]
    fn content_text_handles_both_shapes() {
        use serde_json::json;
        assert_eq!(content_text(&json!({"content": "plain"})), "plain");
        let parts = json!({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]});
        assert_eq!(content_text(&parts), "a | b");
        assert_eq!(content_text(&json!({})), "");
    }
}
