// SPDX-License-Identifier: Apache-2.0
//! Tool-call text detection and extraction. The frozen core does not always return a
//! tool call in the structured OpenAI `message.tool_calls` field — Qwen-family models frequently emit
//! the call as TEXT in `content`, either wrapped in `<tool_call>…</tool_call>` tags (the Qwen/Hermes
//! convention) or as a bare/fenced JSON object. When that happens and llama.cpp's `--jinja` parser
//! didn't lift it, xzo's structured-only read (`recall_loop`) treats the message as a final answer and
//! the intended call is silently DROPPED.
//!
//! This module recognizes those text-shaped calls. Phase 1 (the diagnostic) uses it to COUNT how
//! often a dropped message actually contained a tool call — splitting the "misses" into "the model
//! never called" vs "the model called but we dropped it". Phase 2 (the fix) will feed the extracted
//! calls back into the loop so they execute. Pure + deterministic: no network, unit-testable without
//! the core.

use serde_json::Value;

/// A tool call recovered from message text.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectedCall {
    pub name: String,
    /// Parsed arguments object (from `"arguments"` or `"parameters"`, whichever the model used).
    pub arguments: Value,
}

/// Extract every text-shaped tool call in `content` whose name is in `known`. Tries, in order of
/// precision: `<tool_call>` tags (highest), then fenced ```json blocks, then a bare top-level JSON
/// object — stopping at the first tier that yields a call, so a tagged call isn't also double-counted
/// as bare JSON. Returns empty when the content has no recognizable call (the "genuine non-call"
/// signal for the diagnostic).
pub fn detect_text_tool_calls(content: &str, known: &[&str]) -> Vec<DetectedCall> {
    // Tier 1: <tool_call> ... </tool_call> (case-insensitive tag match).
    let tagged: Vec<DetectedCall> = extract_between_ci(content, "<tool_call>", "</tool_call>")
        .into_iter()
        .filter_map(|blk| parse_call_json(blk.trim(), known))
        .collect();
    if !tagged.is_empty() {
        return tagged;
    }

    // Tier 2: fenced ```json ... ``` blocks.
    let fenced: Vec<DetectedCall> =
        extract_fenced(content).into_iter().filter_map(|blk| parse_call_json(blk.trim(), known)).collect();
    if !fenced.is_empty() {
        return fenced;
    }

    // Tier 3: a bare balanced-brace JSON object somewhere in the text.
    scan_bare_object(content, known).into_iter().collect()
}

/// Parse one JSON object string into a `DetectedCall` if it names a known tool. Accepts `"arguments"`
/// or `"parameters"`, each as either an object or a JSON-encoded string (some models double-encode).
fn parse_call_json(s: &str, known: &[&str]) -> Option<DetectedCall> {
    let v: Value = serde_json::from_str(s).ok()?;
    // A call may be nested under {"function": {...}} or flat {"name":..,"arguments":..}.
    let obj = v.get("function").unwrap_or(&v);
    let name = obj.get("name").and_then(|n| n.as_str())?.to_string();
    if !known.iter().any(|k| k.eq_ignore_ascii_case(&name)) {
        return None;
    }
    let raw_args = obj.get("arguments").or_else(|| obj.get("parameters")).cloned().unwrap_or(Value::Null);
    let arguments = match raw_args {
        // Double-encoded: "arguments": "{\"index\":1}"
        Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::String(s)),
        other => other,
    };
    Some(DetectedCall { name, arguments })
}

/// All substrings between every case-insensitive `open`/`close` pair.
fn extract_between_ci(hay: &str, open: &str, close: &str) -> Vec<String> {
    let lower = hay.to_lowercase();
    let (lo, lc) = (open.to_lowercase(), close.to_lowercase());
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(a) = lower[from..].find(&lo) {
        let start = from + a + open.len();
        if let Some(b) = lower[start..].find(&lc) {
            out.push(hay[start..start + b].to_string());
            from = start + b + close.len();
        } else {
            break;
        }
    }
    out
}

/// Bodies of ```json … ``` (or plain ``` … ```) fenced blocks.
fn extract_fenced(hay: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = hay;
    while let Some(open) = rest.find("```") {
        let after = &rest[open + 3..];
        // Skip an optional language tag on the fence line.
        let body_start = after.find('\n').map(|n| n + 1).unwrap_or(0);
        let body_area = &after[body_start..];
        if let Some(close) = body_area.find("```") {
            out.push(body_area[..close].to_string());
            rest = &body_area[close + 3..];
        } else {
            break;
        }
    }
    out
}

/// Find the first balanced-brace JSON object in `hay` that parses into a known-tool call. Scans each
/// `{` and grows to its matching `}` (string-aware) — cheap and good enough for a single call in text.
fn scan_bare_object(hay: &str, known: &[&str]) -> Option<DetectedCall> {
    let bytes = hay.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b != b'{' {
            continue;
        }
        let mut depth = 0i32;
        let mut in_str = false;
        let mut esc = false;
        for (j, &c) in bytes[i..].iter().enumerate() {
            if in_str {
                if esc {
                    esc = false;
                } else if c == b'\\' {
                    esc = true;
                } else if c == b'"' {
                    in_str = false;
                }
                continue;
            }
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        let slice = &hay[i..i + j + 1];
                        if let Some(call) = parse_call_json(slice, known) {
                            return Some(call);
                        }
                        break; // this object didn't match; move to the next `{`
                    }
                }
                _ => {}
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const KNOWN: &[&str] = &["recall_memory", "get_chunk", "search_chunks"];

    #[test]
    fn qwen_tool_call_tag() {
        let c = "Let me look that up.\n<tool_call>\n{\"name\": \"recall_memory\", \"arguments\": {\"index\": 2}}\n</tool_call>";
        let got = detect_text_tool_calls(c, KNOWN);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].name, "recall_memory");
        assert_eq!(got[0].arguments["index"], 2);
    }

    #[test]
    fn parameters_key_and_double_encoded_args() {
        // Some models use "parameters"; some double-encode arguments as a string.
        let a = "<tool_call>{\"name\":\"get_chunk\",\"parameters\":{\"id\":\"7\"}}</tool_call>";
        assert_eq!(detect_text_tool_calls(a, KNOWN)[0].arguments["id"], "7");
        let b = "<tool_call>{\"name\":\"get_chunk\",\"arguments\":\"{\\\"id\\\":\\\"9\\\"}\"}</tool_call>";
        assert_eq!(detect_text_tool_calls(b, KNOWN)[0].arguments["id"], "9");
    }

    #[test]
    fn fenced_json_block() {
        let c = "Here is the call:\n```json\n{\"name\": \"search_chunks\", \"arguments\": {\"query\": \"port\"}}\n```";
        let got = detect_text_tool_calls(c, KNOWN);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].name, "search_chunks");
        assert_eq!(got[0].arguments["query"], "port");
    }

    #[test]
    fn bare_json_object() {
        let c = "{\"name\": \"recall_memory\", \"arguments\": {\"index\": 1}}";
        assert_eq!(detect_text_tool_calls(c, KNOWN)[0].name, "recall_memory");
    }

    #[test]
    fn function_wrapper() {
        let c = "<tool_call>{\"function\":{\"name\":\"recall_memory\",\"arguments\":{\"index\":3}}}</tool_call>";
        assert_eq!(detect_text_tool_calls(c, KNOWN)[0].arguments["index"], 3);
    }

    #[test]
    fn plain_prose_is_not_a_call() {
        // A message that merely MENTIONS a tool name must not be counted as a call.
        let c = "I could use recall_memory here, but the answer is simply 42.";
        assert!(detect_text_tool_calls(c, KNOWN).is_empty());
    }

    #[test]
    fn unknown_tool_name_ignored() {
        let c = "<tool_call>{\"name\":\"delete_everything\",\"arguments\":{}}</tool_call>";
        assert!(detect_text_tool_calls(c, KNOWN).is_empty());
    }

    #[test]
    fn no_double_count_tag_and_json() {
        // A tagged call must not ALSO be counted as a bare object.
        let c = "<tool_call>{\"name\":\"recall_memory\",\"arguments\":{\"index\":1}}</tool_call>";
        assert_eq!(detect_text_tool_calls(c, KNOWN).len(), 1);
    }

    #[test]
    fn two_tagged_calls() {
        let c = "<tool_call>{\"name\":\"recall_memory\",\"arguments\":{\"index\":1}}</tool_call>\
                 <tool_call>{\"name\":\"recall_memory\",\"arguments\":{\"index\":2}}</tool_call>";
        assert_eq!(detect_text_tool_calls(c, KNOWN).len(), 2);
    }
}
