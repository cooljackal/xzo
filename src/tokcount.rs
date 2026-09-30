// SPDX-License-Identifier: Apache-2.0
//! Prompt token counting that does not depend on the model server.
//!
//! xzo decides when a conversation no longer fits the model's window by counting its tokens. The
//! count used to come from llama-server's `/tokenize`, with a character estimate standing in when
//! that failed. Most other servers (Ollama, MLX, LM Studio) have no `/tokenize`, and the estimate
//! under-counts code and JSON by up to ~2x -- a prompt it says fits can overrun the window.
//!
//! Counting with the model's own `tokenizer.json`, here, is exact and deterministic: same text,
//! same count, whatever server sits behind xzo. The chat template's per-message markers are not in
//! the text and are added by the caller as a fixed overhead.

use serde_json::Value;

/// Tokens the chat template adds once per prompt: BOS, the generation prompt, and — the big one —
/// the fixed instructions a template wraps around the tool definitions.
///
/// Measured with `xzo-probe tokverify` (Qwen2.5-family template, 40 prompts): at 8, short prompts
/// were under-counted by up to 45 tokens (0.2%). 128 covers that with room for templates whose tool
/// preamble is longer. Over-counting only compacts a little early; under-counting overruns the window.
/// Check a new model's template with `tokverify` before trusting this.
pub const PROMPT_OVERHEAD: usize = 128;
/// Tokens the chat template adds per message (role and delimiter markers).
pub const PER_MESSAGE_OVERHEAD: usize = 4;

/// The texts whose tokens make up a chat request, as the overflow gate counts it: each message's
/// flattened content plus its tool calls (agent turns carry their tokens there, not in content),
/// then the tools JSON. The template's own markers are [`prompt_overhead`].
///
/// ONE definition, used by the server and by `xzo-probe tokverify`, so what is verified against the
/// model server is exactly what the gate computes.
pub fn prompt_texts(messages: &[Value], tools: Option<&Value>) -> Vec<String> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    for m in messages {
        let mut text = crate::inject::flatten(m.get("content").unwrap_or(&Value::Null));
        if let Some(tc) = m.get("tool_calls") {
            if !tc.is_null() {
                text.push('\n');
                text.push_str(&tc.to_string());
            }
        }
        out.push(text);
    }
    if let Some(t) = tools {
        if !t.is_null() {
            out.push(t.to_string());
        }
    }
    out
}

pub fn prompt_overhead(n_messages: usize) -> usize {
    PROMPT_OVERHEAD + PER_MESSAGE_OVERHEAD * n_messages
}

/// The model's tokenizer, loaded from a Hugging Face `tokenizer.json` (`XZO_TOKENIZER`).
pub struct LocalTokenizer {
    tok: tokenizers::Tokenizer,
}

impl LocalTokenizer {
    pub fn load(path: &str) -> Result<Self, String> {
        tokenizers::Tokenizer::from_file(path)
            .map(|tok| Self { tok })
            .map_err(|e| format!("cannot load tokenizer {path}: {e}"))
    }

    pub fn from_bytes(json: &[u8]) -> Result<Self, String> {
        tokenizers::Tokenizer::from_bytes(json).map(|tok| Self { tok }).map_err(|e| e.to_string())
    }

    /// Tokens in `text`, without special tokens: the count llama-server's `/tokenize` returns for
    /// `{"content": text}`. If the tokenizer somehow fails on the text, twice the character
    /// estimate -- over-counting only fires compaction early; under-counting overruns the window.
    pub fn count(&self, text: &str) -> usize {
        self.tok
            .encode(text, false)
            .map(|e| e.len())
            .unwrap_or_else(|_| 2 * crate::inject::est_tokens(text))
    }
}

/// How the overflow gate counts prompt tokens. Chosen once, at startup.
pub enum Counter {
    /// The model's tokenizer, in xzo. Exact on any server.
    Local(LocalTokenizer),
    /// The core's `/tokenize` (llama-server, vLLM). Exact; one round-trip per new message.
    Core,
    /// `ceil(chars/4)`. Only when explicitly chosen (`XZO_EXACT_TOKENS=off`): it can under-count
    /// and let a prompt overrun the window.
    Estimate,
}

impl Counter {
    pub fn name(&self) -> &'static str {
        match self {
            Counter::Local(_) => "local tokenizer",
            Counter::Core => "core /tokenize",
            Counter::Estimate => "estimate (can under-count)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A three-word tokenizer, enough to show the count comes from the tokenizer and not from
    /// character length.
    const TINY: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}"#;

    #[test]
    fn counts_come_from_the_tokenizer() {
        let t = LocalTokenizer::from_bytes(TINY.as_bytes()).unwrap();
        assert_eq!(t.count("hello world"), 2);
        // 40 characters, one token per word: the chars/4 estimate would say 10.
        assert_eq!(t.count("hello  world  hello  world  hello  world"), 6);
        assert_eq!(t.count(""), 0);
    }

    #[test]
    fn the_count_is_deterministic() {
        let t = LocalTokenizer::from_bytes(TINY.as_bytes()).unwrap();
        let text = "hello world ".repeat(500);
        assert_eq!(t.count(&text), t.count(&text));
        assert_eq!(t.count(&text), 1000);
    }

    /// An agent turn's tokens are mostly in `tool_calls` and the `tools` definitions — the old
    /// estimate path counted neither.
    #[test]
    fn tool_calls_and_tool_definitions_are_counted() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "fix it"}),
            serde_json::json!({"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"src/main.rs\"}"}}]}),
        ];
        let tools = serde_json::json!([{"type": "function", "function": {"name": "read_file"}}]);
        let texts = prompt_texts(&messages, Some(&tools));
        assert_eq!(texts.len(), 3);
        assert!(texts[1].contains("read_file") && texts[1].contains("src/main.rs"));
        assert!(texts[2].contains("\"read_file\""));
        assert_eq!(prompt_overhead(2), PROMPT_OVERHEAD + 2 * PER_MESSAGE_OVERHEAD);
    }

    #[test]
    fn a_missing_file_is_an_error_not_a_guess() {
        assert!(LocalTokenizer::load("does/not/exist/tokenizer.json").is_err());
    }
}
