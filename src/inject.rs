// SPDX-License-Identifier: Apache-2.0
//! Text helpers shared across the modules: flatten message content, estimate and cap token
//! counts, split an answer into streaming frames.

/// Flatten OpenAI message content to plain text. Content is either a bare string or an array of
/// typed parts (`[{"type":"text","text":"…"}, {"type":"image_url", …}]`) — both are standard, and
/// many clients emit the array form.
///
/// Anything reading a message's text MUST go through here rather than `content.as_str()`, which
/// silently yields `""` for the array form. In `normalize.rs` that meant a system prompt (or a
/// user's question) could be replaced with an empty string and the request forwarded anyway
///.
pub fn flatten(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .map(|p| {
                if p.get("type").and_then(|t| t.as_str()) == Some("image_url") {
                    "[image omitted: text-only core]".to_string()
                } else {
                    p.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Rough token estimate: ~4 chars/token, rounded up.
///
/// CAVEAT: calibrated for Latin script. It under-counts code and JSON by ~10-15%, and
/// CJK by 3-4x (a CJK character is typically one token or more, not a quarter of one). The overflow
/// GATE uses the core's real tokenizer for exactly this reason; everything downstream — chunk
/// sizing, seed/recap/tail budgets, `notes_overflow` — still runs on this estimate.
pub fn est_tokens(s: &str) -> usize {
    let (cjk, other) = s.chars().fold((0usize, 0usize), |(c, o), ch| {
        if is_cjk(ch) {
            (c + 1, o)
        } else {
            (c, o + 1)
        }
    });
    cjk + other.div_ceil(4)
}

/// Characters that subword tokenizers do NOT pack four to a token.
///
/// `chars/4` is a Latin-script rule. Measured against two real tokenizers, CJK text comes out at
/// **one token per character** — so the old estimate under-counted it by 4x, and a "1500-token"
/// chunk of Japanese was really about 6000. That is not a tuning disagreement; it is large enough to
/// blow the context window outright, which is the over-window failure with a specific cause.
///
/// Counting these as one token each is measurement, not a guess: 308 CJK characters tokenized to
/// exactly 308 tokens under both `all-MiniLM-L6-v2` and `nomic-embed-text-v1.5`. Run
/// `xzo-probe tokcount` to reproduce.
///
/// SCOPED DELIBERATELY. Text with no CJK is counted exactly as before, so every existing benchmark
/// and every stored chunk boundary is bit-for-bit unchanged. The Latin-script constant is a separate
/// question and is left alone: the same measurement shows code at 1.26x (under-counted) and ordinary
/// conversation at 0.63x (over-counted), so no single divisor serves both and choosing one is a
/// retrieval-quality tradeoff that wants a core run behind it, not a hunch.
///
/// Note this does not make non-English traffic work — the English-only routing limit routes on English phrasings, so a CJK
/// question may not reach the right lane in the first place. This fixes the sizing, not the routing.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x303F   // CJK symbols and punctuation
        | 0x3040..=0x309F // hiragana
        | 0x30A0..=0x30FF // katakana
        | 0x3400..=0x4DBF // CJK unified ideographs extension A
        | 0x4E00..=0x9FFF // CJK unified ideographs
        | 0xAC00..=0xD7AF // hangul syllables
        | 0xF900..=0xFAFF // CJK compatibility ideographs
        | 0xFF00..=0xFFEF // full-width forms
    )
}

/// Truncate `text` to about `max_tokens` tokens on a char boundary; if cut, append a marker.
/// If it already fits, return it unchanged.
pub fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    let max_chars = max_tokens * 4;
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars).collect();
    format!("{} …[truncated]", truncated)
}

/// Split `content` into ordered pieces for paced-replay ("faux") streaming: each piece is at most
/// `max_frame_chars` characters and prefers whitespace/word boundaries. Delimiters are KEPT inside
/// the pieces, so `pieces.concat() == content` exactly (byte-for-byte). A single word longer than
/// `max_frame_chars` is hard-split on char boundaries (multibyte-safe). Empty input -> empty vec.
pub fn split_for_replay(content: &str, max_frame_chars: usize) -> Vec<String> {
    let cap = max_frame_chars.max(1);
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0usize; // char count of `cur`
    for seg in content.split_inclusive(|c: char| c.is_whitespace()) {
        let seg_len = seg.chars().count();
        if seg_len > cap {
            // flush the current piece, then hard-split the oversized segment on char boundaries
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
                cur_len = 0;
            }
            let mut piece = String::new();
            let mut plen = 0usize;
            for ch in seg.chars() {
                piece.push(ch);
                plen += 1;
                if plen == cap {
                    out.push(std::mem::take(&mut piece));
                    plen = 0;
                }
            }
            if !piece.is_empty() {
                out.push(piece);
            }
            continue;
        }
        if cur_len + seg_len > cap && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
            cur_len = 0;
        }
        cur.push_str(seg);
        cur_len += seg_len;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_est_tokens_rounds_up() {
        assert_eq!(est_tokens(""), 0);          // 0 chars -> 0 tokens
        assert_eq!(est_tokens("abcd"), 1);      // 4 chars -> 1 token
        assert_eq!(est_tokens("abcde"), 2);     // 5 chars -> 2 tokens
        assert_eq!(est_tokens("abc"), 1);       // 3 chars -> 1 token (rounds up)
        assert_eq!(est_tokens("a"), 1);         // 1 char -> 1 token
    }

    #[test]
    fn test_truncate_to_tokens_short_text_unchanged() {
        let text = "hello world";
        assert_eq!(truncate_to_tokens(text, 100), text);
    }

    #[test]
    fn test_truncate_to_tokens_long_text() {
        let text = "a".repeat(1000);  // 1000 chars
        let result = truncate_to_tokens(&text, 100);  // max 400 chars
        assert!(result.contains("…[truncated]"));
        assert!(result.chars().count() <= 420);  // 400 chars + marker
    }

    #[test]
    fn test_truncate_to_tokens_char_safe_multibyte() {
        // Test with multibyte characters (é is multibyte in UTF-8)
        let text = "café ".repeat(100);  // 500 chars (5 per "café ")
        let result = truncate_to_tokens(&text, 10);  // max 40 chars
        // Should not panic and should be valid UTF-8
        let char_count = result.chars().count();
        assert!(char_count <= 60);  // 40 chars + marker allowance
        assert!(result.contains("…[truncated]"));
    }

    #[test]
    fn est_tokens_is_roughly_quarter_len() {
        // Test that est_tokens(s) is within ±1 of ceil(L/4) for several ASCII strings

        // Empty string: 0 chars -> 0 tokens
        assert_eq!(est_tokens(""), 0);

        // 4 chars -> 1 token (4/4)
        assert_eq!(est_tokens("aaaa"), 1);

        // 5 chars -> 2 tokens (ceil(5/4))
        assert_eq!(est_tokens("aaaaa"), 2);

        // 8 chars -> 2 tokens (8/4)
        assert_eq!(est_tokens("aaaaaaaa"), 2);

        // 100 chars -> 25 tokens (100/4)
        let s100 = "a".repeat(100);
        let tokens_100 = est_tokens(&s100);
        assert_eq!(tokens_100, 25, "100 chars should be ~25 tokens, got {}", tokens_100);

        // 99 chars -> 25 tokens (ceil(99/4))
        let s99 = "a".repeat(99);
        let tokens_99 = est_tokens(&s99);
        assert_eq!(tokens_99, 25, "99 chars should be ~25 tokens, got {}", tokens_99);

        // 1 char -> 1 token (ceil(1/4))
        assert_eq!(est_tokens("a"), 1);

        // 3 chars -> 1 token (ceil(3/4))
        assert_eq!(est_tokens("abc"), 1);

        // 12 chars -> 3 tokens (12/4)
        let s12 = "a".repeat(12);
        assert_eq!(est_tokens(&s12), 3);

        // Real-world-ish string
        let msg = "hello world this is a test message";
        let len = msg.chars().count();
        let expected = (len + 3) / 4; // div_ceil
        let actual = est_tokens(msg);
        assert_eq!(actual, expected, "string '{}' ({} chars) -> {} tokens (expected {})", msg, len, actual, expected);
    }

    #[test]
    fn split_for_replay_reassembles_exactly() {
        let content = "The quick brown fox jumps over the lazy dog.\nSecond line here.";
        let pieces = split_for_replay(content, 8);
        assert_eq!(pieces.concat(), content, "pieces must reassemble to the original exactly");
        assert!(pieces.len() > 1, "a long string should split into multiple frames");
        assert!(pieces.iter().all(|p| p.chars().count() <= 8), "no piece exceeds the frame cap");
    }

    #[test]
    fn split_for_replay_hard_splits_long_word() {
        let content = "supercalifragilisticexpialidocious";
        let pieces = split_for_replay(content, 5);
        assert_eq!(pieces.concat(), content);
        assert!(pieces.iter().all(|p| p.chars().count() <= 5));
    }

    #[test]
    fn split_for_replay_multibyte_safe_and_edges() {
        let content = "café \u{1f600} déjà vu"; // multibyte chars
        let pieces = split_for_replay(content, 3);
        assert_eq!(pieces.concat(), content);
        assert!(split_for_replay("", 8).is_empty());
        // cap of 0 is treated as 1 (never loops forever, never empty piece)
        let p0 = split_for_replay("abc", 0);
        assert_eq!(p0.concat(), "abc");
        assert!(p0.iter().all(|p| p.chars().count() <= 1));
    }

    /// The the token under-count fix: CJK is one token per character, not four. The numbers come from
    /// `xzo-probe tokcount` against two real tokenizers, which agreed exactly.
    #[test]
    fn cjk_is_counted_one_token_per_character() {
        let ja = "認証サービスはログインルートの背後で動作しており";
        assert_eq!(est_tokens(ja), ja.chars().count());
        let zh = "这个服务是用围棋语言编写的";
        assert_eq!(est_tokens(zh), zh.chars().count());
    }

    /// Latin text is counted EXACTLY as before, which is what makes this safe to ship without a
    /// core run: no existing chunk boundary and no existing benchmark moves.
    #[test]
    fn latin_text_is_unchanged_by_the_cjk_fix() {
        for s in [
            "the service behind /login is written in golang",
            "{\"role\": \"user\", \"content\": \"what port?\"}",
            "",
            "a",
        ] {
            assert_eq!(est_tokens(s), s.chars().count().div_ceil(4), "changed for {s:?}");
        }
    }

    /// Mixed text splits by character, so a mostly-English message with a CJK phrase in it is not
    /// pushed wildly either way.
    #[test]
    fn mixed_script_counts_each_part_by_its_own_rule() {
        let latin = "the port is ";
        let mixed = format!("{latin}認証");
        assert_eq!(est_tokens(&mixed), latin.chars().count().div_ceil(4) + 2);
        // And it sits between the two pure cases rather than being dragged to either.
        assert!(est_tokens(&mixed) > est_tokens(latin));
        assert!(est_tokens(&mixed) < est_tokens("認証認証認証認証認証"));
    }
}
