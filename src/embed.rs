// SPDX-License-Identifier: Apache-2.0
//! Embedders for the semantic cache.
//!
//! The real small encoder — all-MiniLM-L6-v2 via `fastembed` (ONNX) —
//! replaces the dependency-free hashing bag-of-words. The hashing embedder stays as a
//! fallback when the model can't load (offline / first-run download failure) and for
//! `--no-default-features` builds (no ONNX runtime).

/// lowercase alphanumeric/underscore tokens
pub fn tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Code-aware tokenizer for grounding/lexical-containment scoring.
/// Splits identifiers but preserves useful units, producing lowercased tokens.
/// For an identifier like `snake_case`, `camelCase`, or `HTTPServer`,
/// emits BOTH the whole identifier AND its sub-tokens.
/// Also preserves dotted paths, file paths, and error strings as whole units.
/// Deduplicates while preserving first-seen order.
pub fn code_tokens(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut cur = String::new();

    for ch in s.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch == '.' || ch == '/' || ch == '-' {
            cur.push(ch);
        } else if !cur.is_empty() {
            process_code_unit(&cur, &mut result, &mut seen);
            cur.clear();
        }
    }

    if !cur.is_empty() {
        process_code_unit(&cur, &mut result, &mut seen);
    }

    result
}

fn process_code_unit(unit: &str, result: &mut Vec<String>, seen: &mut std::collections::HashSet<String>) {
    let lower_unit = unit.to_ascii_lowercase();

    // Add the whole unit
    add_token(&lower_unit, result, seen);

    // Split on separators and add parts
    for part in unit.split(|c: char| c == '.' || c == '/' || c == '-' || c == '_') {
        if !part.is_empty() {
            let lower_part = part.to_ascii_lowercase();
            add_token(&lower_part, result, seen);

            // Also split camelCase within this part
            let camel_parts = split_camel_case_parts(part);
            for camel_part in camel_parts {
                let lower_camel = camel_part.to_ascii_lowercase();
                add_token(&lower_camel, result, seen);
            }
        }
    }
}

fn add_token(token: &str, result: &mut Vec<String>, seen: &mut std::collections::HashSet<String>) {
    if !token.is_empty() && seen.insert(token.to_string()) {
        result.push(token.to_string());
    }
}

fn split_camel_case_parts(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return vec![];
    }

    let mut parts = Vec::new();
    let mut current = String::new();

    for (i, &ch) in chars.iter().enumerate() {
        if i == 0 {
            current.push(ch);
        } else {
            let prev_is_upper = chars[i - 1].is_ascii_uppercase();
            let curr_is_upper = ch.is_ascii_uppercase();
            let next_is_lower = (i + 1 < chars.len()) && chars[i + 1].is_ascii_lowercase();

            if curr_is_upper && chars[i - 1].is_ascii_lowercase() {
                // Lowercase to uppercase: new word
                parts.push(current.clone());
                current = ch.to_string();
            } else if curr_is_upper && next_is_lower && prev_is_upper {
                // Uppercase (start of new word in acronym)
                parts.push(current.clone());
                current = ch.to_string();
            } else {
                current.push(ch);
            }
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

pub trait Embedder: Send + Sync {
    fn encode(&self, text: &str) -> Vec<f32>;
    fn dim(&self) -> usize;
    fn name(&self) -> &'static str;
}

/// Deterministic hashed bag-of-words (no model, no network). L2-normalized.
pub struct HashingEmbedder {
    dim: usize,
}
impl HashingEmbedder {
    pub fn new() -> Self {
        Self { dim: 256 }
    }
    #[allow(dead_code)] // public constructor; exercised by the dim-reconcile test
    pub fn with_dim(dim: usize) -> Self {
        Self { dim }
    }
}
impl Embedder for HashingEmbedder {
    fn encode(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        for tok in tokens(text) {
            v[(fnv1a(&tok) as usize) % self.dim] += 1.0;
        }
        l2_normalize(&mut v);
        v
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn name(&self) -> &'static str {
        "hashing"
    }
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// all-MiniLM-L6-v2 via fastembed (ONNX). 384-dim, normalized. The model is fetched
/// and cached on first construction.
#[cfg(feature = "semantic")]
pub struct MiniLmEmbedder {
    model: fastembed::TextEmbedding,
}

#[cfg(feature = "semantic")]
impl MiniLmEmbedder {
    pub fn try_new() -> anyhow::Result<Self> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
        )?;
        Ok(Self { model })
    }
}

#[cfg(feature = "semantic")]
impl Embedder for MiniLmEmbedder {
    fn encode(&self, text: &str) -> Vec<f32> {
        match self.model.embed(vec![text], None) {
            Ok(mut v) => v.pop().unwrap_or_else(|| vec![0.0; 384]),
            Err(_) => vec![0.0; 384],
        }
    }
    fn dim(&self) -> usize {
        384
    }
    fn name(&self) -> &'static str {
        "all-MiniLM-L6-v2"
    }
}

/// Build the configured embedder (shared via Arc): MiniLM if available, else hashing.
pub fn build() -> std::sync::Arc<dyn Embedder> {
    #[cfg(feature = "semantic")]
    {
        match MiniLmEmbedder::try_new() {
            Ok(m) => {
                println!("embedder: all-MiniLM-L6-v2 (fastembed)");
                return std::sync::Arc::new(m);
            }
            Err(e) => eprintln!("embedder: MiniLM load failed ({e}); using hashing fallback"),
        }
    }
    println!("embedder: hashing (bag-of-words)");
    std::sync::Arc::new(HashingEmbedder::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_lowercases_and_splits() {
        assert_eq!(tokens("Hello, World_1! foo"), vec!["hello", "world_1", "foo"]);
        assert!(tokens("   ").is_empty());
    }

    #[test]
    fn hashing_encode_is_normalized_and_deterministic() {
        let e = HashingEmbedder::new();
        let v = e.encode("reverse a string in python");
        assert_eq!(v.len(), 256);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
        assert_eq!(v, e.encode("reverse a string in python")); // deterministic
        assert_eq!(HashingEmbedder::with_dim(384).encode("x").len(), 384);
    }

    #[test]
    fn preserves_identifier_and_subtokens() {
        let tokens = code_tokens("parseConfig");
        assert!(tokens.contains(&"parseconfig".to_string()));
        assert!(tokens.contains(&"parse".to_string()));
        assert!(tokens.contains(&"config".to_string()));

        let tokens = code_tokens("read_file");
        assert!(tokens.contains(&"read_file".to_string()));
        assert!(tokens.contains(&"read".to_string()));
        assert!(tokens.contains(&"file".to_string()));

        let tokens = code_tokens("HTTPServer");
        assert!(tokens.contains(&"httpserver".to_string()));
        assert!(tokens.contains(&"http".to_string()));
        assert!(tokens.contains(&"server".to_string()));
    }

    #[test]
    fn keeps_error_strings_and_paths() {
        let tokens = code_tokens("open config.yaml -> E0433");
        assert!(tokens.contains(&"config.yaml".to_string()));
        assert!(tokens.contains(&"config".to_string()));
        assert!(tokens.contains(&"yaml".to_string()));
        assert!(tokens.contains(&"e0433".to_string()));

        let tokens = code_tokens("src/main.rs");
        assert!(tokens.contains(&"src/main.rs".to_string()));
        assert!(tokens.contains(&"src".to_string()));
        assert!(tokens.contains(&"main".to_string()));
        assert!(tokens.contains(&"rs".to_string()));
    }

    #[test]
    fn dedups_preserving_order() {
        let tokens = code_tokens("a a b");
        assert_eq!(tokens, vec!["a", "b"]);

        // Should not have duplicates
        let seen: std::collections::HashSet<_> = tokens.iter().cloned().collect();
        assert_eq!(seen.len(), tokens.len());

        let tokens = code_tokens("parseConfig parseconfig");
        assert_eq!(tokens, vec!["parseconfig", "parse", "config"]);
    }
}
