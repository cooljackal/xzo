// SPDX-License-Identifier: Apache-2.0
//! Query router for the overflow layer: pick the retrieval lane for a query.
//!
//! `Targeted` -> top-k retrieval + tool loop (cheap, sharp). `Exhaustive` -> digest ALL chunks
//! (needed for count/summarize-all/aggregation, which top-k can't answer). The keyword backend is
//! dependency-free substring matching over a `|`-separated pattern set; the classifier backend is
//! plumbing: main.rs fetches an external verdict (async) and passes it in here.

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Route {
    Targeted,
    Exhaustive,
}

/// Parse a forced-lane override: `"targeted"`/`"exhaustive"` -> `Some(..)`; `"auto"`/anything
/// else -> `None` (let the router decide).
pub fn parse_override(s: &str) -> Option<Route> {
    match s.trim().to_lowercase().as_str() {
        "targeted" => Some(Route::Targeted),
        "exhaustive" => Some(Route::Exhaustive),
        _ => None,
    }
}

/// Keyword lane: `Exhaustive` if the (lowercased) query contains any of the `|`-separated
/// patterns (each matched as a lowercased substring), else `Targeted`. Empty pattern set or empty
/// query -> `Targeted`.
pub fn route_keyword(query: &str, patterns: &str) -> Route {
    let q = query.to_lowercase();
    for p in patterns.split('|') {
        let p = p.trim().to_lowercase();
        if !p.is_empty() && q.contains(&p) {
            return Route::Exhaustive;
        }
    }
    Route::Targeted
}

/// Could the keyword pattern set even have matched this query?
///
/// The shipped patterns are English phrases matched as substrings. A query written in a script with
/// no Latin letters cannot contain one, so `route_keyword` returns `Targeted` without having made a
/// decision — it fell through. From the outside that is indistinguishable from "looked, and this is
/// a lookup question", and the difference is the whole of the English-only routing limit: an aggregation question routed to the
/// targeted lane gets a confident answer computed off a fraction of the data.
///
/// Measured with `xzo-probe routing`: **0 of 15** non-English aggregation questions route correctly,
/// against 3 of 3 in English. The lookup column reads a perfect 100% in every language, which is not
/// good news — it is perfect because nothing matches, so everything falls through to the right answer
/// by accident.
///
/// This DETECTS, it does not correct, for the reason the finding gives: the repair is a measured
/// classifier, and adding translations to the pattern list is a longer list rather than a mechanism.
/// Counting the blind cases is what turns a silent wrong answer into a visible one.
///
/// It is a partial detector and says so: it catches a script mismatch, not a language mismatch.
/// Spanish, German and French all use Latin letters, so they are invisible here even though the
/// probe shows they route just as wrongly. It is a floor on the problem, never a measure of it.
pub fn keyword_patterns_cannot_match(query: &str) -> bool {
    !query.trim().is_empty() && !query.chars().any(|c| c.is_ascii_alphabetic())
}

/// Decide the lane. Precedence: explicit `route_override` wins; else in `classifier` mode a
/// present `classifier_verdict` wins; else the keyword lane. `mode` is `"keyword"` or
/// `"classifier"`.
pub fn route(
    query: &str,
    route_override: &str,
    mode: &str,
    patterns: &str,
    classifier_verdict: Option<Route>,
) -> Route {
    if let Some(r) = parse_override(route_override) {
        return r;
    }
    if mode.trim().eq_ignore_ascii_case("classifier") {
        if let Some(r) = classifier_verdict {
            return r;
        }
    }
    route_keyword(query, patterns)
}

/// Parse an external classifier's reply text into a verdict. Accepts a bare word or JSON with a
/// `route`/`label` field; case-insensitive; `None` if it says neither. (Plumbing for
/// `XZO_OVERFLOW_CLASSIFIER`; main.rs does the HTTP call.)
pub fn parse_classifier_verdict(reply: &str) -> Option<Route> {
    let low = reply.to_lowercase();
    // exhaustive wins ties if both appear (aggregation is the safer default when ambiguous)
    if low.contains("exhaustive") || low.contains("global") || low.contains("aggregat") {
        return Some(Route::Exhaustive);
    }
    if low.contains("targeted") || low.contains("lookup") || low.contains("needle") {
        return Some(Route::Targeted);
    }
    None
}

/// Context-handling strategy, orthogonal to the extraction `Route{Targeted,Exhaustive}` lanes.
/// `Compaction` pins the system + recent tail verbatim and compresses only the old middle
/// (agentic/chat traffic); `Extraction` is the existing digest->answer-from-notes doc-Q&A path.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ContextStrategy {
    Compaction,
    Extraction,
}

/// Parse a forced `XZO_OVERFLOW_MODE`: `"compaction"`/`"extraction"` -> `Some(..)`;
/// `"auto"`/anything else -> `None` (let `detect_strategy` decide).
pub fn parse_mode(s: &str) -> Option<ContextStrategy> {
    match s.trim().to_lowercase().as_str() {
        "compaction" => Some(ContextStrategy::Compaction),
        "extraction" => Some(ContextStrategy::Extraction),
        _ => None,
    }
}

/// The agentic-markers gate: agent traffic (which advertises `tools` or contains tool turns)
/// gets `Compaction`; everything else gets `Extraction`. Deliberately does NOT trigger on
/// system-presence or assistant-turn-count (both would sweep the toolless benchmarks/RAG into
/// compaction and regress the extraction-tuned path).
pub fn detect_strategy(request_has_tools: bool, has_tool_calls: bool, has_tool_messages: bool) -> ContextStrategy {
    if request_has_tools || has_tool_calls || has_tool_messages {
        ContextStrategy::Compaction
    } else {
        ContextStrategy::Extraction
    }
}

/// Resolve the context strategy: a forced `mode` (`XZO_OVERFLOW_MODE`) wins; otherwise the
/// agentic-markers gate decides.
pub fn strategy(mode: &str, request_has_tools: bool, has_tool_calls: bool, has_tool_messages: bool) -> ContextStrategy {
    if let Some(s) = parse_mode(mode) {
        return s;
    }
    detect_strategy(request_has_tools, has_tool_calls, has_tool_messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATTERNS: &str = "how many|every|all of|count|total|summarize";

    #[test]
    fn keyword_routes_aggregation_to_exhaustive() {
        assert_eq!(route_keyword("how many TODOs are there?", PATTERNS), Route::Exhaustive);
        assert_eq!(route_keyword("Count the errors", PATTERNS), Route::Exhaustive);
        assert_eq!(route_keyword("summarize everything", PATTERNS), Route::Exhaustive);
    }

    #[test]
    fn keyword_routes_lookup_to_targeted() {
        assert_eq!(route_keyword("what port is in the config?", PATTERNS), Route::Targeted);
        assert_eq!(route_keyword("which db does /login use?", PATTERNS), Route::Targeted);
    }

    #[test]
    fn route_override_forces_lane() {
        // override beats everything, even an aggregation query
        assert_eq!(route("how many?", "targeted", "keyword", PATTERNS, None), Route::Targeted);
        assert_eq!(route("what port?", "exhaustive", "keyword", PATTERNS, None), Route::Exhaustive);
        assert_eq!(parse_override("auto"), None);
    }

    #[test]
    fn classifier_mode_plumbs_external_verdict() {
        // in classifier mode, a present verdict wins over the keyword lane
        assert_eq!(
            route("what port?", "auto", "classifier", PATTERNS, Some(Route::Exhaustive)),
            Route::Exhaustive
        );
        // no verdict -> falls back to keyword
        assert_eq!(route("what port?", "auto", "classifier", PATTERNS, None), Route::Targeted);
        // verdict parsing
        assert_eq!(parse_classifier_verdict("exhaustive"), Some(Route::Exhaustive));
        assert_eq!(parse_classifier_verdict("targeted lookup"), Some(Route::Targeted));
        assert_eq!(parse_classifier_verdict("needle"), Some(Route::Targeted));
        assert_eq!(parse_classifier_verdict("dunno"), None);
    }

    #[test]
    fn parse_mode_forces_or_defers() {
        assert_eq!(parse_mode("compaction"), Some(ContextStrategy::Compaction));
        assert_eq!(parse_mode("EXTRACTION"), Some(ContextStrategy::Extraction));
        assert_eq!(parse_mode("auto"), None);
        assert_eq!(parse_mode("garbage"), None);
    }

    #[test]
    fn detect_strategy_truth_table() {
        // any agentic marker -> Compaction
        assert_eq!(detect_strategy(true, false, false), ContextStrategy::Compaction);
        assert_eq!(detect_strategy(false, true, false), ContextStrategy::Compaction);
        assert_eq!(detect_strategy(false, false, true), ContextStrategy::Compaction);
        // no markers (toolless benchmark/RAG) -> Extraction, unchanged
        assert_eq!(detect_strategy(false, false, false), ContextStrategy::Extraction);
    }

    #[test]
    fn strategy_forced_mode_overrides_gate() {
        // forced compaction even with no markers
        assert_eq!(strategy("compaction", false, false, false), ContextStrategy::Compaction);
        // forced extraction even when markers are present
        assert_eq!(strategy("extraction", true, true, true), ContextStrategy::Extraction);
        // auto defers to the gate
        assert_eq!(strategy("auto", true, false, false), ContextStrategy::Compaction);
        assert_eq!(strategy("auto", false, false, false), ContextStrategy::Extraction);
    }

    /// The English-only routing limit's detector. It must fire on a script that cannot contain an English phrase, and stay
    /// quiet whenever the pattern set had a real chance.
    #[test]
    fn the_blind_case_is_detected_for_non_latin_scripts() {
        assert!(keyword_patterns_cannot_match("いくつのサービスがありますか？"));
        assert!(keyword_patterns_cannot_match("有多少个服务？"));
        assert!(!keyword_patterns_cannot_match("how many services are there?"));
        // An empty query is not a blind case -- there is nothing to route.
        assert!(!keyword_patterns_cannot_match(""));
        assert!(!keyword_patterns_cannot_match("   "));
    }

    /// The detector is deliberately partial, and the test says so out loud: Latin-script languages
    /// route just as wrongly and are NOT caught. If someone later makes this pass, they have widened
    /// the detector and the probe needs re-running.
    #[test]
    fn latin_script_languages_are_not_caught_and_that_is_known() {
        assert!(!keyword_patterns_cannot_match("¿cuántos servicios usan postgres?"));
        assert!(!keyword_patterns_cannot_match("wie viele dienste verwenden postgres?"));
    }
}
