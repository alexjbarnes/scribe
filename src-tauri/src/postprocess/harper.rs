//! Stage 5: Harper grammar and spelling polish.
//!
//! Uses harper-core's ~200 curated grammar rules for final cleanup.
//! Handles idiom corrections, spelling, and style issues.
//! Pure Rust, no network calls, ~5-10ms.

use std::sync::Mutex;

use harper_core::linting::{LintGroup, Linter};
use harper_core::spell::FstDictionary;
use harper_core::{Dialect, Document};

pub struct HarperLinter {
    linter: Mutex<LintGroup>,
}

// Safety: LintGroup is only accessed behind a Mutex, never shared across threads
// without synchronization. The dyn Linter trait objects are not Send but we
// guarantee single-threaded access via the Mutex.
unsafe impl Send for HarperLinter {}
unsafe impl Sync for HarperLinter {}

impl HarperLinter {
    pub fn new() -> Self {
        let dict = FstDictionary::curated();
        let linter = LintGroup::new_curated(dict, Dialect::American);
        Self {
            linter: Mutex::new(linter),
        }
    }

    /// Lint the text and apply all suggested fixes.
    pub fn lint_and_fix(&self, text: &str) -> String {
        use harper_core::linting::LintKind;

        let doc = Document::new_plain_english_curated(text);
        let mut linter = self.linter.lock().unwrap();
        let lints = linter.lint(&doc);

        if lints.is_empty() {
            return text.to_string();
        }

        let chars: Vec<char> = text.chars().collect();

        // Collect lints that have suggestions, sorted by span start descending
        // so we apply from end to start (avoids offset invalidation).
        let mut fixes: Vec<_> = lints
            .into_iter()
            .filter(|lint| {
                if lint.suggestions.is_empty() {
                    return false;
                }
                // Skip spelling corrections on all-uppercase words (acronyms).
                // Harper doesn't know domain acronyms like POC, API, etc. and
                // will "correct" them to similar dictionary words.
                if lint.lint_kind == LintKind::Spelling {
                    let word: String = chars[lint.span.start..lint.span.end].iter().collect();
                    if word.len() >= 2 && word.chars().all(|c| c.is_ascii_uppercase()) {
                        log::debug!("Harper: skipping spelling correction on acronym \"{word}\"");
                        return false;
                    }
                }
                true
            })
            .collect();

        fixes.sort_by(|a, b| b.span.start.cmp(&a.span.start));

        let mut result = chars;
        for lint in &fixes {
            lint.suggestions[0].apply(lint.span, &mut result);
        }

        result.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_without_panic() {
        let _h = HarperLinter::new();
    }

    #[test]
    fn preserves_clean_text() {
        let h = HarperLinter::new();
        let input = "This is a clean sentence.";
        let output = h.lint_and_fix(input);
        assert_eq!(output, input);
    }

    #[test]
    fn returns_non_empty_for_input() {
        let h = HarperLinter::new();
        let output = h.lint_and_fix("hello world");
        assert!(!output.is_empty());
    }

    #[test]
    fn preserves_uppercase_acronyms() {
        let h = HarperLinter::new();
        // POC was being "corrected" to POS
        assert!(h.lint_and_fix("Would we do POC test?").contains("POC"));
        assert!(h.lint_and_fix("The API is down.").contains("API"));
        assert!(h.lint_and_fix("Check the DNS settings.").contains("DNS"));
        assert!(h.lint_and_fix("Run the CI pipeline.").contains("CI"));
    }
}
