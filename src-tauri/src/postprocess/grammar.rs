//! Stage 3: Grammar correction via nlprule (LanguageTool rules).
//!
//! Uses ~5400 LanguageTool rules compiled into binary data files.
//! Handles grammar, style, and some spelling corrections.
//! Pure Rust (fancy-regex backend), no network calls.

use std::io::Cursor;

use nlprule::{Rules, Tokenizer};

static RULES_BIN: &[u8] = include_bytes!("../../data/nlprule/en_rules.bin");
static TOKENIZER_BIN: &[u8] = include_bytes!("../../data/nlprule/en_tokenizer.bin");

pub struct GrammarChecker {
    tokenizer: Tokenizer,
    rules: Rules,
}

impl GrammarChecker {
    pub fn new() -> Result<Self, String> {
        let tokenizer = Tokenizer::from_reader(Cursor::new(TOKENIZER_BIN))
            .map_err(|e| format!("nlprule tokenizer: {e}"))?;
        let rules = Rules::from_reader(Cursor::new(RULES_BIN))
            .map_err(|e| format!("nlprule rules: {e}"))?;
        Ok(Self { tokenizer, rules })
    }

    /// Correct grammatical errors in the text.
    pub fn correct(&self, text: &str) -> String {
        // Suppress nlprule's verbose INFO-level disambiguation logging.
        let prev = log::max_level();
        log::set_max_level(log::LevelFilter::Warn);
        let result = self.rules.correct(text, &self.tokenizer);
        log::set_max_level(prev);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_without_panic() {
        let _g = GrammarChecker::new().unwrap();
    }

    #[test]
    fn preserves_clean_text() {
        let g = GrammarChecker::new().unwrap();
        let input = "This is a clean sentence.";
        let output = g.correct(input);
        assert_eq!(output, input);
    }

    #[test]
    fn corrects_grammar() {
        let g = GrammarChecker::new().unwrap();
        let output = g.correct("She was not been here since Monday.");
        // nlprule picks "was not being" as the first replacement
        assert_ne!(output, "She was not been here since Monday.");
    }

    #[test]
    fn returns_non_empty_for_input() {
        let g = GrammarChecker::new().unwrap();
        let output = g.correct("hello world");
        assert!(!output.is_empty());
    }
}
