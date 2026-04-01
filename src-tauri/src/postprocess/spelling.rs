//! SymSpell spell correction stage.
//!
//! Uses edit-distance-based correction with frequency dictionaries to fix
//! ASR transcription misspellings that Harper's dictionary misses.
//! Operates word-by-word to preserve punctuation and contractions.
//! Sub-millisecond per sentence.

use symspell::{AsciiStringStrategy, SymSpell, SymSpellBuilder, Verbosity};

const UNIGRAM_DICT: &str = include_str!("../../data/frequency_dictionary_en_82_765.txt");

pub struct SpellCorrector {
    symspell: SymSpell<AsciiStringStrategy>,
}

impl SpellCorrector {
    pub fn new() -> Self {
        let mut symspell: SymSpell<AsciiStringStrategy> = SymSpellBuilder::default()
            .max_dictionary_edit_distance(2)
            .prefix_length(7)
            .count_threshold(1)
            .build()
            .unwrap();

        for line in UNIGRAM_DICT.lines() {
            symspell.load_dictionary_line(line, 0, 1, " ");
        }

        Self { symspell }
    }

    /// Correct spelling word-by-word, preserving punctuation and contractions.
    pub fn correct(&self, text: &str) -> String {
        if text.trim().is_empty() {
            return text.to_string();
        }

        let mut result = String::with_capacity(text.len());
        let mut last_end = 0;

        // Walk through the text, finding word boundaries while preserving
        // all whitespace and punctuation exactly as-is.
        for (i, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if i > last_end {
                    // We have a token from last_end..i
                    result.push_str(&self.correct_token(&text[last_end..i]));
                }
                result.push(ch);
                last_end = i + ch.len_utf8();
            }
        }
        // Handle final token
        if last_end < text.len() {
            result.push_str(&self.correct_token(&text[last_end..]));
        }

        result
    }

    /// Correct a single token (word possibly with attached punctuation).
    ///
    /// Strips leading/trailing punctuation, checks if the core word should
    /// be corrected, then reassembles with original punctuation.
    fn correct_token(&self, token: &str) -> String {
        // Split into leading punct + core word + trailing punct
        let (leading, rest) = split_leading_punct(token);
        let (core, trailing) = split_trailing_punct(rest);

        if core.is_empty() {
            return token.to_string();
        }

        // Never correct protected words
        if should_protect(core) {
            return token.to_string();
        }

        let lower = core.to_lowercase();
        let suggestions = self.symspell.lookup(&lower, Verbosity::Top, 2);

        match suggestions.first() {
            Some(s) if s.distance > 0 => {
                // If the word already exists in the dictionary (distance 0
                // match), leave it alone. Only correct unknown words.
                let exact = self.symspell.lookup(&lower, Verbosity::Top, 0);
                if !exact.is_empty() {
                    return token.to_string();
                }
                // Reject corrections that shorten the word (e.g. "repo" -> "rep").
                // Real misspellings rarely produce shorter corrections.
                if s.term.len() < lower.len() {
                    return token.to_string();
                }
                log::debug!("Spelling: \"{}\" -> \"{}\" (distance={})", core, s.term, s.distance);
                let corrected = transfer_case(core, &s.term);
                format!("{leading}{corrected}{trailing}")
            }
            _ => token.to_string(),
        }
    }
}

/// Check if a word should be protected from spell correction.
fn should_protect(word: &str) -> bool {
    if word.is_empty() {
        return true;
    }
    // Contains apostrophe: contraction (we've, don't, I'm, it's)
    if word.contains('\'') || word.contains('\u{2019}') {
        return true;
    }
    // All uppercase (2+ chars): acronym (API, DNS, POC)
    if word.len() >= 2 && word.chars().all(|c| c.is_ascii_uppercase()) {
        return true;
    }
    // Contains digits (numbers, codes, versions)
    if word.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    // Single character — not worth correcting
    if word.chars().count() <= 1 {
        return true;
    }
    false
}

/// Transfer the casing pattern from the original word to the corrected word.
fn transfer_case(original: &str, corrected: &str) -> String {
    if original.chars().all(|c| c.is_uppercase()) {
        // ALL CAPS -> ALL CAPS
        corrected.to_uppercase()
    } else if original.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        // Title Case -> Title Case
        let mut chars = corrected.chars();
        match chars.next() {
            Some(first) => first.to_uppercase().to_string() + chars.as_str(),
            None => corrected.to_string(),
        }
    } else {
        corrected.to_string()
    }
}

/// Split leading punctuation from a string.
/// ",hello" -> (",", "hello")
fn split_leading_punct(s: &str) -> (&str, &str) {
    let start = s
        .char_indices()
        .find(|(_, c)| !c.is_ascii_punctuation())
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    (&s[..start], &s[start..])
}

/// Split trailing punctuation from a string.
/// "hello," -> ("hello", ",")
fn split_trailing_punct(s: &str) -> (&str, &str) {
    let end = s
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_punctuation())
        .last()
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    (&s[..end], &s[end..])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corrector() -> SpellCorrector {
        SpellCorrector::new()
    }

    #[test]
    fn preserves_correct_text() {
        let c = corrector();
        assert_eq!(c.correct("hello world"), "hello world");
    }

    #[test]
    fn preserves_empty() {
        let c = corrector();
        assert_eq!(c.correct(""), "");
        assert_eq!(c.correct("  "), "  ");
    }

    #[test]
    fn preserves_acronyms() {
        let c = corrector();
        let result = c.correct("the API is working");
        assert!(result.contains("API"), "got: {result}");
    }

    #[test]
    fn preserves_numbers() {
        let c = corrector();
        let result = c.correct("I have 23 items");
        assert!(result.contains("23"), "got: {result}");
    }

    #[test]
    fn corrects_misspelling() {
        let c = corrector();
        let result = c.correct("becuase it works");
        assert_eq!(result, "because it works", "got: {result}");
    }

    #[test]
    fn preserves_contractions() {
        let c = corrector();
        assert_eq!(
            c.correct("I don't think we've got that"),
            "I don't think we've got that"
        );
        assert_eq!(c.correct("it's working"), "it's working");
        assert_eq!(c.correct("I'm here"), "I'm here");
    }

    #[test]
    fn preserves_commas() {
        let c = corrector();
        let result = c.correct("Yes please, particularly around this");
        assert!(result.contains("please,"), "comma lost: {result}");
    }

    #[test]
    fn preserves_sentence_punctuation() {
        let c = corrector();
        let result = c.correct("hello world.");
        assert!(result.ends_with('.'), "got: {result}");
        let result = c.correct("is it working?");
        assert!(result.ends_with('?'), "got: {result}");
    }

    #[test]
    fn preserves_valid_words() {
        let c = corrector();
        // "repo" is a valid word, should not be "corrected" to "rep"
        let result = c.correct("the repo is clean");
        assert!(result.contains("repo"), "repo mangled: {result}");
    }

    #[test]
    fn preserves_casing() {
        let c = corrector();
        let result = c.correct("Becuase it works");
        assert!(result.starts_with("Because"), "got: {result}");
    }

    #[test]
    fn should_protect_contractions() {
        assert!(should_protect("we've"));
        assert!(should_protect("don't"));
        assert!(should_protect("I'm"));
        assert!(should_protect("it's"));
    }

    #[test]
    fn should_protect_acronyms() {
        assert!(should_protect("API"));
        assert!(should_protect("DNS"));
    }

    #[test]
    fn should_protect_numbers() {
        assert!(should_protect("23"));
        assert!(should_protect("v2"));
    }

    #[test]
    fn should_protect_single_chars() {
        assert!(should_protect("I"));
        assert!(should_protect("a"));
    }

    #[test]
    fn split_punct_works() {
        assert_eq!(split_trailing_punct("hello,"), ("hello", ","));
        assert_eq!(split_trailing_punct("world."), ("world", "."));
        assert_eq!(split_trailing_punct("hello"), ("hello", ""));
        assert_eq!(split_trailing_punct("..."), ("", "..."));
    }

    #[test]
    fn split_leading_punct_works() {
        assert_eq!(split_leading_punct("(hello"), ("(", "hello"));
        assert_eq!(split_leading_punct("hello"), ("", "hello"));
        assert_eq!(split_leading_punct("\"word"), ("\"", "word"));
    }

    #[test]
    fn transfer_case_works() {
        assert_eq!(transfer_case("Hello", "world"), "World");
        assert_eq!(transfer_case("HELLO", "world"), "WORLD");
        assert_eq!(transfer_case("hello", "world"), "world");
    }
}
