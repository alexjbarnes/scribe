//! Stage 1: Filler word removal.
//!
//! Strips disfluencies common in spoken language: "um", "uh", "er",
//! filler phrases like "you know" and "I mean", false starts, and
//! consecutive repeated words.

/// Multi-word filler phrases to remove (checked first, case-insensitive).
const FILLER_PHRASES: &[&str] = &[
    "you know",
    "i mean",
    "you see",
    "i guess",
    "i think um",
    "i think uh",
];

/// Single filler words to remove when standalone (not part of real speech).
const FILLER_WORDS: &[&str] = &[
    "um", "umm", "uh", "uhh", "er", "err", "ah", "ahh", "hmm", "hm",
    "mm", "mmm", "mhm",
];

/// Remove filler words, filler phrases, and consecutive repeated words.
pub fn remove_fillers(text: &str) -> String {
    let mut s = text.to_string();

    // Remove multi-word filler phrases first (case-insensitive)
    for phrase in FILLER_PHRASES {
        s = remove_phrase_case_insensitive(&s, phrase);
    }

    // Tokenize, remove single fillers and consecutive duplicates
    let words: Vec<&str> = s.split_whitespace().collect();
    let mut result: Vec<&str> = Vec::with_capacity(words.len());

    for word in &words {
        let lower = word.to_ascii_lowercase();
        let bare = lower.trim_matches(|c: char| c.is_ascii_punctuation());

        // Skip single filler words
        if FILLER_WORDS.contains(&bare) {
            continue;
        }

        // Skip consecutive duplicate words (false starts like "I I think")
        if let Some(prev) = result.last() {
            let prev_lower = prev.to_ascii_lowercase();
            let prev_bare = prev_lower.trim_matches(|c: char| c.is_ascii_punctuation());
            if bare == prev_bare && !bare.is_empty() {
                continue;
            }
        }

        result.push(word);
    }

    result.join(" ")
}

/// Remove all occurrences of a phrase, case-insensitive, preserving surrounding text.
fn remove_phrase_case_insensitive(text: &str, phrase: &str) -> String {
    let lower = text.to_lowercase();
    let mut result = String::with_capacity(text.len());
    let mut pos = 0;

    while let Some(idx) = lower[pos..].find(phrase) {
        let abs_idx = pos + idx;

        // Only remove if it's at a word boundary
        let before_ok = abs_idx == 0
            || text.as_bytes()[abs_idx - 1].is_ascii_whitespace();
        let after_end = abs_idx + phrase.len();
        let after_ok = after_end >= text.len()
            || text.as_bytes()[after_end].is_ascii_whitespace()
            || text.as_bytes()[after_end].is_ascii_punctuation();

        if before_ok && after_ok {
            result.push_str(&text[pos..abs_idx]);
            pos = after_end;
        } else {
            result.push_str(&text[pos..abs_idx + 1]);
            pos = abs_idx + 1;
        }
    }

    result.push_str(&text[pos..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_um_uh() {
        assert_eq!(remove_fillers("um hello uh world"), "hello world");
    }

    #[test]
    fn removes_filler_phrases() {
        assert_eq!(
            remove_fillers("so you know the thing is"),
            "so the thing is"
        );
    }

    #[test]
    fn removes_repeated_words() {
        assert_eq!(remove_fillers("I I think so"), "I think so");
        assert_eq!(remove_fillers("the the cat"), "the cat");
    }

    #[test]
    fn preserves_legitimate_words() {
        assert_eq!(remove_fillers("I mean the river"), "the river");
        assert_eq!(remove_fillers("hello world"), "hello world");
        // "sort of" and "kind of" are not fillers — they have legitimate uses
        assert_eq!(remove_fillers("it's sort of complicated"), "it's sort of complicated");
        assert_eq!(remove_fillers("kind of a big deal"), "kind of a big deal");
    }

    #[test]
    fn handles_empty() {
        assert_eq!(remove_fillers(""), "");
        assert_eq!(remove_fillers("um"), "");
    }

    #[test]
    fn case_insensitive_fillers() {
        assert_eq!(remove_fillers("Um Hello Uh World"), "Hello World");
    }

    #[test]
    fn preserves_punctuation_context() {
        assert_eq!(
            remove_fillers("hello, um, world"),
            "hello, world"
        );
    }
}
