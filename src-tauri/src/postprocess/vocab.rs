use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabEntry {
    pub from: String,
    pub to: String,
}

static VOCAB: OnceLock<RwLock<Vec<VocabEntry>>> = OnceLock::new();

fn vocab_path() -> Option<PathBuf> {
    #[cfg(target_os = "android")]
    {
        std::env::var_os("VERBA_DATA_DIR")
            .map(|d| PathBuf::from(d).join("vocab.json"))
    }
    #[cfg(not(target_os = "android"))]
    {
        dirs::data_local_dir().map(|d| d.join("verba").join("vocab.json"))
    }
}

fn global() -> &'static RwLock<Vec<VocabEntry>> {
    VOCAB.get_or_init(|| RwLock::new(load_from_disk()))
}

fn load_from_disk() -> Vec<VocabEntry> {
    let Some(path) = vocab_path() else {
        return Vec::new();
    };
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            log::warn!("Bad vocab.json, ignoring: {e}");
            Vec::new()
        }),
        Err(_) => Vec::new(),
    }
}

fn save_to_disk(entries: &[VocabEntry]) {
    let Some(path) = vocab_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(entries) {
        Ok(s) => {
            if let Err(e) = std::fs::write(&path, s) {
                log::warn!("Failed to save vocab.json: {e}");
            }
        }
        Err(e) => log::warn!("Failed to serialize vocab: {e}"),
    }
}

pub fn get_entries() -> Vec<VocabEntry> {
    global().read().unwrap().clone()
}

pub fn add_entry(from: String, to: String) -> Result<(), String> {
    if from.trim().is_empty() {
        return Err("'from' cannot be empty".into());
    }
    let key = from.trim().to_lowercase();
    let entry = VocabEntry { from: key, to: to.trim().to_string() };
    let mut entries = global().write().unwrap();
    if let Some(pos) = entries.iter().position(|e| e.from == entry.from) {
        entries[pos] = entry;
    } else {
        entries.push(entry);
    }
    save_to_disk(&entries);
    Ok(())
}

pub fn remove_entry(from: &str) -> Result<(), String> {
    let key = from.trim().to_lowercase();
    let mut entries = global().write().unwrap();
    let before = entries.len();
    entries.retain(|e| e.from != key);
    if entries.len() == before {
        return Err(format!("no entry for '{from}'"));
    }
    save_to_disk(&entries);
    Ok(())
}

/// Built-in informal speech normalizations applied to everyone.
/// Ordered longest-match first where phrases share a prefix.
const BUILTIN_SUBS: &[(&str, &str)] = &[
    ("shoulda", "should have"),
    ("coulda", "could have"),
    ("woulda", "would have"),
    ("mighta", "might have"),
    ("hafta", "have to"),
    ("oughta", "ought to"),
    ("gonna", "going to"),
    ("wanna", "want to"),
    ("tryna", "trying to"),
    ("gotta", "got to"),
    ("lemme", "let me"),
    ("gimme", "give me"),
    ("kinda", "kind of"),
    ("sorta", "sort of"),
    ("dunno", "don't know"),
];

/// Apply all vocab substitutions to `text`.
/// Runs built-in informal normalizations first, then user entries.
/// Matches are case-insensitive with word-boundary enforcement.
pub fn apply(text: &str) -> String {
    let mut result = text.to_string();
    for (from, to) in BUILTIN_SUBS {
        result = apply_one(&result, from, to);
    }
    let entries = global().read().unwrap();
    for entry in entries.iter() {
        result = apply_one(&result, &entry.from, &entry.to);
    }
    result
}

fn apply_one(text: &str, from: &str, to: &str) -> String {
    let from_lower = from.to_lowercase();
    let text_lower = text.to_lowercase();
    let from_bytes = from_lower.len();

    let mut result = String::with_capacity(text.len());
    let mut last = 0usize;
    let mut search_start = 0usize;

    loop {
        let Some(rel) = text_lower[search_start..].find(&*from_lower) else {
            break;
        };
        let abs = search_start + rel;
        let end = abs + from_bytes;

        let before_ok = abs == 0 || {
            let ch = text[..abs].chars().next_back().unwrap();
            !ch.is_alphanumeric() && ch != '\''
        };
        let after_ok = end >= text.len() || {
            let ch = text[end..].chars().next().unwrap();
            !ch.is_alphanumeric() && ch != '\''
        };

        if before_ok && after_ok {
            result.push_str(&text[last..abs]);
            result.push_str(to);
            last = end;
            search_start = end;
        } else {
            // Advance past this byte position by one char to avoid looping.
            search_start = abs + text[abs..].chars().next().map_or(1, |c| c.len_utf8());
        }
    }
    result.push_str(&text[last..]);
    result
}

#[cfg(test)]
mod tests {
    use super::{apply, apply_one};

    #[test]
    fn replaces_whole_word() {
        assert_eq!(apply_one("going to maine today", "maine", "main"), "going to main today");
    }

    #[test]
    fn case_insensitive_match() {
        assert_eq!(apply_one("going to Maine today", "maine", "main"), "going to main today");
    }

    #[test]
    fn no_partial_word_match() {
        assert_eq!(apply_one("mainelander", "maine", "main"), "mainelander");
    }

    #[test]
    fn multi_word_phrase() {
        assert_eq!(apply_one("use git hub actions", "git hub", "GitHub"), "use GitHub actions");
    }

    #[test]
    fn multiple_occurrences() {
        assert_eq!(apply_one("maine and maine again", "maine", "main"), "main and main again");
    }

    #[test]
    fn at_start_of_string() {
        assert_eq!(apply_one("maine is great", "maine", "main"), "main is great");
    }

    #[test]
    fn at_end_of_string() {
        assert_eq!(apply_one("going to maine", "maine", "main"), "going to main");
    }

    #[test]
    fn no_match_leaves_unchanged() {
        assert_eq!(apply_one("hello world", "maine", "main"), "hello world");
    }

    #[test]
    fn empty_text() {
        assert_eq!(apply_one("", "maine", "main"), "");
    }

    #[test]
    fn one_to_many_words() {
        assert_eq!(apply_one("I'm gonna go", "gonna", "going to"), "I'm going to go");
        assert_eq!(apply_one("I wanna leave", "wanna", "want to"), "I want to leave");
    }

    #[test]
    fn builtin_informal_contractions() {
        assert_eq!(apply("I'm gonna go"), "I'm going to go");
        assert_eq!(apply("I wanna leave"), "I want to leave");
        assert_eq!(apply("I shoulda done that"), "I should have done that");
        assert_eq!(apply("I coulda been a contender"), "I could have been a contender");
        assert_eq!(apply("lemme see it"), "let me see it");
        assert_eq!(apply("gimme a break"), "give me a break");
        assert_eq!(apply("I dunno what happened"), "I don't know what happened");
        assert_eq!(apply("tryna figure it out"), "trying to figure it out");
        assert_eq!(apply("kinda tired"), "kind of tired");
    }
}
