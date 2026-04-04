use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static SNIPPETS: OnceLock<SnippetManager> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snippet {
    pub id: String,
    /// All trigger phrases that activate this snippet (case-insensitive).
    /// The first element is the canonical trigger; additional entries are
    /// accumulated via self-healing when the recognizer mishears the trigger.
    pub triggers: Vec<String>,
    pub body: String,
    pub created_at: String,
}

pub struct SnippetManager {
    snippets: Mutex<Vec<Snippet>>,
}

impl SnippetManager {
    pub fn init_global() -> &'static Self {
        SNIPPETS.get_or_init(|| {
            let snippets = Self::load_from_disk().unwrap_or_default();
            Self { snippets: Mutex::new(snippets) }
        })
    }

    pub fn global() -> &'static Self {
        SNIPPETS.get().expect("SnippetManager not initialized")
    }

    pub fn list(&self) -> Vec<Snippet> {
        self.snippets.lock().unwrap().clone()
    }

    /// Create a new snippet with a single trigger and return it.
    pub fn add(&self, trigger: String, body: String) -> Snippet {
        let snippet = Snippet {
            id: unique_id(),
            triggers: vec![trigger],
            body,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        self.snippets.lock().unwrap().push(snippet.clone());
        self.save();
        snippet
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        let mut list = self.snippets.lock().unwrap();
        let before = list.len();
        list.retain(|s| s.id != id);
        if list.len() == before {
            return Err(format!("snippet not found: {id}"));
        }
        drop(list);
        self.save();
        Ok(())
    }

    /// Update an existing snippet's triggers and body.
    pub fn update(&self, id: &str, triggers: Vec<String>, body: String) -> Result<Snippet, String> {
        let mut list = self.snippets.lock().unwrap();
        let snippet = list.iter_mut()
            .find(|s| s.id == id)
            .ok_or_else(|| format!("snippet not found: {id}"))?;
        snippet.triggers = triggers;
        snippet.body = body;
        let updated = snippet.clone();
        drop(list);
        self.save();
        Ok(updated)
    }

    /// Add a trigger phrase to an existing snippet (self-healing).
    /// No-ops if the normalised trigger already exists.
    pub fn add_trigger(&self, id: &str, trigger: String) -> Result<(), String> {
        let mut list = self.snippets.lock().unwrap();
        let snippet = list.iter_mut()
            .find(|s| s.id == id)
            .ok_or_else(|| format!("snippet not found: {id}"))?;
        let norm = normalize(&trigger);
        if !snippet.triggers.iter().any(|t| normalize(t) == norm) {
            snippet.triggers.push(trigger);
        }
        drop(list);
        self.save();
        Ok(())
    }

    /// Return the best-matching snippet for `text`.
    ///
    /// First tries an exact case-insensitive match across all triggers.
    /// If none is found, picks the snippet with the lowest normalised
    /// Levenshtein distance (distance / max_len) across its triggers,
    /// returning it only when that distance is ≤ [`FUZZY_THRESHOLD`].
    pub fn find_match(&self, text: &str) -> Option<Snippet> {
        let norm = normalize(text);
        let list = self.snippets.lock().unwrap();

        // Exact match first.
        if let Some(s) = list.iter().find(|s| s.triggers.iter().any(|t| normalize(t) == norm)) {
            log::info!("Snippet exact match: \"{}\" → id={}", norm, s.id);
            return Some(s.clone());
        }

        // Fuzzy fallback: lowest normalised distance below threshold.
        let best = list.iter().filter_map(|s| {
            let dist = s.triggers.iter()
                .map(|t| normalized_levenshtein(&normalize(t), &norm))
                .fold(f32::MAX, f32::min);
            if dist <= FUZZY_THRESHOLD { Some((s, dist)) } else { None }
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((s, dist)) = best {
            log::info!("Snippet fuzzy match: \"{}\" → id={} (dist={:.2})", norm, s.id, dist);
            Some(s.clone())
        } else {
            None
        }
    }

    fn data_path() -> Option<PathBuf> {
        #[cfg(target_os = "android")]
        {
            std::env::var_os("VERBA_DATA_DIR")
                .map(|d| PathBuf::from(d).join("snippets.json"))
        }
        #[cfg(not(target_os = "android"))]
        {
            dirs::config_dir().map(|d| d.join("verba").join("snippets.json"))
        }
    }

    fn load_from_disk() -> Option<Vec<Snippet>> {
        let path = Self::data_path()?;
        let data = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&data).ok()
    }

    fn save(&self) {
        let Some(path) = Self::data_path() else { return; };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let list = self.snippets.lock().unwrap();
        match serde_json::to_string_pretty(&*list) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    log::warn!("Failed to write snippets.json: {e}");
                }
            }
            Err(e) => log::warn!("Failed to serialize snippets: {e}"),
        }
    }
}

/// Maximum normalised Levenshtein distance (distance / max_len) to still
/// consider a trigger a match.  0.30 allows roughly one wrong word in a
/// three-word phrase while staying far enough from unrelated snippets.
const FUZZY_THRESHOLD: f32 = 0.30;

fn normalize(s: &str) -> String {
    s.trim().to_lowercase()
}

/// Levenshtein edit distance between two strings (character-level).
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1]
            } else {
                1 + prev[j - 1].min(prev[j]).min(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Levenshtein distance normalised by the length of the longer string,
/// giving a value in [0.0, 1.0] where 0.0 is identical.
fn normalized_levenshtein(a: &str, b: &str) -> f32 {
    let dist = levenshtein_distance(a, b);
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 { return 0.0; }
    dist as f32 / max_len as f32
}

fn unique_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{millis:x}{nanos:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager() -> SnippetManager {
        SnippetManager { snippets: Mutex::new(vec![]) }
    }

    fn make_snippet(id: &str, triggers: &[&str], body: &str) -> Snippet {
        Snippet {
            id: id.to_string(),
            triggers: triggers.iter().map(|t| t.to_string()).collect(),
            body: body.to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    // -- normalize --

    #[test]
    fn normalize_trims_and_lowercases() {
        assert_eq!(normalize("  Hello World  "), "hello world");
    }

    #[test]
    fn normalize_already_clean() {
        assert_eq!(normalize("hello"), "hello");
    }

    // -- levenshtein_distance --

    #[test]
    fn levenshtein_identical() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn levenshtein_one_substitution() {
        assert_eq!(levenshtein_distance("cat", "car"), 1);
    }

    #[test]
    fn levenshtein_one_insertion() {
        assert_eq!(levenshtein_distance("cat", "cats"), 1);
    }

    #[test]
    fn levenshtein_one_deletion() {
        assert_eq!(levenshtein_distance("cats", "cat"), 1);
    }

    #[test]
    fn levenshtein_empty_strings() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn levenshtein_completely_different() {
        assert_eq!(levenshtein_distance("abc", "xyz"), 3);
    }

    // -- normalized_levenshtein --

    #[test]
    fn normalized_levenshtein_identical_is_zero() {
        assert_eq!(normalized_levenshtein("hello", "hello"), 0.0);
    }

    #[test]
    fn normalized_levenshtein_empty_both_is_zero() {
        assert_eq!(normalized_levenshtein("", ""), 0.0);
    }

    #[test]
    fn normalized_levenshtein_completely_different_is_one() {
        // "abc" vs "xyz" = 3 edits / max(3,3) = 1.0
        assert!((normalized_levenshtein("abc", "xyz") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn normalized_levenshtein_one_edit() {
        // "cat" vs "car" = 1 edit / max(3,3) = 0.333
        let d = normalized_levenshtein("cat", "car");
        assert!((d - 1.0 / 3.0).abs() < 0.01);
    }

    // -- unique_id --

    #[test]
    fn unique_id_is_nonempty() {
        assert!(!unique_id().is_empty());
    }

    #[test]
    fn unique_ids_differ() {
        let a = unique_id();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let b = unique_id();
        assert_ne!(a, b);
    }

    // -- SnippetManager CRUD --

    #[test]
    fn add_and_list() {
        let mgr = test_manager();
        let s = mgr.add("hello world".into(), "greeting body".into());
        assert_eq!(s.triggers, vec!["hello world"]);
        assert_eq!(s.body, "greeting body");
        let list = mgr.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, s.id);
    }

    #[test]
    fn delete_existing() {
        let mgr = test_manager();
        let s = mgr.add("trigger".into(), "body".into());
        assert!(mgr.delete(&s.id).is_ok());
        assert!(mgr.list().is_empty());
    }

    #[test]
    fn delete_nonexistent_errors() {
        let mgr = test_manager();
        assert!(mgr.delete("nonexistent").is_err());
    }

    #[test]
    fn update_existing() {
        let mgr = test_manager();
        let s = mgr.add("old trigger".into(), "old body".into());
        let updated = mgr.update(&s.id, vec!["new trigger".into()], "new body".into()).unwrap();
        assert_eq!(updated.triggers, vec!["new trigger"]);
        assert_eq!(updated.body, "new body");
        assert_eq!(mgr.list()[0].body, "new body");
    }

    #[test]
    fn update_nonexistent_errors() {
        let mgr = test_manager();
        assert!(mgr.update("nope", vec![], "body".into()).is_err());
    }

    #[test]
    fn add_trigger_appends() {
        let mgr = test_manager();
        let s = mgr.add("hello".into(), "body".into());
        mgr.add_trigger(&s.id, "helo".into()).unwrap();
        let list = mgr.list();
        assert_eq!(list[0].triggers, vec!["hello", "helo"]);
    }

    #[test]
    fn add_trigger_deduplicates_normalized() {
        let mgr = test_manager();
        let s = mgr.add("hello".into(), "body".into());
        // Same trigger with different case should be a no-op
        mgr.add_trigger(&s.id, "Hello".into()).unwrap();
        assert_eq!(mgr.list()[0].triggers.len(), 1);
    }

    #[test]
    fn add_trigger_nonexistent_errors() {
        let mgr = test_manager();
        assert!(mgr.add_trigger("nope", "trigger".into()).is_err());
    }

    // -- find_match --

    #[test]
    fn find_exact_match_case_insensitive() {
        let mgr = test_manager();
        {
            let mut list = mgr.snippets.lock().unwrap();
            list.push(make_snippet("1", &["email template"], "Dear Sir"));
        }
        let found = mgr.find_match("Email Template");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "1");
    }

    #[test]
    fn find_exact_match_secondary_trigger() {
        let mgr = test_manager();
        {
            let mut list = mgr.snippets.lock().unwrap();
            list.push(make_snippet("1", &["email template", "female template"], "Dear Sir"));
        }
        let found = mgr.find_match("female template");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "1");
    }

    #[test]
    fn find_fuzzy_match_within_threshold() {
        let mgr = test_manager();
        {
            let mut list = mgr.snippets.lock().unwrap();
            list.push(make_snippet("1", &["email template"], "Dear Sir"));
        }
        // "emal template" is 1 edit from "email template" (14 chars) = 0.071
        let found = mgr.find_match("emal template");
        assert!(found.is_some(), "expected fuzzy match for 'emal template'");
    }

    #[test]
    fn find_no_match_beyond_threshold() {
        let mgr = test_manager();
        {
            let mut list = mgr.snippets.lock().unwrap();
            list.push(make_snippet("1", &["email template"], "Dear Sir"));
        }
        // Completely unrelated text
        let found = mgr.find_match("weather report");
        assert!(found.is_none());
    }

    #[test]
    fn find_prefers_exact_over_fuzzy() {
        let mgr = test_manager();
        {
            let mut list = mgr.snippets.lock().unwrap();
            list.push(make_snippet("fuzzy", &["helo world"], "fuzzy body"));
            list.push(make_snippet("exact", &["hello world"], "exact body"));
        }
        let found = mgr.find_match("hello world").unwrap();
        assert_eq!(found.id, "exact");
    }

    #[test]
    fn find_no_snippets_returns_none() {
        let mgr = test_manager();
        assert!(mgr.find_match("anything").is_none());
    }

    // -- Snippet serialization --

    #[test]
    fn snippet_serialization_roundtrip() {
        let s = make_snippet("abc", &["trigger one", "trigger two"], "body text");
        let json = serde_json::to_string(&s).unwrap();
        let deserialized: Snippet = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "abc");
        assert_eq!(deserialized.triggers, vec!["trigger one", "trigger two"]);
        assert_eq!(deserialized.body, "body text");
    }
}
