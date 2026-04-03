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
