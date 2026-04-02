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

    /// Return the first snippet whose trigger list contains an exact
    /// case-insensitive match for `text` (after trimming whitespace).
    pub fn find_match(&self, text: &str) -> Option<Snippet> {
        let norm = normalize(text);
        let list = self.snippets.lock().unwrap();
        list.iter()
            .find(|s| s.triggers.iter().any(|t| normalize(t) == norm))
            .cloned()
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

fn normalize(s: &str) -> String {
    s.trim().to_lowercase()
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
