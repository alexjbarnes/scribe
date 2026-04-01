//! Post-processing pipeline for transcribed text.
//!
//! 1. Filler word removal (rule-based, ~1ms)
//! 2. Inverse text normalization (rule-based, ~5ms)
//! 3. Grammar correction (nlprule / LanguageTool, ~5-10ms)
//! 4. SymSpell spell correction (edit-distance, <1ms)

mod filler;
mod grammar;
mod itn;
mod spelling;

use std::sync::OnceLock;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// A snapshot of text after a pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    pub name: String,
    pub text: String,
    #[serde(default)]
    pub changed: bool,
    #[serde(default)]
    pub duration_ms: u64,
}

/// Result of the post-processing pipeline with intermediate stage snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub text: String,
    pub stages: Vec<PipelineStage>,
    #[serde(default)]
    pub total_ms: u64,
}

struct Pipeline {
    grammar: grammar::GrammarChecker,
    spelling: spelling::SpellCorrector,
}

static PIPELINE: OnceLock<Pipeline> = OnceLock::new();

fn get_pipeline() -> &'static Pipeline {
    PIPELINE.get_or_init(|| {
        let grammar = grammar::GrammarChecker::new()
            .expect("failed to load nlprule grammar checker");
        Pipeline {
            grammar,
            spelling: spelling::SpellCorrector::new(),
        }
    })
}

/// Force-initialize the pipeline (Harper dictionary + rules).
/// Call during startup so the first transcription isn't slow.
pub fn warm_up() {
    let t = Instant::now();
    let _ = get_pipeline();
    log::info!("Pipeline: warm-up took {}ms", t.elapsed().as_millis());
}

/// Run the full post-processing pipeline on transcribed text.
/// Returns the final text along with snapshots after each stage that made a change.
pub fn postprocess(text: &str) -> PipelineResult {
    if text.trim().is_empty() {
        return PipelineResult {
            text: String::new(),
            stages: Vec::new(),
            total_ms: 0,
        };
    }

    let total_start = Instant::now();
    let mut stages = Vec::new();
    let mut s = text.to_string();

    stages.push(PipelineStage {
        name: "Raw transcription".to_string(),
        text: s.clone(),
        changed: false,
        duration_ms: 0,
    });

    // Stage 1: Filler word removal
    let t = Instant::now();
    let prev = s.clone();
    s = filler::remove_fillers(&s);
    let changed = s != prev;
    let ms = t.elapsed().as_millis() as u64;
    log::debug!("Pipeline stage 1 (fillers): {ms}ms changed={changed}");
    stages.push(PipelineStage {
        name: "Filler removal".to_string(),
        text: s.clone(),
        changed,
        duration_ms: ms,
    });

    // Stage 2: Inverse text normalization
    let t = Instant::now();
    let prev = s.clone();
    s = itn::normalize(&s);
    let changed = s != prev;
    let ms = t.elapsed().as_millis() as u64;
    log::debug!("Pipeline stage 2 (ITN): {ms}ms changed={changed}");
    stages.push(PipelineStage {
        name: "ITN".to_string(),
        text: s.clone(),
        changed,
        duration_ms: ms,
    });

    // Stage 3: Grammar correction (nlprule / LanguageTool rules)
    let pipeline = get_pipeline();
    let t = Instant::now();
    let prev = s.clone();
    s = pipeline.grammar.correct(&s);
    let changed = s != prev;
    let ms = t.elapsed().as_millis() as u64;
    log::debug!("Pipeline stage 3 (Grammar): {ms}ms changed={changed}");
    stages.push(PipelineStage {
        name: "Grammar".to_string(),
        text: s.clone(),
        changed,
        duration_ms: ms,
    });

    // Stage 4: Spell correction
    let t = Instant::now();
    let prev = s.clone();
    s = pipeline.spelling.correct(&s);
    let changed = s != prev;
    let ms = t.elapsed().as_millis() as u64;
    log::debug!("Pipeline stage 4 (Spelling): {ms}ms changed={changed}");
    stages.push(PipelineStage {
        name: "Spelling".to_string(),
        text: s.clone(),
        changed,
        duration_ms: ms,
    });

    // Final cleanup
    let t = Instant::now();
    let prev = s.clone();
    s = final_cleanup(&s);
    let changed = s != prev;
    let ms = t.elapsed().as_millis() as u64;
    stages.push(PipelineStage {
        name: "Cleanup".to_string(),
        text: s.clone(),
        changed,
        duration_ms: ms,
    });

    let total_ms = total_start.elapsed().as_millis() as u64;
    log::info!("Pipeline total: {total_ms}ms ({} stages changed)", stages.iter().filter(|s| s.changed).count());
    PipelineResult { text: s, stages, total_ms }
}

/// Join multiple transcription chunks, fixing false sentence boundaries.
///
/// When VAD splits audio mid-sentence, each chunk gets independent punctuation
/// from the transcriber. "if it's one." + "chunk of text" should join as
/// "if it's one chunk of text", not "if it's one. chunk of text".
///
/// Rule: strip trailing period from a chunk if the next chunk starts with a
/// lowercase letter (suggesting the sentence continues).
pub fn join_chunks(chunks: &[&str]) -> String {
    if chunks.is_empty() {
        return String::new();
    }
    if chunks.len() == 1 {
        return chunks[0].to_string();
    }

    let mut parts: Vec<String> = Vec::with_capacity(chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            continue;
        }

        if i + 1 < chunks.len() {
            let next_first = chunks[i + 1].trim().chars().next();
            if let Some(c) = next_first {
                if c.is_ascii_lowercase() && trimmed.ends_with('.') {
                    // Next chunk continues the sentence -- strip the false period
                    parts.push(trimmed.trim_end_matches('.').to_string());
                    continue;
                }
            }
        }

        parts.push(trimmed.to_string());
    }

    parts.join(" ")
}

/// Final pass: ensure capitalization and trailing punctuation.
fn final_cleanup(text: &str) -> String {
    let mut s = text.trim().to_string();
    if s.is_empty() {
        return s;
    }

    // Collapse multiple spaces
    while s.contains("  ") {
        s = s.replace("  ", " ");
    }

    // Fix false mid-sentence capitalization from the model.
    // Lowercase words that are capitalized but don't follow sentence-ending
    // punctuation and aren't acronyms or "I".
    s = fix_mid_sentence_caps(&s);

    // Capitalize first character
    let mut chars = s.chars();
    if let Some(first) = chars.next() {
        if first.is_ascii_lowercase() {
            s = first.to_ascii_uppercase().to_string() + chars.as_str();
        }
    }

    // Add trailing period if no sentence-ending punctuation
    let last = s.chars().last().unwrap_or(' ');
    if !matches!(last, '.' | '!' | '?' | ',' | ';' | ':' | '"' | '\'' | ')') {
        s.push('.');
    }

    s
}

/// Lowercase words that the model falsely capitalized mid-sentence.
/// Preserves capitalization after sentence-ending punctuation, for acronyms
/// (all caps, 2+ chars), and for "I".
fn fix_mid_sentence_caps(text: &str) -> String {
    let words: Vec<&str> = text.split(' ').collect();
    if words.len() <= 1 {
        return text.to_string();
    }

    let mut result = Vec::with_capacity(words.len());
    result.push(words[0].to_string());

    for i in 1..words.len() {
        let word = words[i];
        let prev = words[i - 1];

        // Check if previous word ends with sentence-ending punctuation
        let after_sentence_end = prev.ends_with('.') || prev.ends_with('!') || prev.ends_with('?');

        if after_sentence_end {
            result.push(word.to_string());
            continue;
        }

        // Strip leading punctuation to examine the core word
        let core_start = word.find(|c: char| !c.is_ascii_punctuation()).unwrap_or(word.len());
        let prefix = &word[..core_start];
        let rest = &word[core_start..];

        if rest.is_empty() {
            result.push(word.to_string());
            continue;
        }

        let first_char = rest.chars().next().unwrap();
        if !first_char.is_uppercase() {
            result.push(word.to_string());
            continue;
        }

        // Find where the alphabetic part ends (core word without trailing punct)
        let alpha_end = rest.find(|c: char| !c.is_alphabetic() && c != '\'').unwrap_or(rest.len());
        let core = &rest[..alpha_end];

        // Preserve "I" standalone
        if core == "I" {
            result.push(word.to_string());
            continue;
        }

        // Preserve acronyms (all uppercase, 2+ chars): API, DNS, USA
        if core.len() >= 2 && core.chars().all(|c| c.is_uppercase()) {
            result.push(word.to_string());
            continue;
        }

        // Lowercase the first character
        let lowered_first: String = first_char.to_lowercase().collect();
        let fixed = format!("{prefix}{lowered_first}{}", &rest[first_char.len_utf8()..]);
        result.push(fixed);
    }

    result.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postprocess_empty() {
        assert_eq!(postprocess("").text, "");
        assert_eq!(postprocess("  ").text, "");
    }

    #[test]
    fn postprocess_basic_text() {
        let result = postprocess("hello world");
        assert!(result.text.starts_with('H'));
        assert!(result.text.ends_with('.'));
        assert!(!result.stages.is_empty());
        assert_eq!(result.stages[0].name, "Raw transcription");
    }

    #[test]
    fn postprocess_removes_fillers() {
        let result = postprocess("um hello uh world");
        assert!(!result.text.contains(" um "));
        assert!(!result.text.contains(" uh "));
    }

    #[test]
    fn final_cleanup_capitalizes() {
        assert_eq!(final_cleanup("hello world"), "Hello world.");
    }

    #[test]
    fn final_cleanup_preserves_punctuation() {
        assert_eq!(final_cleanup("Hello world!"), "Hello world!");
    }

    #[test]
    fn final_cleanup_collapses_spaces() {
        assert_eq!(final_cleanup("hello  world"), "Hello world.");
    }

    #[test]
    fn postprocess_counting_not_summed() {
        let result = postprocess("one, two, three, four, five");
        assert!(
            result.text.contains("1, 2, 3, 4, 5"),
            "Expected individual numbers, got: {}",
            result.text
        );
    }

    #[test]
    fn postprocess_preserves_trailing_punct() {
        let result = postprocess("I have three.");
        assert_eq!(result.text, "I have 3.");
    }

    #[test]
    fn join_chunks_strips_false_boundary() {
        // "one." + "chunk of text" -- next starts lowercase, strip period
        let chunks = vec!["if it's one.", "chunk of text or speech."];
        assert_eq!(join_chunks(&chunks), "if it's one chunk of text or speech.");
    }

    #[test]
    fn join_chunks_keeps_real_boundary() {
        // "one." + "Chunk" -- next starts uppercase, keep period
        let chunks = vec!["That was number one.", "Chunk two is next."];
        assert_eq!(join_chunks(&chunks), "That was number one. Chunk two is next.");
    }

    #[test]
    fn join_chunks_single() {
        assert_eq!(join_chunks(&vec!["hello world"]), "hello world");
    }

    #[test]
    fn join_chunks_empty() {
        let empty: Vec<&str> = vec![];
        assert_eq!(join_chunks(&empty), "");
    }

    #[test]
    fn join_chunks_three_chunks_mixed() {
        let chunks = vec!["First sentence.", "but continues here.", "New sentence."];
        assert_eq!(join_chunks(&chunks), "First sentence but continues here. New sentence.");
    }

    #[test]
    fn fix_caps_lowercases_mid_sentence() {
        assert_eq!(fix_mid_sentence_caps("assert what We want"), "assert what we want");
    }

    #[test]
    fn fix_caps_preserves_after_period() {
        assert_eq!(fix_mid_sentence_caps("Hello. World is great."), "Hello. World is great.");
    }

    #[test]
    fn fix_caps_preserves_after_exclamation() {
        assert_eq!(fix_mid_sentence_caps("Stop! That hurts."), "Stop! That hurts.");
    }

    #[test]
    fn fix_caps_preserves_after_question() {
        assert_eq!(fix_mid_sentence_caps("Really? That works."), "Really? That works.");
    }

    #[test]
    fn fix_caps_preserves_acronyms() {
        assert_eq!(fix_mid_sentence_caps("the API is working"), "the API is working");
        assert_eq!(fix_mid_sentence_caps("use DNS for that"), "use DNS for that");
    }

    #[test]
    fn fix_caps_preserves_i() {
        assert_eq!(fix_mid_sentence_caps("then I went home"), "then I went home");
    }

    #[test]
    fn fix_caps_preserves_first_word() {
        assert_eq!(fix_mid_sentence_caps("Hello world"), "Hello world");
    }

    #[test]
    fn fix_caps_multiple_false_caps() {
        assert_eq!(
            fix_mid_sentence_caps("this Is Really Bad"),
            "this is really bad"
        );
    }

    #[test]
    fn fix_caps_mixed_real_and_false() {
        assert_eq!(
            fix_mid_sentence_caps("That works. And Then We left."),
            "That works. And then we left."
        );
    }

    #[test]
    fn final_cleanup_fixes_mid_sentence_caps() {
        assert_eq!(
            final_cleanup("assert what We want out the other side"),
            "Assert what we want out the other side."
        );
    }
}
