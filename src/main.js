const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

// ── Tab switching ──

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('border-primary', 'text-primary');
      b.classList.add('border-transparent', 'text-on-surface-variant');
    });
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));

    btn.classList.remove('border-transparent', 'text-on-surface-variant');
    btn.classList.add('border-primary', 'text-primary');
    document.getElementById(btn.dataset.tab).classList.remove('hidden');

    // Refresh data when switching to relevant tabs
    if (btn.dataset.tab === 'tab-history') loadHistory();
    if (btn.dataset.tab === 'tab-models') loadModels();
  });
});

// ── History tab ──

const historyList = document.getElementById('history-list');
const historyPlaceholder = document.getElementById('history-placeholder');

function formatDuration(ms) {
  if (ms < 1000) return ms + 'ms';
  return (ms / 1000).toFixed(1) + 's';
}

function formatTimestamp(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
      ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  } catch (_) {
    return iso;
  }
}

function formatSpeed(entry) {
  if (!entry.audio_duration_ms || !entry.duration_ms) return '';
  const rtf = entry.duration_ms / entry.audio_duration_ms;
  return rtf.toFixed(2) + 'x RTF';
}

function formatAudioDuration(ms) {
  if (!ms) return '';
  const secs = ms / 1000;
  if (secs < 60) return secs.toFixed(1) + 's spoken';
  const mins = Math.floor(secs / 60);
  const rem = (secs % 60).toFixed(0);
  return mins + 'm ' + rem + 's spoken';
}

function renderChunkTimings(chunks) {
  if (!chunks || chunks.length === 0) return '';
  let html = '<div class="mt-2">';
  html += '<span class="text-[10px] font-semibold uppercase tracking-wider text-primary/70">Transcription chunks</span>';
  html += '<div class="mt-1 space-y-0.5">';
  let total = 0;
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    total += c.transcribe_ms;
    const audioSec = (c.audio_ms / 1000).toFixed(1);
    html += `<p class="text-xs text-on-surface-variant font-mono">${c.transcribe_ms}ms (${audioSec}s audio)</p>`;
  }
  if (chunks.length > 1) {
    html += `<p class="text-xs text-on-surface-variant font-mono font-semibold">Total: ${total}ms</p>`;
  }
  html += '</div></div>';
  return html;
}

function renderPipelineStages(stages, chunkTimings) {
  const hasStages = stages && stages.length > 1;
  const hasChunks = chunkTimings && chunkTimings.length > 0;
  if (!hasStages && !hasChunks) return '';

  let html = '<div class="pipeline-stages hidden mt-3 pt-3 border-t border-outline-variant/20 space-y-2">';
  if (hasChunks) {
    html += renderChunkTimings(chunkTimings);
  }
  if (hasStages) {
    for (const stage of stages) {
      const dim = stage.changed === false ? ' opacity-40' : '';
      const tag = stage.changed === false ? ' (no change)' : '';
      const timing = stage.duration_ms ? ` ${stage.duration_ms}ms` : '';
      html += `
        <div class="${dim}">
          <span class="text-[10px] font-semibold uppercase tracking-wider text-primary/70">${escapeHtml(stage.name)}${tag}${timing}</span>
          <p class="text-xs text-on-surface-variant leading-relaxed mt-0.5 select-text">${escapeHtml(stage.text)}</p>
        </div>`;
    }
  }
  html += '</div>';
  return html;
}

function formatEntryForCopy(entry) {
  const lines = [entry.text, ''];

  const stats = [];
  if (entry.timestamp) stats.push(formatTimestamp(entry.timestamp));
  if (entry.audio_duration_ms) stats.push(formatAudioDuration(entry.audio_duration_ms));
  if (entry.duration_ms) stats.push(formatDuration(entry.duration_ms) + ' to transcribe');
  if (entry.postprocess_ms) stats.push(entry.postprocess_ms + 'ms postprocess');
  const speed = formatSpeed(entry);
  if (speed) stats.push(speed);
  if (entry.filtered_segments) stats.push(entry.filtered_segments + ' filtered (' + (entry.filtered_audio_ms / 1000).toFixed(1) + 's)');
  if (entry.model_id) stats.push(entry.model_id);
  lines.push(stats.join(' | '));

  if (entry.chunk_timings && entry.chunk_timings.length > 0) {
    lines.push('', 'Chunks:');
    let total = 0;
    for (const c of entry.chunk_timings) {
      total += c.transcribe_ms;
      lines.push('  ' + c.transcribe_ms + 'ms (' + (c.audio_ms / 1000).toFixed(1) + 's audio)');
    }
    if (entry.chunk_timings.length > 1) {
      lines.push('  Total: ' + total + 'ms');
    }
  }

  if (entry.pipeline_stages && entry.pipeline_stages.length > 1) {
    const totalLabel = entry.postprocess_ms ? ` (${entry.postprocess_ms}ms total)` : '';
    lines.push('', 'Pipeline' + totalLabel + ':');
    for (const stage of entry.pipeline_stages) {
      const tag = stage.changed === false ? ' (no change)' : '';
      const timing = stage.duration_ms ? ` [${stage.duration_ms}ms]` : '';
      lines.push('  ' + stage.name + tag + timing + ': ' + stage.text);
    }
  }

  return lines.join('\n');
}

function renderHistory(entries) {
  historyList.innerHTML = '';
  if (!entries || entries.length === 0) {
    historyList.innerHTML = `
      <div class="flex flex-col items-center justify-center pt-16 text-on-surface-variant">
        <span class="material-symbols-outlined text-4xl mb-3 opacity-30">history</span>
        <p class="text-sm">No transcriptions yet</p>
      </div>`;
    return;
  }
  for (const entry of [...entries].reverse()) {
    const card = document.createElement('div');
    card.className = 'bg-surface-container-low rounded-xl border border-outline-variant/20 p-4';

    const hasStages = entry.pipeline_stages && entry.pipeline_stages.length > 1;
    const hasChunks = entry.chunk_timings && entry.chunk_timings.length > 0;
    const hasDetails = hasStages || hasChunks;

    const stats = [
      formatTimestamp(entry.timestamp),
      formatAudioDuration(entry.audio_duration_ms),
      formatDuration(entry.duration_ms) + ' to transcribe',
      entry.postprocess_ms ? entry.postprocess_ms + 'ms postprocess' : null,
      formatSpeed(entry),
      entry.filtered_segments ? entry.filtered_segments + ' filtered (' + (entry.filtered_audio_ms / 1000).toFixed(1) + 's)' : null,
      escapeHtml(entry.model_id),
    ].filter(Boolean);

    const toggleBtn = hasDetails
      ? '<button class="pipeline-toggle text-[10px] font-semibold text-primary/70 hover:text-primary transition-colors cursor-pointer">Details</button>'
      : '';

    card.innerHTML = `
      <p class="text-sm text-on-surface leading-relaxed mb-2 select-text">${escapeHtml(entry.text)}</p>
      <div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-on-surface-variant">
        ${stats.map(s => '<span>' + s + '</span>').join('')}
        ${toggleBtn}
        <button class="copy-entry-btn text-[10px] font-semibold text-primary/70 hover:text-primary transition-colors cursor-pointer">Copy</button>
      </div>
      ${renderPipelineStages(entry.pipeline_stages, entry.chunk_timings)}`;

    if (hasDetails) {
      card.querySelector('.pipeline-toggle').addEventListener('click', (e) => {
        const stagesEl = card.querySelector('.pipeline-stages');
        stagesEl.classList.toggle('hidden');
        e.target.textContent = stagesEl.classList.contains('hidden') ? 'Details' : 'Hide';
      });
    }

    card.querySelector('.copy-entry-btn').addEventListener('click', (e) => {
      const text = formatEntryForCopy(entry);
      navigator.clipboard.writeText(text).then(() => {
        e.target.textContent = 'Copied!';
        setTimeout(() => { e.target.textContent = 'Copy'; }, 1500);
      });
    });

    historyList.appendChild(card);
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function showToast(msg) {
  const el = document.createElement('div');
  el.textContent = msg;
  el.className = 'fixed bottom-6 left-1/2 -translate-x-1/2 z-50 bg-surface-container-highest text-on-surface text-sm px-5 py-3 rounded-xl shadow-lg border border-outline-variant/20 transition-opacity duration-300';
  document.body.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; }, 2500);
  setTimeout(() => { el.remove(); }, 2800);
}

async function loadHistory() {
  try {
    const entries = await invoke('list_history');
    renderHistory(entries);
  } catch (err) {
    console.error('Failed to load history:', err);
  }
}

// Auto-refresh history when the app comes back to foreground
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    loadHistory();
  }
});

// Refresh history when a transcription completes (in-app dictation path)
listen('transcription-result', () => {
  loadHistory();
});

document.getElementById('clear-history').addEventListener('click', async () => {
  if (!confirm('Clear all history?')) return;
  try {
    await invoke('clear_history');
    renderHistory([]);
  } catch (err) {
    console.error('Failed to clear history:', err);
  }
});

// ── Models tab ──

function renderModelRow(model) {
  const isActive = model.status === 'active';
  const isDownloaded = model.status === 'downloaded';
  const isDownloading = model.status === 'downloading';

  const deleteBtn = `<button class="del-btn text-on-surface-variant hover:text-error transition-colors cursor-pointer p-1.5 rounded-lg hover:bg-error/10" data-id="${model.id}" title="Delete"><span class="material-symbols-outlined text-base">delete</span></button>`;

  let actionHtml;
  if (isActive) {
    actionHtml = `<span class="text-xs text-primary font-semibold px-3 py-1.5 bg-primary/10 rounded-lg">Active</span>${deleteBtn}`;
  } else if (isDownloaded) {
    actionHtml = `<button class="use-btn text-xs font-semibold px-3 py-1.5 bg-primary text-on-primary rounded-lg hover:brightness-110 transition-all cursor-pointer" data-id="${model.id}">Use</button>${deleteBtn}`;
  } else if (isDownloading) {
    const pct = Math.round(model.progress * 100);
    actionHtml = `
      <div class="w-28" id="progress-${model.id}">
        <div class="progress-bar"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
        <span class="text-[10px] text-on-surface-variant mt-1 block text-right">${pct}%</span>
      </div>`;
  } else {
    actionHtml = `<button class="dl-btn text-xs font-semibold px-3 py-1.5 border border-outline-variant/30 text-on-surface rounded-lg hover:bg-surface-container-highest transition-colors cursor-pointer" data-id="${model.id}">Download</button>`;
  }

  return `
    <div class="flex items-center justify-between px-4 py-3" data-model-id="${model.id}">
      <div class="flex items-center gap-3 min-w-0 flex-1 mr-4">
        <span class="material-symbols-outlined text-lg ${isActive ? 'text-primary' : 'text-on-surface-variant'}">layers</span>
        <div class="min-w-0">
          <div class="text-sm font-medium ${isActive ? 'text-primary' : 'text-on-surface'}">${model.name}</div>
          <div class="text-xs text-on-surface-variant mt-0.5">${model.desc}</div>
        </div>
      </div>
      <div class="flex items-center gap-3 shrink-0">
        <span class="text-xs font-mono text-on-surface-variant">${model.size}</span>
        ${actionHtml}
      </div>
    </div>`;
}

async function loadModels() {
  const models = await invoke('list_models');

  document.getElementById('whisper-models').innerHTML = models
    .filter(m => m.engine === 'whisper')
    .map(renderModelRow)
    .join('');

  document.getElementById('parakeet-models').innerHTML = models
    .filter(m => m.engine === 'parakeet')
    .map(renderModelRow)
    .join('');

  // Attach download button handlers
  document.querySelectorAll('.dl-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.id;
      btn.disabled = true;
      btn.textContent = 'Starting...';
      btn.classList.add('opacity-50');

      try {
        await invoke('download_model', { id });
      } catch (err) {
        console.error('Download failed:', err);
        alert(`Download failed: ${err}`);
      }

      // Refresh model list after download completes or fails
      await loadModels();
    });
  });

  // Attach use button handlers
  document.querySelectorAll('.use-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.id;
      try {
        await invoke('switch_model', { id });
        showToast('Switching model...');
      } catch (err) {
        console.error('Switch failed:', err);
        alert(`Failed to switch model: ${err}`);
      }
      await loadModels();
    });
  });

  // Attach delete button handlers
  document.querySelectorAll('.del-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.id;
      if (!confirm(`Delete model ${id}?`)) return;
      try {
        await invoke('delete_model', { id });
      } catch (err) {
        console.error('Delete failed:', err);
        alert(`Failed to delete model: ${err}`);
      }
      await loadModels();
    });
  });
}

// ── Download progress events ──

listen('download-progress', (event) => {
  const { id, progress } = event.payload;
  const pct = Math.round(progress * 100);

  const container = document.querySelector(`[data-model-id="${id}"]`);
  if (!container) return;

  // Replace button with progress bar if not already showing
  let progressEl = document.getElementById(`progress-${id}`);
  if (!progressEl) {
    const actionArea = container.querySelector('.dl-btn, .use-btn');
    if (actionArea) {
      const wrapper = document.createElement('div');
      wrapper.className = 'w-28';
      wrapper.id = `progress-${id}`;
      wrapper.innerHTML = `
        <div class="progress-bar"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
        <span class="text-[10px] text-on-surface-variant mt-1 block text-right">${pct}%</span>`;
      actionArea.replaceWith(wrapper);
      progressEl = wrapper;
    }
  } else {
    const fill = progressEl.querySelector('.progress-bar-fill');
    const label = progressEl.querySelector('span');
    if (fill) fill.style.width = `${pct}%`;
    if (label) label.textContent = `${pct}%`;
  }
});

listen('download-complete', async () => {
  await loadModels();
});

listen('model-loaded', (event) => {
  const id = event.payload?.id || '';
  showToast(`Model ready: ${id}`);
});

// ── Audio tab ──

async function loadAudioDevices() {
  const devices = await invoke('list_audio_devices');
  const sel = document.getElementById('audio-device');
  // Keep "System Default" as first option, add real devices
  sel.innerHTML = '<option value="-1">System Default</option>';
  for (const dev of devices) {
    const opt = document.createElement('option');
    opt.value = dev.index;
    opt.textContent = dev.name;
    sel.appendChild(opt);
  }
}

// ── General tab ──

async function loadConfig() {
  const cfg = await invoke('get_config');
  document.getElementById('cfg-language').value = cfg.language;
  document.getElementById('cfg-threads').value = cfg.threads;
  document.getElementById('cfg-output-dir').value = cfg.output_dir;
  document.getElementById('cfg-haptic').checked = cfg.haptic_feedback;
  // Restore audio device selection
  document.getElementById('audio-device').value = cfg.device_index;
}

async function saveConfig() {
  const cfg = {
    language: document.getElementById('cfg-language').value,
    threads: parseInt(document.getElementById('cfg-threads').value, 10),
    output_dir: document.getElementById('cfg-output-dir').value,
    device_index: parseInt(document.getElementById('audio-device').value, 10),
    active_engine: 'whisper',
    active_model_id: '',
    haptic_feedback: document.getElementById('cfg-haptic').checked,
  };
  try {
    await invoke('save_config', { cfg });
    const btn = document.getElementById('save-config');
    btn.textContent = 'Saved!';
    setTimeout(() => { btn.textContent = 'Save Changes'; }, 1500);
  } catch (err) {
    console.error('Save failed:', err);
    alert(`Save failed: ${err}`);
  }
}

// ── Voice Enrollment ──

const enrollBtn = document.getElementById('enroll-btn');
const clearEnrollBtn = document.getElementById('clear-enroll-btn');
const enrollmentLabel = document.getElementById('enrollment-label');
const enrollmentRecording = document.getElementById('enrollment-recording');

async function loadEnrollmentStatus() {
  try {
    const enrolled = await invoke('get_speaker_enrollment_status');
    if (enrolled) {
      enrollmentLabel.textContent = 'Enrolled. Only your voice will be transcribed.';
      clearEnrollBtn.classList.remove('hidden');
    } else {
      enrollmentLabel.textContent = 'Not enrolled. All voices will be transcribed.';
      clearEnrollBtn.classList.add('hidden');
    }
  } catch (_) {
    // Command may not exist on mobile
  }
}

enrollBtn.addEventListener('click', async () => {
  enrollBtn.disabled = true;
  enrollBtn.textContent = 'Recording...';
  enrollBtn.classList.add('opacity-50');
  enrollmentRecording.classList.remove('hidden');
  enrollmentRecording.scrollIntoView({ behavior: 'smooth', block: 'center' });

  // Let the browser paint the recording UI before the invoke blocks
  await new Promise(r => setTimeout(r, 100));

  try {
    await invoke('enroll_speaker');
    enrollmentLabel.textContent = 'Enrolled. Only your voice will be transcribed.';
    clearEnrollBtn.classList.remove('hidden');
    showToast('Voice enrolled successfully');
  } catch (err) {
    showToast('Enrollment failed: ' + err);
  }

  enrollmentRecording.classList.add('hidden');
  enrollBtn.disabled = false;
  enrollBtn.textContent = 'Enroll Voice';
  enrollBtn.classList.remove('opacity-50');
});

clearEnrollBtn.addEventListener('click', async () => {
  try {
    await invoke('clear_speaker_enrollment');
    enrollmentLabel.textContent = 'Not enrolled. All voices will be transcribed.';
    clearEnrollBtn.classList.add('hidden');
    showToast('Voice enrollment cleared');
  } catch (err) {
    showToast('Failed to clear enrollment: ' + err);
  }
});

// ── Debug tab ──

const logOutput = document.getElementById('log-output');

listen('log-message', (event) => {
  const { level, line } = event.payload;
  const el = document.createElement('div');
  el.textContent = line;
  if (level === 'ERROR') el.className = 'text-red-400';
  else if (level === 'WARN') el.className = 'text-yellow-400';
  else if (level === 'DEBUG') el.className = 'text-gray-500';
  else el.className = 'text-gray-300';
  logOutput.appendChild(el);
  logOutput.scrollTop = logOutput.scrollHeight;
});

document.getElementById('clear-logs').addEventListener('click', () => {
  logOutput.innerHTML = '';
});

document.getElementById('copy-logs').addEventListener('click', () => {
  const text = logOutput.innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copy-logs');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  });
});

// ── Init ──

document.addEventListener('DOMContentLoaded', async () => {
  await loadHistory();
  await loadModels();
  await loadAudioDevices();
  await loadConfig();
  await loadEnrollmentStatus();

  document.getElementById('save-config').addEventListener('click', saveConfig);
});
