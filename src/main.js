const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

// ── Tab switching ──

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('border-accent', 'text-accent');
      b.classList.add('border-transparent', 'text-gray-500');
    });
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));

    btn.classList.remove('border-transparent', 'text-gray-500');
    btn.classList.add('border-accent', 'text-accent');
    document.getElementById(btn.dataset.tab).classList.remove('hidden');
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

function renderHistory(entries) {
  historyList.innerHTML = '';
  if (!entries || entries.length === 0) {
    historyList.innerHTML = '<p class="text-gray-500 text-center mt-8 text-sm">No transcriptions yet</p>';
    return;
  }
  // Show newest first
  for (const entry of [...entries].reverse()) {
    const card = document.createElement('div');
    card.className = 'bg-surface rounded-lg border border-card-border p-4';

    const stats = [
      formatTimestamp(entry.timestamp),
      formatAudioDuration(entry.audio_duration_ms),
      formatDuration(entry.duration_ms) + ' to transcribe',
      formatSpeed(entry),
      escapeHtml(entry.model_id),
    ].filter(Boolean);

    card.innerHTML = `
      <p class="text-sm text-gray-200 leading-relaxed mb-2">${escapeHtml(entry.text)}</p>
      <div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-500">
        ${stats.map(s => '<span>' + s + '</span>').join('')}
      </div>`;
    historyList.appendChild(card);
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
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

  const deleteBtn = `<button class="del-btn text-xs font-medium px-2 py-1.5 text-gray-500 hover:text-red-400 transition-colors cursor-pointer" data-id="${model.id}" title="Delete">&#x2715;</button>`;

  let actionHtml;
  if (isActive) {
    actionHtml = `<span class="text-xs text-accent font-medium px-3 py-1.5 bg-accent/10 rounded-md">Active</span>${deleteBtn}`;
  } else if (isDownloaded) {
    actionHtml = `<button class="use-btn text-xs font-medium px-3 py-1.5 bg-accent text-gray-900 rounded-md hover:bg-accent-hover transition-colors cursor-pointer" data-id="${model.id}">Use</button>${deleteBtn}`;
  } else if (isDownloading) {
    const pct = Math.round(model.progress * 100);
    actionHtml = `
      <div class="w-28" id="progress-${model.id}">
        <div class="progress-bar"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
        <span class="text-[10px] text-gray-500 mt-1 block text-right">${pct}%</span>
      </div>`;
  } else {
    actionHtml = `<button class="dl-btn text-xs font-medium px-3 py-1.5 border border-card-border text-gray-300 rounded-md hover:bg-card transition-colors cursor-pointer" data-id="${model.id}">Download</button>`;
  }

  return `
    <div class="flex items-center justify-between px-4 py-3" data-model-id="${model.id}">
      <div class="min-w-0 flex-1 mr-4">
        <div class="text-sm font-medium ${isActive ? 'text-accent' : 'text-gray-200'}">${model.name}</div>
        <div class="text-xs text-gray-500 mt-0.5">${model.desc}</div>
      </div>
      <div class="flex items-center gap-3 shrink-0">
        <span class="text-xs font-mono text-gray-500">${model.size}</span>
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
        <span class="text-[10px] text-gray-500 mt-1 block text-right">${pct}%</span>`;
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
  document.getElementById('cfg-ollama-url').value = cfg.ollama_url;
  document.getElementById('cfg-ollama-model').value = cfg.ollama_model;
  document.getElementById('cfg-output-dir').value = cfg.output_dir;
  // Restore audio device selection
  document.getElementById('audio-device').value = cfg.device_index;
}

async function saveConfig() {
  const cfg = {
    language: document.getElementById('cfg-language').value,
    threads: parseInt(document.getElementById('cfg-threads').value, 10),
    ollama_url: document.getElementById('cfg-ollama-url').value,
    ollama_model: document.getElementById('cfg-ollama-model').value,
    output_dir: document.getElementById('cfg-output-dir').value,
    device_index: parseInt(document.getElementById('audio-device').value, 10),
    active_engine: 'whisper',
    active_model_id: '',
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

  document.getElementById('save-config').addEventListener('click', saveConfig);
});
