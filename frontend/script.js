const API_BASE = 'http://127.0.0.1:8000';
const textInput = document.querySelector('#post-text');
const count = document.querySelector('#char-count');
const submitButton = document.querySelector('#analyze-button');
const formError = document.querySelector('#form-error');
const statusLabel = document.querySelector('#api-status');
const statusDot = document.querySelector('#status-dot');
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));

function setHealth(online) {
  statusDot.className = `status-dot ${online ? 'online' : 'offline'}`;
  statusLabel.textContent = online ? 'Model online' : 'API offline';
  document.querySelector('#hero-model-status').textContent = online ? 'ONLINE' : 'OFFLINE';
}

async function refreshHealth() {
  try { const response = await fetch(`${API_BASE}/health`); const data = await response.json(); setHealth(Boolean(response.ok && data.model_loaded)); }
  catch { setHealth(false); }
}

function renderMetrics(info) {
  const note = document.querySelector('#metrics-note');
  const grid = document.querySelector('#metric-grid');
  if (!info?.available) { note.textContent = 'Local training metrics unavailable'; grid.innerHTML = '<article class="glass metric-card"><span class="micro-label">MODEL</span><strong>Unavailable</strong><small>Run the training command to populate this dashboard.</small></article>'; return; }
  const selected = info.models?.['TF-IDF + VADER · Logistic Regression'];
  note.textContent = `Stratified group holdout · seed ${info.split.random_seed} · ${info.split.test_rows.toLocaleString()} test posts`;
  grid.innerHTML = [
    ['SELECTED MODEL', info.selected_model, `${info.dataset.rows_used.toLocaleString()} posts used`],
    ['ACCURACY', `${(selected.accuracy * 100).toFixed(2)}%`, 'Local holdout result'],
    ['MACRO F1', selected.macro_f1.toFixed(3), 'Equal weight per category'],
    ['WEIGHTED F1', selected.weighted_f1.toFixed(3), 'Weighted by class support']
  ].map(([label, value, sub]) => `<article class="glass metric-card"><span class="micro-label">${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong><small>${escapeHTML(sub)}</small></article>`).join('');
  document.querySelector('#comparison-body').innerHTML = Object.entries(info.models).map(([name, metric]) => `<tr><td>${escapeHTML(name)}</td><td>${(metric.accuracy * 100).toFixed(2)}%</td><td>${metric.macro_f1.toFixed(3)}</td><td>${metric.weighted_f1.toFixed(3)}</td></tr>`).join('');
  document.querySelector('#category-list').innerHTML = info.dataset.labels.map(label => `<span class="category-chip">${escapeHTML(label.replaceAll('_', ' '))}</span>`).join('');
  renderMatrix(info.dataset.labels, selected.confusion_matrix);
}

function renderMatrix(labels, matrix) {
  const largest = Math.max(...matrix.flat(), 1);
  let markup = '<table class="matrix-table" aria-label="Confusion matrix"><thead><tr><th>Actual ↓ / Predicted →</th>';
  markup += labels.map(label => `<th>${escapeHTML(label.replaceAll('_', ' '))}</th>`).join('') + '</tr></thead><tbody>';
  matrix.forEach((row, i) => { markup += `<tr><th>${escapeHTML(labels[i].replaceAll('_', ' '))}</th>` + row.map(value => `<td class="matrix-cell" style="--intensity:${(0.08 + 0.72 * value / largest).toFixed(2)}">${value.toLocaleString()}</td>`).join('') + '</tr>'; });
  document.querySelector('#confusion-matrix').innerHTML = markup + '</tbody></table>';
}

async function loadMetrics() {
  try { const response = await fetch(`${API_BASE}/model-info`); renderMetrics(await response.json()); }
  catch { document.querySelector('#metrics-note').textContent = 'Could not reach local API'; }
}

textInput.addEventListener('input', () => { count.textContent = `${textInput.value.length.toLocaleString()} / 5,000`; formError.hidden = true; });
document.querySelector('#example-button').addEventListener('click', () => { textInput.value = 'You are such an idiot. Everyone in class knows you are pathetic.'; textInput.dispatchEvent(new Event('input')); textInput.focus(); });
document.querySelector('#clear-button').addEventListener('click', () => { textInput.value = ''; textInput.dispatchEvent(new Event('input')); showState('empty'); textInput.focus(); });

function showState(state) {
  document.querySelector('#result-empty').hidden = state !== 'empty';
  document.querySelector('#result-loading').hidden = state !== 'loading';
  document.querySelector('#result-content').hidden = state !== 'result';
}

function renderResult(result) {
  document.querySelector('#prediction-category').textContent = result.category.replaceAll('_', ' ');
  document.querySelector('#sentiment-label').textContent = result.sentiment;
  const confidenceWrap = document.querySelector('#confidence-wrap');
  confidenceWrap.hidden = result.confidence == null;
  if (result.confidence != null) document.querySelector('#confidence-value').textContent = `${(result.confidence * 100).toFixed(1)}%`;
  const names = [['negative','Negative'],['neutral','Neutral'],['positive','Positive'],['compound','Compound']];
  document.querySelector('#score-list').innerHTML = names.map(([key,label]) => {
    const value = result.sentiment_scores[key]; const width = key === 'compound' ? Math.abs(value) * 100 : value * 100;
    return `<div class="score-item"><span>${label}</span><b>${value.toFixed(3)}</b><div class="score-track"><i style="width:${width.toFixed(1)}%"></i></div></div>`;
  }).join('');
  showState('result');
}

submitButton.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) { formError.textContent = 'Enter some text before analyzing.'; formError.hidden = false; textInput.focus(); return; }
  formError.hidden = true; submitButton.disabled = true; submitButton.querySelector('span:first-child').textContent = 'Analyzing…'; showState('loading');
  try {
    const response = await fetch(`${API_BASE}/predict`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'The API could not analyze this text.');
    renderResult(body);
  } catch (error) {
    showState('empty'); formError.textContent = error.message || 'Could not connect to the analysis API.'; formError.hidden = false;
  } finally { submitButton.disabled = false; submitButton.querySelector('span:first-child').textContent = 'Analyze text'; refreshHealth(); }
});

textInput.addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') submitButton.click(); });
refreshHealth(); loadMetrics(); setInterval(refreshHealth, 15000);
