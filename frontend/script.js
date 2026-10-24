(() => {
  "use strict";

  const FRONTEND_VERSION = document.querySelector('meta[name="signal-app-version"]')?.content || "dev";
  const API_BASE = (
    document.querySelector('meta[name="signal-api-base"]')?.content
    || window.SIGNAL_API_BASE_URL
    || "http://127.0.0.1:8000"
  ).trim().replace(/\/+$/, "");

  window.__SIGNAL_DEBUG__ = Object.assign(window.__SIGNAL_DEBUG__ || {}, {
    frontendVersion: FRONTEND_VERSION,
    apiBase: API_BASE,
    apiAvailable: false,
    modelLoaded: false,
  });

  const MAX_CHARS = 5000;
  const CATEGORY_LABELS = {
    age: "Age-related bullying",
    ethnicity: "Ethnicity-related bullying",
    gender: "Gender-related bullying",
    religion: "Religion-related bullying",
    other_cyberbullying: "Other cyberbullying",
    not_cyberbullying: "Not classified as cyberbullying",
  };

  const ui = {
    form: document.querySelector("#analyzer-form"),
    input: document.querySelector("#post-text"),
    count: document.querySelector("#char-count"),
    formError: document.querySelector("#form-error"),
    analyzeButton: document.querySelector("#analyze-button"),
    analyzeLabel: document.querySelector("#analyze-button-label"),
    clearButton: document.querySelector("#clear-button"),
    resultPanel: document.querySelector("#result-panel"),
    emptyState: document.querySelector("#result-empty"),
    loadingState: document.querySelector("#result-loading"),
    resultContent: document.querySelector("#result-content"),
    category: document.querySelector("#prediction-category"),
    sentiment: document.querySelector("#sentiment-label"),
    scoreList: document.querySelector("#score-list"),
  };

  let analyzing = false;

  function setDebugStatus(apiAvailable, modelLoaded) {
    window.__SIGNAL_DEBUG__ = Object.assign(window.__SIGNAL_DEBUG__ || {}, {
      apiAvailable,
      modelLoaded,
      frontendVersion: FRONTEND_VERSION,
      apiBase: API_BASE,
    });

    if (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost") {
      console.info("[Signal debug]", {
        frontendVersion: FRONTEND_VERSION,
        apiBase: API_BASE,
        apiAvailable,
        modelLoaded,
      });
    }
  }

  function showError(message) {
    if (!message) {
      ui.formError.textContent = "";
      ui.formError.hidden = true;
      ui.formError.classList.remove("shake");
      return;
    }

    ui.formError.textContent = message;
    ui.formError.hidden = false;
    ui.formError.classList.remove("shake");
    void ui.formError.offsetWidth;
    ui.formError.classList.add("shake");
  }

  function updateCount() {
    const currentLength = ui.input.value.length;
    ui.count.textContent = `${currentLength.toLocaleString()} / ${MAX_CHARS.toLocaleString()}`;
    ui.count.classList.toggle("is-warning", currentLength >= MAX_CHARS * 0.9);
  }

  function showResultState(state) {
    const isEmpty = state === "empty";
    const isLoading = state === "loading";
    const isResult = state === "result";

    ui.emptyState.hidden = !isEmpty;
    ui.loadingState.hidden = !isLoading;
    ui.resultContent.hidden = !isResult;
    ui.resultPanel.setAttribute("aria-busy", String(isLoading));
  }

  function setLoadingState(isLoading) {
    analyzing = isLoading;
    ui.analyzeButton.disabled = isLoading;
    ui.input.disabled = isLoading;
    ui.analyzeButton.classList.toggle("loading", isLoading);
    ui.analyzeLabel.textContent = isLoading ? "Analyzing..." : "Analyze text";
  }

  function formatCategory(value) {
    return CATEGORY_LABELS[value] || value.replaceAll("_", " ");
  }

  function scoreCard(label, value, key) {
    const row = document.createElement("div");
    row.className = "score-row";

    const meta = document.createElement("div");
    meta.className = "score-meta";

    const labelEl = document.createElement("span");
    labelEl.textContent = label;

    const valueEl = document.createElement("span");
    valueEl.className = "score-value";
    valueEl.textContent = Number(value).toFixed(3);

    meta.append(labelEl, valueEl);

    const track = document.createElement("div");
    track.className = "score-track";

    const fill = document.createElement("span");
    fill.className = `score-fill ${key}`;

    if (key === "compound") {
      fill.style.width = `${((Number(value) + 1) / 2) * 100}%`;
    } else {
      fill.style.width = `${Number(value) * 100}%`;
    }

    track.append(fill);
    row.append(meta, track);
    return row;
  }

  function renderResult(data) {
    if (!data || typeof data.category !== "string" || typeof data.sentiment !== "string") {
      throw new Error("The API returned an incomplete result.");
    }

    const scores = data.sentiment_scores;
    const validKeys = ["negative", "neutral", "positive", "compound"];
    if (!scores || validKeys.some((key) => !Number.isFinite(scores[key]))) {
      throw new Error("The API returned an incomplete sentiment score set.");
    }

    ui.category.textContent = formatCategory(data.category);
    ui.sentiment.textContent = data.sentiment;
    ui.scoreList.replaceChildren(
      scoreCard("Negative", scores.negative, "negative"),
      scoreCard("Neutral", scores.neutral, "neutral"),
      scoreCard("Positive", scores.positive, "positive"),
      scoreCard("Compound", scores.compound, "compound")
    );

    showResultState("result");
  }

  function getErrorMessage(error) {
    if (error && error.name === "AbortError") {
      return "The request timed out. Please try again.";
    }

    if (error && typeof error.status === "number") {
      if (error.status === 422) {
        return "Please enter text with at least one non-whitespace character.";
      }
      if (error.status === 503) {
        return "The model is currently unavailable. Try again in a moment.";
      }
      return `Analysis failed with HTTP ${error.status}. Please try again.`;
    }

    if (error && error.message) {
      return error.message;
    }

    return "Could not connect to the local analysis API. Make sure the backend is running.";
  }

  async function checkApiStatus() {
    try {
      const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
      const data = response.ok ? await response.json() : null;
      const apiAvailable = Boolean(response.ok && data && typeof data.model_loaded === "boolean");
      const modelLoaded = Boolean(data && data.model_loaded === true);
      setDebugStatus(apiAvailable, modelLoaded);
      return { apiAvailable, modelLoaded };
    } catch {
      setDebugStatus(false, false);
      return { apiAvailable: false, modelLoaded: false };
    }
  }

  async function analyzeText() {
    if (analyzing) return;

    const rawText = ui.input.value;
    const trimmed = rawText.trim();

    if (!trimmed) {
      showError("Please enter some text before analyzing.");
      ui.input.focus();
      return;
    }

    if (rawText.length > MAX_CHARS) {
      showError(`Text must be ${MAX_CHARS.toLocaleString()} characters or fewer.`);
      ui.input.focus();
      return;
    }

    showError("");
    setLoadingState(true);
    showResultState("loading");

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: rawText }),
      });

      let payload = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        const detail = payload && typeof payload.detail === "string" ? payload.detail : "Prediction failed.";
        throw Object.assign(new Error(detail), { status: response.status });
      }

      if (!payload || typeof payload !== "object") {
        throw new Error("The API returned an invalid result.");
      }

      renderResult(payload);
    } catch (error) {
      showResultState("empty");
      showError(getErrorMessage(error));
    } finally {
      setLoadingState(false);
    }
  }

  function clearAnalysis() {
    ui.input.value = "";
    updateCount();
    showError("");
    showResultState("empty");
    ui.input.focus();
  }

  function handleInput() {
    updateCount();
    if (ui.input.value.length >= MAX_CHARS) {
      showError(`Maximum length reached: ${MAX_CHARS.toLocaleString()} characters.`);
      return;
    }
    if (!ui.formError.hidden) {
      showError("");
    }
  }

  function init() {
    updateCount();
    showResultState("empty");
    ui.form.addEventListener("submit", (event) => {
      event.preventDefault();
      analyzeText();
    });
    ui.clearButton.addEventListener("click", clearAnalysis);
    ui.input.addEventListener("input", handleInput);
    checkApiStatus();
  }

  init();
})();
