const MATERIALS_PATH = "data/materials.json";
const START_API = "/api/start";
const SUBMIT_API = "/api/submit";

let currentUser = null;

// 只用于预热浏览器缓存，不再把 blob URL 交给 audio 播放
const audioWarmCache = new Map();
const audioWarmPromises = new Map();

let activeAudio = null;

const AUDIO_PRELOAD_CONCURRENCY = 3;
let audioPreloadStarted = false;

async function main() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("token");

  if (!token) {
    await autoAssignToken();
    return;
  }

  await loadUserByToken(token);
}

async function autoAssignToken() {
  const app = document.getElementById("app");

  app.innerHTML = `
    <div class="loading-card">
      <h2>正在分配实验任务</h2>
      <p>请稍候，系统正在为你分配本次用户研究的问卷。</p>
    </div>
  `;

  try {
    const res = await fetch(START_API, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      }
    });

    const data = await res.json();

    if (!res.ok) {
      app.innerHTML = `
        <div class="error-card">
          <h2>分配失败</h2>
          <p>${escapeHtml(data.error || "实验名额已满或分配失败，请联系研究者。")}</p>
        </div>
      `;
      return;
    }

    const token = encodeURIComponent(data.token);
    window.location.href = `${window.location.origin}${window.location.pathname}?token=${token}`;
  } catch (err) {
    console.error(err);

    app.innerHTML = `
      <div class="error-card">
        <h2>无法连接服务器</h2>
        <p>请稍后重试，或联系研究者。</p>
      </div>
    `;
  }
}

async function loadUserByToken(token) {
  const app = document.getElementById("app");

  try {
    const res = await fetch(MATERIALS_PATH);
    const materials = await res.json();

    currentUser = materials.find(x => x.token === token);

    if (!currentUser) {
      app.innerHTML = `
        <div class="error-card">
          <h2>token 无效</h2>
          <p>请使用研究者提供的链接，或联系研究者。</p>
        </div>
      `;
      return;
    }

    const userInfo = document.getElementById("user-info");
    if (userInfo) {
      userInfo.innerText = `当前用户：${currentUser.token}`;
    }

    renderPage(currentUser);
  } catch (err) {
    console.error(err);

    app.innerHTML = `
      <div class="error-card">
        <h2>加载实验材料失败</h2>
        <p>请稍后重试，或联系研究者。</p>
      </div>
    `;
  }
}

function renderPage(user) {
  const app = document.getElementById("app");
  app.innerHTML = "";

  const intro = document.createElement("div");
  intro.className = "intro-card";
  intro.innerHTML = `
    <h2>研究说明</h2>
    <p>
      本研究用于评估不同音乐推荐方法在<strong>情绪调节</strong>场景下的效果。
      你将看到 <strong>3 个情绪场景</strong>。每一轮包含方法 A、方法 B、方法 C，
      每个方法提供 <strong>1 段约 15 秒音乐</strong>。
    </p>
    <p>
      请根据每一轮的<strong>当前状态</strong>和<strong>调节目标</strong>，分别试听三首音乐，
      并从 1–7 分评价：
      <strong>1 表示非常不同意，7 表示非常同意</strong>。
      每轮最后还需要选择你最愿意用于本次情绪调节的一首音乐。
    </p>
    <p>
      建议使用 Wi-Fi 或稳定网络，并佩戴耳机完成。页面打开后会自动在后台准备音频。
      若个别播放器显示加载失败，请刷新页面或换浏览器打开。
    </p>
  `;
  app.appendChild(intro);

  const progress = document.createElement("div");
  progress.className = "progress-card";
  progress.innerHTML = `
    <div>
      <strong>完成进度</strong>
      <span id="progress-text">0%</span>
    </div>
    <div class="progress-track">
      <div id="progress-fill" class="progress-fill"></div>
    </div>
  `;
  app.appendChild(progress);

  user.trials.forEach(trial => {
    const div = document.createElement("div");
    div.className = "trial";

    div.innerHTML = `
      <div class="trial-header">
        <h2>第 ${trial.trial_index} 轮</h2>
        <span class="strategy-badge">调节目标：${escapeHtml(getStrategyDisplay(trial))}</span>
      </div>

      ${renderTrialContext(trial)}

      <p class="trial-hint">
        请想象你正处于上述情绪状态中。分别试听方法 A、方法 B、方法 C 的推荐音乐，
        然后判断每首音乐是否适合帮助你从当前情绪调节到目标状态。
      </p>

      <div class="methods">
        ${renderMethod(trial, "A", trial.A)}
        ${renderMethod(trial, "B", trial.B)}
        ${renderMethod(trial, "C", trial.C)}
      </div>

      <div class="preference-box">
        <h3>${escapeHtml(getPreferencePrompt(trial))}</h3>
        <p class="preference-hint">请在方法 A、方法 B、方法 C 中选择一个。</p>
        ${renderPreference(trial.trial_index)}
      </div>
    `;

    app.appendChild(div);
  });

  const submitBox = document.createElement("div");
  submitBox.className = "submit-bar";
  submitBox.innerHTML = `
    <div class="submit-left">
      <div class="submit-title">完成所有评分后提交问卷</div>
      <div id="submit-msg" class="submit-msg">请确认每一轮 A/B/C 三首音乐的评分和偏好选择都已完成。</div>
    </div>
    <button id="submit-btn" class="primary-submit">提交问卷</button>
  `;
  app.appendChild(submitBox);

  document.getElementById("submit-btn").addEventListener("click", submitResponses);

  startAutoPreloadAllAudio();

  app.addEventListener("change", updateProgress);
  updateProgress();
}

function renderTrialContext(trial) {
  const currentState = trial.current_state || trial.query_text || "";
  const regulationGoal = trial.regulation_goal || "";
  const fullText = trial.query_text || "";

  if (currentState || regulationGoal) {
    return `
      <div class="query-block">
        <div class="query-section">
          <strong>当前状态：</strong>
          <p class="query">${escapeHtml(currentState)}</p>
        </div>
        <div class="query-section">
          <strong>调节目标：</strong>
          <p class="query">${escapeHtml(regulationGoal || fullText)}</p>
        </div>
      </div>
    `;
  }

  return `<p class="query">${escapeHtml(fullText)}</p>`;
}

function renderMethod(trial, label, data) {
  const ratingFields = getRatingFields(trial);
  const ratingPrompts = getRatingPrompts(trial);

  return `
    <div class="method-card">
      <h3>方法 ${label}</h3>

      <div class="songs">
        ${renderSongs(data.items)}
      </div>

      <div class="ratings">
        ${ratingFields.map(field => {
          const prompt = ratingPrompts[field] || defaultRatingPrompt(field);
          return renderRating(trial.trial_index, label, field, prompt);
        }).join("")}
      </div>
    </div>
  `;
}

function renderSongs(items) {
  if (!items || items.length === 0) {
    return `<p class="placeholder">暂无推荐音乐。</p>`;
  }

  return `
    <div class="song-list single-song">
      ${items.map(() => `
        <div class="song-item">
          <p class="song-title">推荐音乐</p>
          ${renderAudio(items[0])}
        </div>
      `).join("")}
    </div>
  `;
}

function renderAudio(item) {
  const src = item.audio_url || item.public_audio_url || item.audio_path || "";

  if (!src) {
    return `<p class="placeholder">暂无音频链接</p>`;
  }

  return `
    <div class="audio-box is-pending" data-src="${escapeAttribute(src)}">
      <span class="audio-status">音频正在准备...</span>
      <audio controls preload="auto" class="study-audio" src="${escapeAttribute(src)}">
        你的浏览器不支持音频播放。
      </audio>
    </div>
  `;
}

function startAutoPreloadAllAudio() {
  if (audioPreloadStarted) {
    return;
  }

  audioPreloadStarted = true;

  const boxes = Array.from(document.querySelectorAll(".audio-box"));

  if (boxes.length === 0) {
    return;
  }

  updateAudioPreloadSummary(0, boxes.length, false);

  let nextIndex = 0;
  let doneCount = 0;

  async function worker() {
    while (nextIndex < boxes.length) {
      const box = boxes[nextIndex];
      nextIndex += 1;

      await preloadOneAudioBox(box);

      doneCount += 1;
      updateAudioPreloadSummary(doneCount, boxes.length, doneCount === boxes.length);
    }
  }

  const workerCount = Math.min(AUDIO_PRELOAD_CONCURRENCY, boxes.length);
  const workers = [];

  for (let i = 0; i < workerCount; i += 1) {
    workers.push(worker());
  }

  Promise.all(workers).then(() => {
    updateAudioPreloadSummary(doneCount, boxes.length, true);
  });
}

async function preloadOneAudioBox(box) {
  const src = box.dataset.src;
  const status = box.querySelector(".audio-status");
  const audio = box.querySelector("audio");

  if (!src) {
    box.className = "audio-box is-error";
    status.innerText = "音频链接为空。";
    return;
  }

  bindAudioEvents(audio, box, status);

  try {
    box.className = "audio-box is-loading";
    status.innerText = "正在准备音频...";

    audio.src = src;
    audio.preload = "auto";
    audio.load();

    await warmAudioCache(src);

    if (audio.readyState >= 3) {
      box.className = "audio-box is-ready";
      status.innerText = "音频已准备好，可直接播放。";
    } else {
      box.className = "audio-box is-loading";
      status.innerText = "音频已缓存，播放器正在准备。";
    }
  } catch (err) {
    console.error(err);

    audio.src = src;
    audio.preload = "auto";
    audio.load();

    box.className = "audio-box is-loading";
    status.innerText = "自动缓存失败，播放器将直接加载音频。";
  }
}

async function warmAudioCache(src) {
  if (audioWarmCache.has(src)) {
    return true;
  }

  if (audioWarmPromises.has(src)) {
    return await audioWarmPromises.get(src);
  }

  const promise = fetch(src, {
    method: "GET",
    mode: "cors",
    cache: "force-cache"
  })
    .then(async res => {
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const buffer = await res.arrayBuffer();

      if (!buffer || buffer.byteLength === 0) {
        throw new Error("Empty audio body");
      }

      audioWarmCache.set(src, true);
      return true;
    })
    .finally(() => {
      audioWarmPromises.delete(src);
    });

  audioWarmPromises.set(src, promise);
  return await promise;
}

function bindAudioEvents(audio, box, status) {
  if (audio.dataset.bound === "1") {
    return;
  }

  audio.dataset.bound = "1";

  audio.addEventListener("loadstart", () => {
    if (!box.classList.contains("is-error")) {
      box.className = "audio-box is-loading";
      status.innerText = "播放器正在加载音频...";
    }
  });

  audio.addEventListener("canplay", () => {
    if (!box.classList.contains("is-error")) {
      box.className = "audio-box is-ready";
      status.innerText = "音频已准备好，可直接播放。";
    }
  });

  audio.addEventListener("canplaythrough", () => {
    if (!box.classList.contains("is-error")) {
      box.className = "audio-box is-ready";
      status.innerText = "音频已充分加载，可稳定播放。";
    }
  });

  audio.addEventListener("waiting", () => {
    if (!box.classList.contains("is-error")) {
      box.className = "audio-box is-loading";
      status.innerText = "音频缓冲中，请稍候。";
    }
  });

  audio.addEventListener("playing", () => {
    box.className = "audio-box is-ready";
    status.innerText = "正在播放。";
  });

  audio.addEventListener("ended", () => {
    if (!box.classList.contains("is-error")) {
      status.innerText = "播放结束。";
    }
  });

  audio.addEventListener("error", () => {
    const code = audio.error ? audio.error.code : "";
    box.className = "audio-box is-error";
    status.innerText = `播放器加载失败，请刷新页面或换浏览器打开。错误码：${code}`;
  });

  audio.addEventListener("play", () => {
    if (activeAudio && activeAudio !== audio) {
      activeAudio.pause();
    }

    document.querySelectorAll("audio").forEach(a => {
      if (a !== audio) {
        a.pause();
      }
    });

    activeAudio = audio;
  });
}

function updateAudioPreloadSummary(done, total, finished = false) {
  let card = document.getElementById("audio-preload-summary");

  if (!card) {
    const app = document.getElementById("app");
    const firstTrial = document.querySelector(".trial");

    card = document.createElement("div");
    card.id = "audio-preload-summary";
    card.className = "audio-preload-summary";

    if (firstTrial) {
      app.insertBefore(card, firstTrial);
    } else {
      app.appendChild(card);
    }
  }

  if (finished) {
    card.innerHTML = `
      <strong>音频准备完成</strong>
      <span>所有 ${total} 段音频已尝试加载。若个别播放器显示错误，请刷新页面或换浏览器打开。</span>
    `;
    card.className = "audio-preload-summary is-ready";
    return;
  }

  card.innerHTML = `
    <strong>正在准备音频</strong>
    <span>${done}/${total} 段已处理。你可以先阅读说明，音频会在后台自动准备。</span>
  `;
  card.className = "audio-preload-summary";
}

function renderRating(trialIndex, methodLabel, field, text) {
  const name = `t${trialIndex}_${methodLabel}_${field}`;

  return `
    <label class="rating-row">
      <span>${escapeHtml(text)}</span>
      <select name="${name}" required>
        <option value="">请选择</option>
        <option value="1">1 - 非常不同意</option>
        <option value="2">2</option>
        <option value="3">3</option>
        <option value="4">4 - 一般</option>
        <option value="5">5</option>
        <option value="6">6</option>
        <option value="7">7 - 非常同意</option>
      </select>
    </label>
  `;
}

function renderPreference(trialIndex) {
  return `
    <div class="preference-options">
      <label>
        <input type="radio" name="t${trialIndex}_preference" value="A">
        <span>方法 A</span>
      </label>
      <label>
        <input type="radio" name="t${trialIndex}_preference" value="B">
        <span>方法 B</span>
      </label>
      <label>
        <input type="radio" name="t${trialIndex}_preference" value="C">
        <span>方法 C</span>
      </label>
    </div>
  `;
}

async function submitResponses() {
  const response = collectResponse();
  const error = validateResponse(response);
  const msg = document.getElementById("submit-msg");
  const btn = document.getElementById("submit-btn");

  if (error) {
    msg.innerText = error;
    msg.className = "submit-msg error";
    scrollToFirstIncomplete(response);
    return;
  }

  btn.disabled = true;
  btn.innerText = "正在提交...";
  msg.innerText = "正在提交，请不要关闭页面。";
  msg.className = "submit-msg";

  try {
    const res = await fetch(SUBMIT_API, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(response)
    });

    const data = await res.json();

    if (!res.ok) {
      msg.innerText = data.error || "提交失败，请稍后重试。";
      msg.className = "submit-msg error";
      btn.disabled = false;
      btn.innerText = "提交问卷";
      return;
    }

    msg.innerText = "提交成功，感谢参与！";
    msg.className = "submit-msg success";
    btn.innerText = "已提交";
    disableAllInputs();
  } catch (err) {
    console.error(err);
    msg.innerText = "提交失败：无法连接服务器。";
    msg.className = "submit-msg error";
    btn.disabled = false;
    btn.innerText = "提交问卷";
  }
}

function collectResponse() {
  const response = {
    token: currentUser.token,
    participant_index: currentUser.participant_index,
    submitted_at: new Date().toISOString(),
    version: "v2_single_song_single_rating",
    trials: []
  };

  for (const trial of currentUser.trials) {
    const ratingFields = getRatingFields(trial);

    const trialResult = {
      trial_index: trial.trial_index,
      query_id: trial.query_id,
      strategy: trial.strategy,
      strategy_zh: trial.strategy_zh || "",
      current_state: trial.current_state || "",
      regulation_goal: trial.regulation_goal || "",
      preference: getRadioValue(`t${trial.trial_index}_preference`),
      ratings: {}
    };

    for (const label of ["A", "B", "C"]) {
      trialResult.ratings[label] = {};

      for (const field of ratingFields) {
        trialResult.ratings[label][field] =
          getSelectValue(`t${trial.trial_index}_${label}_${field}`);
      }
    }

    response.trials.push(trialResult);
  }

  return response;
}

function validateResponse(response) {
  for (const trial of response.trials) {
    const materialTrial = currentUser.trials.find(x => x.trial_index === trial.trial_index);
    const ratingFields = getRatingFields(materialTrial || {});

    for (const label of ["A", "B", "C"]) {
      const r = trial.ratings[label] || {};

      for (const field of ratingFields) {
        if (!r[field]) {
          return `请完成第 ${trial.trial_index} 轮方法 ${label} 的评分。`;
        }
      }
    }

    if (!trial.preference) {
      return `请选择第 ${trial.trial_index} 轮最适合的音乐。`;
    }
  }

  return "";
}

function scrollToFirstIncomplete(response) {
  for (const trial of response.trials) {
    const materialTrial = currentUser.trials.find(x => x.trial_index === trial.trial_index);
    const ratingFields = getRatingFields(materialTrial || {});

    for (const label of ["A", "B", "C"]) {
      const r = trial.ratings[label] || {};

      for (const field of ratingFields) {
        if (!r[field]) {
          focusElement(`select[name="t${trial.trial_index}_${label}_${field}"]`);
          return;
        }
      }
    }

    if (!trial.preference) {
      focusElement(`input[name="t${trial.trial_index}_preference"]`);
      return;
    }
  }
}

function focusElement(selector) {
  const el = document.querySelector(selector);

  if (!el) {
    return;
  }

  const trial = el.closest(".trial");

  if (trial) {
    trial.scrollIntoView({
      behavior: "smooth",
      block: "center"
    });
  } else {
    el.scrollIntoView({
      behavior: "smooth",
      block: "center"
    });
  }

  setTimeout(() => {
    try {
      el.focus();
    } catch (err) {
      // ignore
    }
  }, 300);
}

function updateProgress() {
  if (!currentUser) {
    return;
  }

  let total = 0;
  let done = 0;

  for (const trial of currentUser.trials) {
    const ratingFields = getRatingFields(trial);

    total += 3 * ratingFields.length;
    total += 1;

    for (const label of ["A", "B", "C"]) {
      for (const field of ratingFields) {
        const value = getSelectValue(`t${trial.trial_index}_${label}_${field}`);
        if (value) {
          done += 1;
        }
      }
    }

    const preference = getRadioValue(`t${trial.trial_index}_preference`);
    if (preference) {
      done += 1;
    }
  }

  const percent = total > 0 ? Math.round((done / total) * 100) : 0;

  const text = document.getElementById("progress-text");
  const fill = document.getElementById("progress-fill");
  const msg = document.getElementById("submit-msg");

  if (text) {
    text.innerText = `${percent}%（${done}/${total}）`;
  }

  if (fill) {
    fill.style.width = `${percent}%`;
  }

  if (msg && percent === 100 && !msg.classList.contains("success")) {
    msg.innerText = "所有评分已完成，可以提交问卷。";
    msg.className = "submit-msg ready";
  }
}

function disableAllInputs() {
  document.querySelectorAll("select, input, button").forEach(el => {
    el.disabled = true;
  });

  document.querySelectorAll("audio").forEach(audio => {
    audio.controls = false;
  });
}

function getSelectValue(name) {
  const el = document.querySelector(`select[name="${name}"]`);
  return el ? el.value : "";
}

function getRadioValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : "";
}

function getRatingFields(trial) {
  if (trial && Array.isArray(trial.rating_fields) && trial.rating_fields.length > 0) {
    return trial.rating_fields;
  }

  return ["regulation_suitability"];
}

function getRatingPrompts(trial) {
  if (trial && trial.rating_prompts && typeof trial.rating_prompts === "object") {
    return trial.rating_prompts;
  }

  return {
    regulation_suitability: "这首音乐适合帮助我从当前情绪调节到目标状态。"
  };
}

function defaultRatingPrompt(field) {
  if (field === "regulation_suitability") {
    return "这首音乐适合帮助我从当前情绪调节到目标状态。";
  }

  return "请对这首音乐进行评分。";
}

function getPreferencePrompt(trial) {
  return trial.preference_prompt || "本轮三首音乐中，我最愿意选择哪一首来帮助自己完成情绪调节？";
}

function getStrategyDisplay(trial) {
  if (trial.strategy_zh) {
    return trial.strategy_zh;
  }

  return strategyText(trial.strategy);
}

function strategyText(strategy) {
  const s = String(strategy || "").toLowerCase();

  if (s === "maintain") return "保持当前状态";
  if (s === "calm") return "平静下来";
  if (s === "comfort") return "获得安慰";
  if (s === "energize") return "振作起来";

  return strategy || "情绪调节";
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(text) {
  return escapeHtml(text);
}

main();