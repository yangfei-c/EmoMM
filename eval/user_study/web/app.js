const MATERIALS_PATH = "data/materials.json";
const START_API = "/api/start";
const SUBMIT_API = "/api/submit";

const RATING_FIELD = "regulation_suitability";
const RATING_PROMPT = "这首音乐适合帮助我从当前情绪调节到目标状态。";

// 手机端不要一次性并发太高；9 段音频，2 并发比较稳。
const AUDIO_PRELOAD_CONCURRENCY = 2;

let currentUser = null;
let activeAudio = null;

// 原始 Spaces URL -> Blob URL
const audioBlobUrls = new Map();
const audioPreloadPromises = new Map();

// 隐藏时间记录：用户看不到，只随提交结果写入 JSON
const pageStartedAt = new Date().toISOString();
const pageStartedMs = Date.now();

main();

async function main() {
  const token = new URLSearchParams(window.location.search).get("token");

  if (!token) {
    await autoAssignToken();
    return;
  }

  await loadUser(token);
}

async function autoAssignToken() {
  const app = document.getElementById("app");

  app.innerHTML = `
    <div class="loading-card">
      <h2>正在分配实验任务</h2>
      <p>请稍候，系统正在为你分配问卷。</p>
    </div>
  `;

  try {
    const res = await fetch(START_API, {
      method: "POST"
    });

    const data = await res.json();

    if (!res.ok) {
      app.innerHTML = `
        <div class="error-card">
          <h2>分配失败</h2>
          <p>${escapeHtml(data.error || "实验名额已满或暂时不可用。")}</p>
        </div>
      `;
      return;
    }

    window.location.href =
      `${window.location.origin}${window.location.pathname}?token=${encodeURIComponent(data.token)}`;
  } catch (e) {
    console.error(e);

    app.innerHTML = `
      <div class="error-card">
        <h2>无法连接服务器</h2>
        <p>请稍后重试，或联系研究者。</p>
      </div>
    `;
  }
}

async function loadUser(token) {
  const app = document.getElementById("app");

  try {
    const res = await fetch(MATERIALS_PATH, {
      cache: "no-cache"
    });

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

    renderPage();

    // 不阻塞页面渲染；页面先出现，音频在后台下载成 Blob。
    preloadAllAudio();
  } catch (e) {
    console.error(e);

    app.innerHTML = `
      <div class="error-card">
        <h2>加载问卷失败</h2>
        <p>请刷新页面，或联系研究者。</p>
      </div>
    `;
  }
}

function renderPage() {
  const app = document.getElementById("app");
  app.innerHTML = "";

  app.appendChild(htmlToNode(`
    <div class="intro-card">
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
        每轮最后请选择你最愿意用于本次情绪调节的一首音乐。
      </p>
      <p>
        页面打开后会自动准备音频。请等待播放器上方显示“音频已加载，可直接播放”后再试听。
        建议使用稳定网络，并佩戴耳机完成。
      </p>
    </div>
  `));

  app.appendChild(htmlToNode(`
    <div class="audio-preload-summary" id="audio-preload-summary">
      <strong>正在准备音频</strong>
      <span id="audio-preload-text">音频会在后台加载，请稍候。</span>
    </div>
  `));

  app.appendChild(htmlToNode(`
    <div class="progress-card">
      <div>
        <strong>完成进度</strong>
        <span id="progress-text">0%</span>
      </div>
      <div class="progress-track">
        <div id="progress-fill" class="progress-fill"></div>
      </div>
    </div>
  `));

  for (const trial of currentUser.trials) {
    app.appendChild(htmlToNode(renderTrial(trial)));
  }

  app.appendChild(htmlToNode(`
    <div class="submit-bar">
      <div class="submit-left">
        <div class="submit-title">完成所有评分后提交问卷</div>
        <div id="submit-msg" class="submit-msg">
          请完成每一轮 A/B/C 三首音乐的评分和偏好选择。
        </div>
      </div>
      <button id="submit-btn" class="primary-submit">提交问卷</button>
    </div>
  `));

  document.getElementById("submit-btn").addEventListener("click", submitResponses);
  app.addEventListener("change", updateProgress);
  app.addEventListener("play", onAudioPlay, true);

  window.addEventListener("beforeunload", releaseBlobUrls);

  updateProgress();
}

function renderTrial(trial) {
  return `
    <div class="trial">
      <div class="trial-header">
        <h2>第 ${trial.trial_index} 轮</h2>
        <span class="strategy-badge">
          调节目标：${escapeHtml(trial.strategy_zh || trial.strategy || "情绪调节")}
        </span>
      </div>

      <div class="query-block">
        <div class="query-section">
          <strong>当前状态：</strong>
          <p class="query">${escapeHtml(trial.current_state || trial.query_text || "")}</p>
        </div>
        <div class="query-section">
          <strong>调节目标：</strong>
          <p class="query">${escapeHtml(trial.regulation_goal || "")}</p>
        </div>
      </div>

      <p class="trial-hint">
        请想象自己正处于上述状态中，判断每首音乐是否适合帮助你从当前情绪调节到目标状态。
      </p>

      <div class="methods">
        ${renderMethod(trial, "A")}
        ${renderMethod(trial, "B")}
        ${renderMethod(trial, "C")}
      </div>

      <div class="preference-box">
        <h3>${escapeHtml(trial.preference_prompt || "本轮三首音乐中，我最愿意选择哪一首来帮助自己完成情绪调节？")}</h3>
        <div class="preference-options">
          ${["A", "B", "C"].map(label => `
            <label>
              <input type="radio" name="t${trial.trial_index}_preference" value="${label}">
              <span>方法 ${label}</span>
            </label>
          `).join("")}
        </div>
      </div>
    </div>
  `;
}

function renderMethod(trial, label) {
  const item = trial[label].items[0];
  const audioUrl = item.audio_url || item.public_audio_url || item.audio_path || "";

  return `
    <div class="method-card">
      <h3>方法 ${label}</h3>

      <div class="song-item">
        <p class="song-title">推荐音乐</p>
        ${
          audioUrl
            ? `
              <div class="audio-box" data-audio-url="${escapeAttribute(audioUrl)}">
                <div class="audio-status">音频准备中，请稍候...</div>
                <audio
                  controls
                  preload="auto"
                  class="study-audio"
                  src="${escapeAttribute(audioUrl)}"
                >
                  你的浏览器不支持音频播放。
                </audio>
              </div>
            `
            : `<p class="placeholder">暂无音频链接</p>`
        }
      </div>

      <label class="rating-row">
        <span>${escapeHtml(getRatingPrompt(trial))}</span>
        <select name="t${trial.trial_index}_${label}_${RATING_FIELD}" required>
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
    </div>
  `;
}

function getRatingPrompt(trial) {
  if (trial.rating_prompts && trial.rating_prompts[RATING_FIELD]) {
    return trial.rating_prompts[RATING_FIELD];
  }

  return RATING_PROMPT;
}

async function preloadAllAudio() {
  const urls = getAllAudioUrls();

  if (urls.length === 0) {
    updateAudioSummary(0, 0, true);
    return;
  }

  let nextIndex = 0;
  let done = 0;

  updateAudioSummary(done, urls.length, false);

  async function worker() {
    while (nextIndex < urls.length) {
      const url = urls[nextIndex];
      nextIndex += 1;

      await preloadAudioBlob(url);
      applyAudioSource(url);

      done += 1;
      updateAudioSummary(done, urls.length, done === urls.length);
    }
  }

  const workerCount = Math.min(AUDIO_PRELOAD_CONCURRENCY, urls.length);
  const workers = Array.from({ length: workerCount }, () => worker());

  await Promise.all(workers);
}

function getAllAudioUrls() {
  const urls = [];

  for (const trial of currentUser.trials) {
    for (const label of ["A", "B", "C"]) {
      const item = trial[label].items[0];
      const url = item.audio_url || item.public_audio_url || item.audio_path || "";

      if (url && !urls.includes(url)) {
        urls.push(url);
      }
    }
  }

  return urls;
}

async function preloadAudioBlob(url) {
  if (audioBlobUrls.has(url)) {
    return audioBlobUrls.get(url);
  }

  if (audioPreloadPromises.has(url)) {
    return await audioPreloadPromises.get(url);
  }

  const promise = fetch(url, {
    method: "GET",
    mode: "cors",
    cache: "force-cache"
  })
    .then(async res => {
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const blob = await res.blob();

      if (!blob || blob.size === 0) {
        throw new Error("empty audio blob");
      }

      const blobUrl = URL.createObjectURL(blob);
      audioBlobUrls.set(url, blobUrl);

      return blobUrl;
    })
    .catch(err => {
      console.warn("音频预加载失败，回退原始 URL:", url, err);
      return "";
    })
    .finally(() => {
      audioPreloadPromises.delete(url);
    });

  audioPreloadPromises.set(url, promise);
  return await promise;
}

function applyAudioSource(originalUrl) {
  const blobUrl = audioBlobUrls.get(originalUrl);

  document.querySelectorAll(".audio-box").forEach(box => {
    if (box.dataset.audioUrl !== originalUrl) {
      return;
    }

    const audio = box.querySelector("audio");
    const status = box.querySelector(".audio-status");

    if (!audio || !status) {
      return;
    }

    bindAudioStatus(audio, status, box);

    if (blobUrl) {
      audio.src = blobUrl;
      audio.load();
      status.innerText = "音频已加载，可直接播放。";
      box.classList.remove("is-fallback", "is-error");
      box.classList.add("is-ready");
    } else {
      audio.src = originalUrl;
      audio.load();
      status.innerText = "音频将在线播放。若加载失败，请刷新页面。";
      box.classList.remove("is-ready", "is-error");
      box.classList.add("is-fallback");
    }
  });
}

function bindAudioStatus(audio, status, box) {
  if (audio.dataset.bound === "1") {
    return;
  }

  audio.dataset.bound = "1";

  audio.addEventListener("loadstart", () => {
    if (!box.classList.contains("is-ready")) {
      status.innerText = "播放器正在准备音频...";
    }
  });

  audio.addEventListener("canplay", () => {
    if (!box.classList.contains("is-error")) {
      status.innerText = "音频已准备好，可播放。";
      box.classList.add("is-ready");
    }
  });

  audio.addEventListener("canplaythrough", () => {
    if (!box.classList.contains("is-error")) {
      status.innerText = "音频已充分加载，可稳定播放。";
      box.classList.add("is-ready");
    }
  });

  audio.addEventListener("waiting", () => {
    if (!box.classList.contains("is-error")) {
      status.innerText = "音频缓冲中，请稍候。";
    }
  });

  audio.addEventListener("playing", () => {
    status.innerText = "正在播放。";
  });

  audio.addEventListener("ended", () => {
    if (!box.classList.contains("is-error")) {
      status.innerText = "播放结束。";
    }
  });

  audio.addEventListener("error", () => {
    const originalUrl = box.dataset.audioUrl || "";

    if (audio.src.startsWith("blob:") && originalUrl) {
      audio.src = originalUrl;
      audio.load();
      status.innerText = "已切换为在线播放，请重新点击播放。";
      box.classList.remove("is-ready");
      box.classList.add("is-fallback");
      return;
    }

    status.innerText = "音频加载失败，请刷新页面或换浏览器打开。";
    box.classList.add("is-error");
  });
}

function updateAudioSummary(done, total, finished) {
  const text = document.getElementById("audio-preload-text");
  const box = document.getElementById("audio-preload-summary");

  if (!text || !box) {
    return;
  }

  if (total === 0) {
    text.innerText = "没有需要加载的音频。";
    box.classList.add("is-ready");
    return;
  }

  if (finished) {
    text.innerText = `音频准备完成：${done}/${total}。可以开始试听。`;
    box.classList.add("is-ready");
    return;
  }

  text.innerText = `正在准备音频：${done}/${total}。你可以先阅读场景文本。`;
}

function onAudioPlay(e) {
  if (!e.target.classList.contains("study-audio")) {
    return;
  }

  if (activeAudio && activeAudio !== e.target) {
    activeAudio.pause();
  }

  document.querySelectorAll("audio").forEach(audio => {
    if (audio !== e.target) {
      audio.pause();
    }
  });

  activeAudio = e.target;
}

function collectResponse() {
  const submitClickedAt = new Date().toISOString();
  const submitClickedMs = Date.now();

  return {
    token: currentUser.token,
    participant_index: currentUser.participant_index,
    submitted_at: submitClickedAt,
    version: "v2_single_song_single_rating",

    timing: {
      page_started_at: pageStartedAt,
      submit_clicked_at: submitClickedAt,
      total_duration_seconds: secondsBetween(pageStartedMs, submitClickedMs)
    },

    trials: currentUser.trials.map(trial => ({
      trial_index: trial.trial_index,
      query_id: trial.query_id,
      strategy: trial.strategy,
      strategy_zh: trial.strategy_zh || "",
      current_state: trial.current_state || "",
      regulation_goal: trial.regulation_goal || "",
      preference: getRadioValue(`t${trial.trial_index}_preference`),
      ratings: {
        A: { [RATING_FIELD]: getSelectValue(`t${trial.trial_index}_A_${RATING_FIELD}`) },
        B: { [RATING_FIELD]: getSelectValue(`t${trial.trial_index}_B_${RATING_FIELD}`) },
        C: { [RATING_FIELD]: getSelectValue(`t${trial.trial_index}_C_${RATING_FIELD}`) }
      }
    }))
  };
}

function validateResponse(response) {
  for (const trial of response.trials) {
    for (const label of ["A", "B", "C"]) {
      if (!trial.ratings[label][RATING_FIELD]) {
        return `请完成第 ${trial.trial_index} 轮方法 ${label} 的评分。`;
      }
    }

    if (!trial.preference) {
      return `请选择第 ${trial.trial_index} 轮最愿意使用的音乐。`;
    }
  }

  return "";
}

async function submitResponses() {
  const response = collectResponse();
  const error = validateResponse(response);
  const msg = document.getElementById("submit-msg");
  const btn = document.getElementById("submit-btn");

  if (error) {
    msg.innerText = error;
    msg.className = "submit-msg error";
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
    disableInputs();
  } catch (e) {
    console.error(e);

    msg.innerText = "提交失败：无法连接服务器。";
    msg.className = "submit-msg error";
    btn.disabled = false;
    btn.innerText = "提交问卷";
  }
}

function updateProgress() {
  let total = 0;
  let done = 0;

  for (const trial of currentUser.trials) {
    total += 4;

    for (const label of ["A", "B", "C"]) {
      if (getSelectValue(`t${trial.trial_index}_${label}_${RATING_FIELD}`)) {
        done += 1;
      }
    }

    if (getRadioValue(`t${trial.trial_index}_preference`)) {
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

function disableInputs() {
  document.querySelectorAll("select, input, button").forEach(el => {
    el.disabled = true;
  });

  document.querySelectorAll("audio").forEach(audio => {
    audio.controls = false;
  });
}

function releaseBlobUrls() {
  audioBlobUrls.forEach(blobUrl => {
    URL.revokeObjectURL(blobUrl);
  });

  audioBlobUrls.clear();
}

function secondsBetween(startMs, endMs) {
  return round2((endMs - startMs) / 1000);
}

function round2(x) {
  return Math.round(Number(x || 0) * 100) / 100;
}

function getSelectValue(name) {
  const el = document.querySelector(`select[name="${name}"]`);
  return el ? el.value : "";
}

function getRadioValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : "";
}

function htmlToNode(html) {
  const div = document.createElement("div");
  div.innerHTML = html.trim();
  return div.firstElementChild;
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