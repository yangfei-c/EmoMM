const MATERIALS_PATH = "data/materials.json";
const START_API = "/api/start";
const SUBMIT_API = "/api/submit";

const RATING_FIELD = "regulation_suitability";
const RATING_PROMPT = "这首音乐适合帮助我从当前情绪调节到目标状态。";
const AUDIO_WARM_CONCURRENCY = 2;

let currentUser = null;
let activeAudio = null;
let warmed = new Set();

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
    const res = await fetch(START_API, { method: "POST" });
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
    const res = await fetch(MATERIALS_PATH, { cache: "no-cache" });
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
    if (userInfo) userInfo.innerText = `当前用户：${currentUser.token}`;

    renderPage();
    warmAllAudio();
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
        你将看到 <strong>3 个情绪场景</strong>。每一轮包含方法 A、B、C，
        每个方法提供 <strong>1 段约 15 秒音乐</strong>。
      </p>
      <p>
        请根据当前状态和调节目标，分别试听三首音乐，并从 1–7 分评价。
        1 表示非常不同意，7 表示非常同意。每轮最后请选择你最愿意使用的一首音乐。
      </p>
      <p>建议使用稳定网络和耳机。页面会在后台轻量准备音频，但不会自动播放。</p>
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
        <div id="submit-msg" class="submit-msg">请完成每一轮 A/B/C 三首音乐的评分和偏好选择。</div>
      </div>
      <button id="submit-btn" class="primary-submit">提交问卷</button>
    </div>
  `));

  document.getElementById("submit-btn").addEventListener("click", submitResponses);
  app.addEventListener("change", updateProgress);
  app.addEventListener("play", onAudioPlay, true);

  updateProgress();
}

function renderTrial(trial) {
  return `
    <div class="trial">
      <div class="trial-header">
        <h2>第 ${trial.trial_index} 轮</h2>
        <span class="strategy-badge">调节目标：${escapeHtml(trial.strategy_zh || trial.strategy || "情绪调节")}</span>
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
            ? `<audio controls preload="metadata" class="study-audio" src="${escapeAttribute(audioUrl)}"></audio>`
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

function onAudioPlay(e) {
  if (!e.target.classList.contains("study-audio")) return;

  if (activeAudio && activeAudio !== e.target) activeAudio.pause();

  document.querySelectorAll("audio").forEach(audio => {
    if (audio !== e.target) audio.pause();
  });

  activeAudio = e.target;
}

async function warmAllAudio() {
  const urls = [];

  for (const trial of currentUser.trials) {
    for (const label of ["A", "B", "C"]) {
      const item = trial[label].items[0];
      const url = item.audio_url || item.public_audio_url || item.audio_path || "";
      if (url && !urls.includes(url)) urls.push(url);
    }
  }

  let index = 0;

  async function worker() {
    while (index < urls.length) {
      const url = urls[index++];
      await warmAudio(url);
    }
  }

  const count = Math.min(AUDIO_WARM_CONCURRENCY, urls.length);
  await Promise.all(Array.from({ length: count }, worker));
}

async function warmAudio(url) {
  if (warmed.has(url)) return;

  try {
    const res = await fetch(url, { method: "GET", mode: "cors", cache: "force-cache" });
    if (res.ok) warmed.add(url);
  } catch (e) {
    console.warn("音频预热失败，播放器仍会直接加载:", url);
  }
}

function collectResponse() {
  return {
    token: currentUser.token,
    participant_index: currentUser.participant_index,
    submitted_at: new Date().toISOString(),
    version: "v2_single_song_single_rating",
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
      headers: { "Content-Type": "application/json" },
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
      if (getSelectValue(`t${trial.trial_index}_${label}_${RATING_FIELD}`)) done += 1;
    }

    if (getRadioValue(`t${trial.trial_index}_preference`)) done += 1;
  }

  const percent = Math.round((done / total) * 100);

  document.getElementById("progress-text").innerText = `${percent}%（${done}/${total}）`;
  document.getElementById("progress-fill").style.width = `${percent}%`;

  const msg = document.getElementById("submit-msg");
  if (percent === 100 && msg && !msg.classList.contains("success")) {
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