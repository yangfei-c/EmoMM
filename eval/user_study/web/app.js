const MATERIALS_PATH = "data/materials.json";
const START_API = "/api/start";
const SUBMIT_API = "/api/submit";

let currentUser = null;

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
  app.innerHTML = "正在为你分配实验任务，请稍候...";

  try {
    const res = await fetch(START_API, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      }
    });

    const data = await res.json();

    if (!res.ok) {
      app.innerHTML = data.error || "实验名额已满或分配失败，请联系研究者。";
      return;
    }

    const token = encodeURIComponent(data.token);
    window.location.href = `${window.location.origin}${window.location.pathname}?token=${token}`;
  } catch (err) {
    console.error(err);
    app.innerHTML = "无法连接服务器，请稍后重试。";
  }
}

async function loadUserByToken(token) {
  const app = document.getElementById("app");

  try {
    const res = await fetch(MATERIALS_PATH);
    const materials = await res.json();

    currentUser = materials.find(x => x.token === token);

    if (!currentUser) {
      app.innerHTML = "token 无效，请联系研究者。";
      return;
    }

    document.getElementById("user-info").innerText = `当前用户：${currentUser.token}`;
    renderPage(currentUser);
  } catch (err) {
    console.error(err);
    app.innerHTML = "加载实验材料失败，请稍后重试。";
  }
}

function renderPage(user) {
  const app = document.getElementById("app");
  app.innerHTML = "";

  const intro = document.createElement("div");
  intro.className = "intro";
  intro.innerHTML = `
    <p>请根据每一轮情绪场景，分别试听方法 A、方法 B、方法 C 的推荐结果，并完成评分。</p>
    <p>评分范围为 1–7 分，1 表示非常不同意，7 表示非常同意。</p>
  `;
  app.appendChild(intro);

  user.trials.forEach(trial => {
    const div = document.createElement("div");
    div.className = "trial";

    div.innerHTML = `
      <h2>第 ${trial.trial_index} 轮</h2>
      <p class="query">${escapeHtml(trial.query_text)}</p>

      <div class="methods">
        ${renderMethod(trial.trial_index, "A", trial.A)}
        ${renderMethod(trial.trial_index, "B", trial.B)}
        ${renderMethod(trial.trial_index, "C", trial.C)}
      </div>

      <div class="preference-box">
        <h3>本轮你最喜欢哪一组推荐？</h3>
        ${renderPreference(trial.trial_index)}
      </div>
    `;

    app.appendChild(div);
  });

  const submitBox = document.createElement("div");
  submitBox.className = "submit-box";
  submitBox.innerHTML = `
    <button id="submit-btn">提交问卷</button>
    <p id="submit-msg"></p>
  `;
  app.appendChild(submitBox);

  document.getElementById("submit-btn").addEventListener("click", submitResponses);
}

function renderMethod(trialIndex, label, data) {
  return `
    <div class="method-card">
      <h3>方法 ${label}</h3>

      <div class="songs">
        ${renderSongs(data.items)}
      </div>

      <div class="ratings">
        ${renderRating(trialIndex, label, "regulation_fit", "这组推荐符合我想要的情绪调节方向")}
        ${renderRating(trialIndex, label, "helpfulness", "这组推荐能帮助我调节当前情绪")}
        ${renderRating(trialIndex, label, "satisfaction", "我对这组推荐整体满意")}
      </div>
    </div>
  `;
}

function renderSongs(items) {
  if (!items || items.length === 0) {
    return `<p class="placeholder">推荐歌曲稍后填入。</p>`;
  }

  return `
    <ol class="song-list">
      ${items.map((item, i) => `
        <li>
          <p class="song-title">歌曲 ${i + 1}</p>
          ${renderAudio(item)}
        </li>
      `).join("")}
    </ol>
  `;
}

function renderAudio(item) {
  const src = item.audio_url || item.public_audio_url || item.audio_path || "";

  if (!src) {
    return `<p class="placeholder">暂无音频链接</p>`;
  }

  return `
    <audio controls preload="none">
      <source src="${escapeAttribute(src)}">
      你的浏览器不支持音频播放。
    </audio>
  `;
}

function renderRating(trialIndex, methodLabel, field, text) {
  const name = `t${trialIndex}_${methodLabel}_${field}`;

  return `
    <label class="rating-row">
      <span>${text}</span>
      <select name="${name}" required>
        <option value="">请选择</option>
        <option value="1">1</option>
        <option value="2">2</option>
        <option value="3">3</option>
        <option value="4">4</option>
        <option value="5">5</option>
        <option value="6">6</option>
        <option value="7">7</option>
      </select>
    </label>
  `;
}

function renderPreference(trialIndex) {
  return `
    <label>
      <input type="radio" name="t${trialIndex}_preference" value="A">
      方法 A
    </label>
    <label>
      <input type="radio" name="t${trialIndex}_preference" value="B">
      方法 B
    </label>
    <label>
      <input type="radio" name="t${trialIndex}_preference" value="C">
      方法 C
    </label>
  `;
}

async function submitResponses() {
  const response = collectResponse();
  const error = validateResponse(response);
  const msg = document.getElementById("submit-msg");
  const btn = document.getElementById("submit-btn");

  if (error) {
    msg.innerText = error;
    return;
  }

  btn.disabled = true;
  msg.innerText = "正在提交，请稍候...";

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
      btn.disabled = false;
      return;
    }

    msg.innerText = "提交成功，感谢参与！";
  } catch (err) {
    console.error(err);
    msg.innerText = "提交失败：无法连接服务器。";
    btn.disabled = false;
  }
}

function collectResponse() {
  const response = {
    token: currentUser.token,
    participant_index: currentUser.participant_index,
    submitted_at: new Date().toISOString(),
    trials: []
  };

  for (const trial of currentUser.trials) {
    const trialResult = {
      trial_index: trial.trial_index,
      query_id: trial.query_id,
      strategy: trial.strategy,
      preference: getRadioValue(`t${trial.trial_index}_preference`),
      ratings: {}
    };

    for (const label of ["A", "B", "C"]) {
      trialResult.ratings[label] = {
        regulation_fit: getSelectValue(`t${trial.trial_index}_${label}_regulation_fit`),
        helpfulness: getSelectValue(`t${trial.trial_index}_${label}_helpfulness`),
        satisfaction: getSelectValue(`t${trial.trial_index}_${label}_satisfaction`)
      };
    }

    response.trials.push(trialResult);
  }

  return response;
}

function validateResponse(response) {
  for (const trial of response.trials) {
    for (const label of ["A", "B", "C"]) {
      const r = trial.ratings[label];

      if (!r.regulation_fit || !r.helpfulness || !r.satisfaction) {
        return `请完成第 ${trial.trial_index} 轮方法 ${label} 的所有评分。`;
      }
    }

    if (!trial.preference) {
      return `请选择第 ${trial.trial_index} 轮最喜欢的方法。`;
    }
  }

  return "";
}

function getSelectValue(name) {
  const el = document.querySelector(`select[name="${name}"]`);
  return el ? el.value : "";
}

function getRadioValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : "";
}

function escapeHtml(text) {
  return String(text)
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