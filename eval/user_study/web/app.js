const MATERIALS_PATH = "data/materials.json";
const SUBMIT_API = "/api/submit";

let currentUser = null;

async function main() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("token");

  if (!token) {
    document.getElementById("app").innerHTML = "缺少 token，请使用专属链接进入。";
    return;
  }

  const res = await fetch(MATERIALS_PATH);
  const materials = await res.json();

  currentUser = materials.find(x => x.token === token);

  if (!currentUser) {
    document.getElementById("app").innerHTML = "token 无效。";
    return;
  }

  document.getElementById("user-info").innerText = `当前用户：${currentUser.token}`;
  renderPage(currentUser);
}

function renderPage(user) {
  const app = document.getElementById("app");
  app.innerHTML = "";

  user.trials.forEach(trial => {
    const div = document.createElement("div");
    div.className = "trial";

    div.innerHTML = `
      <h2>第 ${trial.trial_index} 轮</h2>
      <p class="query">${trial.query_text}</p>

      <div class="methods">
        ${renderMethod(trial.trial_index, "A", trial.A)}
        ${renderMethod(trial.trial_index, "B", trial.B)}
        ${renderMethod(trial.trial_index, "C", trial.C)}
      </div>

      <div class="preference-box">
        <h3>本轮你最喜欢哪一组？</h3>
        ${renderPreference(trial.trial_index)}
      </div>
    `;

    app.appendChild(div);
  });

  const submitBox = document.createElement("div");
  submitBox.className = "submit-box";
  submitBox.innerHTML = `
    <button id="submit-btn">提交</button>
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
        ${renderRating(trialIndex, label, "regulation_fit", "情绪调节匹配度")}
        ${renderRating(trialIndex, label, "helpfulness", "帮助程度")}
        ${renderRating(trialIndex, label, "satisfaction", "整体满意度")}
      </div>
    </div>
  `;
}

function renderSongs(items) {
  if (!items || items.length === 0) {
    return `<p>推荐歌曲稍后填入。</p>`;
  }

  return `
    <ol>
      ${items.map((item, i) => `
        <li>
          <p>歌曲 ${i + 1}</p>
          ${renderAudio(item)}
        </li>
      `).join("")}
    </ol>
  `;
}

function renderAudio(item) {
  const src = item.audio_url || item.public_audio_url || item.audio_path || "";

  if (!src) {
    return `<p>暂无音频链接</p>`;
  }

  return `
    <audio controls preload="none">
      <source src="${src}">
      你的浏览器不支持音频播放。
    </audio>
  `;
}

function renderRating(trialIndex, methodLabel, field, text) {
  const name = `t${trialIndex}_${methodLabel}_${field}`;

  return `
    <label>
      ${text}：
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
      ratings: {},
      preference: getRadioValue(`t${trial.trial_index}_preference`)
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

  const error = validateResponse(response);
  const msg = document.getElementById("submit-msg");

  if (error) {
    msg.innerText = error;
    return;
  }

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
      return;
    }

    msg.innerText = "提交成功，感谢参与！";
    document.getElementById("submit-btn").disabled = true;
  } catch (e) {
    msg.innerText = "提交失败：无法连接服务器。";
    console.error(e);
  }
}

function getSelectValue(name) {
  const el = document.querySelector(`select[name="${name}"]`);
  return el ? el.value : "";
}

function getRadioValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : "";
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

main();