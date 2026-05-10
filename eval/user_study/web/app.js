const MATERIALS_PATH = "../data/materials.json";

async function main() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("token");

  if (!token) {
    document.getElementById("app").innerHTML = "缺少 token，请使用专属链接进入。";
    return;
  }

  const res = await fetch(MATERIALS_PATH);
  const materials = await res.json();

  const user = materials.find(x => x.token === token);

  if (!user) {
    document.getElementById("app").innerHTML = "token 无效。";
    return;
  }

  document.getElementById("user-info").innerText =
    `当前用户：${user.token}`;

  const app = document.getElementById("app");
  app.innerHTML = "";

  user.trials.forEach(trial => {
    const div = document.createElement("div");
    div.className = "trial";

    div.innerHTML = `
      <h2>第 ${trial.trial_index} 轮</h2>
      <p class="query">${trial.query_text}</p>

      <div class="methods">
        ${renderMethod("A", trial.A)}
        ${renderMethod("B", trial.B)}
        ${renderMethod("C", trial.C)}
      </div>
    `;

    app.appendChild(div);
  });
}

function renderMethod(label, data) {
  return `
    <div class="method-card">
      <h3>方法 ${label}</h3>
      <p class="hidden-method">真实方法：${data.method}</p>
      <p>推荐歌曲稍后填入。</p>
    </div>
  `;
}

main();