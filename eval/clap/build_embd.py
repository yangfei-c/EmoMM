import sys
from pathlib import Path
import json
import numpy as np
from msclap import CLAP

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir / "src"))

from core.music_repository import load_tracks


OUT_DIR = Path(r"D:\AAAA_audio_dataset\eval\clap")
OUT_NPZ = OUT_DIR / "audio_embd.npz"
TEXT_NPZ = OUT_DIR / "text_embd.npz"
QUERY_JSON = root_dir / "eval" / "queries_zh16.json"

save_every = 100


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_existing():
    if not OUT_NPZ.exists():
        return [], [], []

    data = np.load(OUT_NPZ, allow_pickle=True)

    song_ids = [str(x) for x in data["song_ids"]]
    audio_paths = [str(x) for x in data["audio_paths"]]
    audio_emb = [x.astype(np.float32) for x in data["audio_emb"]]

    return song_ids, audio_paths, audio_emb


def save_npz(song_ids, audio_paths, audio_emb):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not audio_emb:
        return

    np.savez_compressed(
        OUT_NPZ,
        song_ids=np.asarray(song_ids, dtype=object),
        audio_paths=np.asarray(audio_paths, dtype=object),
        audio_emb=np.stack(audio_emb).astype(np.float32),
    )


def load_queries():
    return json.loads(QUERY_JSON.read_text(encoding="utf-8"))


def q_id(q):
    return str(q.get("query_id") or q.get("id"))


def query_text(q):
    text = str(q.get("text") or "").strip()
    if text:
        return text

    text = f"{q.get('current_state', '')}{q.get('regulation_goal', '')}".strip()
    if not text:
        raise ValueError(f"{q_id(q)} 缺少 text")
    return text


def need_rebuild_text(query_ids, query_texts):
    if not TEXT_NPZ.exists():
        return True

    data = np.load(TEXT_NPZ, allow_pickle=True)

    if "query_ids" not in data or "query_texts" not in data or "text_emb" not in data:
        return True

    old_ids = [str(x) for x in data["query_ids"]]
    old_texts = [str(x) for x in data["query_texts"]]

    return old_ids != query_ids or old_texts != query_texts


def build_text_emb(model, query_ids, query_texts):
    vecs = []

    for i, text in enumerate(query_texts, start=1):
        emb = model.get_text_embeddings([text])
        vecs.append(to_numpy(emb).astype(np.float32))
        print(f"text embedding: {i}/{len(query_texts)}")

    text_emb = np.concatenate(vecs, axis=0)

    TEXT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TEXT_NPZ,
        query_ids=np.asarray(query_ids, dtype=object),
        query_texts=np.asarray(query_texts, dtype=object),
        text_emb=text_emb,
    )


def ensure_text_emb(model):
    queries = load_queries()
    query_ids = [q_id(q) for q in queries]
    query_texts = [query_text(q) for q in queries]

    if need_rebuild_text(query_ids, query_texts):
        print("开始构建文本 embedding...")
        build_text_emb(model, query_ids, query_texts)
    else:
        print("文本 embedding 已存在且匹配，跳过构建。")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks()

    items = []
    for t in tracks:
        song_id = str(t["song_id"])
        audio_path = str(t.get("song_path", ""))

        if audio_path and Path(audio_path).exists():
            items.append((song_id, audio_path))

    song_ids, audio_paths, audio_emb = load_existing()
    done = set(song_ids)

    print(f"全部音乐数量: {len(items)}")
    print(f"已完成数量: {len(done)}")
    print(f"输出文件: {OUT_NPZ}")
    print("加载 CLAP 模型...")

    model = CLAP(version="2023", use_cuda=True)

    ensure_text_emb(model)

    for idx, (song_id, audio_path) in enumerate(items, start=1):
        if song_id in done:
            continue

        try:
            emb = model.get_audio_embeddings([audio_path])
            emb = to_numpy(emb).astype(np.float32).squeeze()

            if emb.ndim != 1:
                print(f"[跳过] embedding 维度异常: {song_id}, shape={emb.shape}")
                continue

            song_ids.append(song_id)
            audio_paths.append(audio_path)
            audio_emb.append(emb)
            done.add(song_id)

        except Exception as e:
            print(f"[失败] {idx}/{len(items)} {song_id} {audio_path}")
            print(repr(e))
            continue

        if len(song_ids) % save_every == 0:
            save_npz(song_ids, audio_paths, audio_emb)
            print(f"已保存: {len(song_ids)}/{len(items)}")

    save_npz(song_ids, audio_paths, audio_emb)

    print("完成 CLAP 全量 embedding")
    print(f"成功数量: {len(song_ids)}")
    print(f"保存位置: {OUT_NPZ}")


if __name__ == "__main__":
    main()