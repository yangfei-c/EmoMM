from application.core import Emo2Music
import numpy as np

emo = Emo2Music()

music_va = np.array([
    row["semantic_vector_28d"][:2]
    for row in emo.rows
    if row.get("semantic_vector_28d") is not None and len(row["semantic_vector_28d"]) >= 28
], dtype=float)

print("music VA min:", music_va.min(axis=0))
print("music VA max:", music_va.max(axis=0))

tests = [
    "我很焦虑，很紧绷",
    "我很低落，很疲惫",
    "我很开心，很兴奋",
    "我现在很平静"
]
for t in tests:
    z = emo.infer_z(t)
    print(t, "->", z)