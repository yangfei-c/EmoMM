# 音乐语义库导出结果

## 一、音乐语义库结构
说明每首音乐由 2 维 Music VA 预测值和 26 维 Music Tag 预测概率构成，共 28 维情绪—语义表示。
本次导出成功 49553 条，跳过 0 条（缺少 VA 或 26 维 Tag）。

## 二、26 个具体标签

| 序号 | 字段名 | 标签名 |
| --- | --- | --- |
| 1 | tag_ambient | ambient |
| 2 | tag_classical | classical |
| 3 | tag_electronic | electronic |
| 4 | tag_folk | folk |
| 5 | tag_hiphop | hiphop |
| 6 | tag_jazz | jazz |
| 7 | tag_metal | metal |
| 8 | tag_pop | pop |
| 9 | tag_rock | rock |
| 10 | tag_soundtrack | soundtrack |
| 11 | tag_bass | bass |
| 12 | tag_drums | drums |
| 13 | tag_guitar | guitar |
| 14 | tag_piano | piano |
| 15 | tag_strings | strings |
| 16 | tag_synthesizer | synthesizer |
| 17 | tag_voice | voice |
| 18 | tag_dark | dark |
| 19 | tag_dream | dream |
| 20 | tag_energetic | energetic |
| 21 | tag_epic | epic |
| 22 | tag_film | film |
| 23 | tag_happy | happy |
| 24 | tag_love | love |
| 25 | tag_relaxing | relaxing |
| 26 | tag_sad | sad |

## 三、语义库样例

| song_id | valence | arousal | top5_tags |
| --- | --- | --- | --- |
| track_0484375 | 0.343872 | 0.212545 | electronic,pop,synthesizer,ambient,bass |
| track_0484381 | 0.284878 | 0.365397 | electronic,synthesizer,bass,pop,drums |
| track_0484382 | 0.506637 | 0.71801 | electronic,synthesizer,pop,bass,drums |
| track_0484383 | 0.633125 | 0.81062 | electronic,pop,synthesizer,bass,rock |
| track_0484384 | 0.60373 | 0.692173 | electronic,bass,synthesizer,drums,pop |
| track_0484385 | 0.335042 | 0.547942 | electronic,pop,synthesizer,bass,drums |
| track_0484386 | 0.4041 | 0.36201 | electronic,bass,pop,synthesizer,drums |
| track_0484387 | 0.218721 | 0.303021 | electronic,ambient,synthesizer,pop,bass |
| track_0484388 | 0.599635 | 0.70827 | electronic,synthesizer,bass,drums,pop |
| track_0484389 | 0.449625 | 0.608491 | electronic,synthesizer,bass,pop,drums |
