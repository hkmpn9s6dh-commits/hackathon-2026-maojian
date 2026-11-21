"""两遍 loudnorm：−16 LUFS / 真峰 ≤ −1.5 dBTP，输出 48k 立体声母版。

为什么是 −16：这条片子会同时投比赛和发抖音/视频号，平台归一化在 −14 到 −16 之间；
−16 留 2 dB 余量，手机上不会糊，评审用监听耳机也不会被推爆。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "out" / "audio.wav"
DST = ROOT / "out" / "audio-master.wav"
TARGET_I, TARGET_TP, TARGET_LRA = -16.0, -1.5, 11.0


def run(args: list[str]) -> str:
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(p.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"ffmpeg 失败：{' '.join(args[:6])}…")
    return p.stderr


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"缺 {SRC}（先跑 py -3 audio.py）")
    # 不做额外压缩：混音的峰均比在源头已经用软拐点控住了（见 audio.py）。
    # 这里只做响度归一；一旦挂 acompressor，LRA 会掉到 2.6 LU，撞击声会「闷」。
    COMP = "anull"
    # 第一遍：只测
    log = run(["ffmpeg", "-hide_banner", "-i", str(SRC), "-af",
               f"{COMP},loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json",
               "-f", "null", "-"])
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", log, re.S)
    if not m:
        raise SystemExit("拿不到 loudnorm 测量结果")
    d = json.loads(m.group(0))
    print(f"pass1: I={d['input_i']} TP={d['input_tp']} LRA={d['input_lra']} thresh={d['input_thresh']}")

    # 第二遍：按测量值线性归一
    af = (f"{COMP},loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:"
          f"measured_I={d['input_i']}:measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}:"
          f"measured_thresh={d['input_thresh']}:offset={d['target_offset']}:linear=true")
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(SRC), "-af", af,
         "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(DST)])
    print("wrote", DST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
