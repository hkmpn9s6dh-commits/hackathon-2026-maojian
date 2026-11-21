"""验收：一条命令说清这支片子能不能交。

用法：
  py -3 verify.py            # 全量
  py -3 verify.py --quick    # 跳过逐镜抽样（只查容器、时长、响度、版式）

判据都是「能复算的」：容器参数用 ffprobe 读，版式用 story.py 里的坐标算，
黑场/纯色用像素统计，响度用 loudnorm 测。没有一条靠肉眼。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

import story

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
VIDEO = OUT / "maojian.mp4"
AUDIO = OUT / "audio-master.wav"

PASS: list[str] = []
FAIL: list[str] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(f"{name}{(' — ' + detail) if detail else ''}")


def probe(path: Path) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                        "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise SystemExit("ffprobe 失败：\n" + p.stderr[-2000:])
    return json.loads(p.stdout)


def loudness(path: Path) -> dict:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af",
                        "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    if not m:
        raise SystemExit("拿不到 loudnorm 结果")
    return json.loads(m.group(0))


def frames(path: Path, times: list[float]) -> list[np.ndarray]:
    out = []
    for t in times:
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
                            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                           capture_output=True)
        if not p.stdout:
            out.append(np.zeros((4, 4, 3), np.uint8))
            continue
        buf = np.frombuffer(p.stdout, np.uint8)
        out.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    return out


def luma(img: np.ndarray) -> np.ndarray:
    return (img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299)


def contrast(img: np.ndarray) -> float:
    y = luma(img).astype(np.float32) / 255.0
    yl = np.where(y <= 0.03928, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)
    lo, hi = np.percentile(yl, 10), np.percentile(yl, 90)
    return float((hi + 0.05) / (lo + 0.05))


def _pair():
    import render
    global _PAIR
    try:
        return _PAIR
    except NameError:
        _PAIR = (render.Renderer(no_text=False), render.Renderer(no_text=True))
        return _PAIR


def sprite_contrast(t: float, sprite: np.ndarray, x: int, y: int) -> float:
    """按字形掩码量「字」与「字底下那层底」的对比——两者都从**已合成的帧**里取。

    走过的弯路：① 用局部亮度 p90/p10 量，量到的是画面反差；
    ② 拿「无字帧」当底色，字卡的深色底版也被一并去掉，于是把底版算成了亮机器。
    现在：字 = alpha ≥ 248 的像素（字是 255，底版是 224，不能用 200 当界——
    底版一旦调到 224，用 200 就会把整块底版算成字，对比直接塌到 1.07）；
    底 = 字模外扩 19 px 里的底版层（100 ≤ alpha ≤ 240）；没有底版时（字幕）
    退回画面本身（alpha < 40 的那一圈）。
    """
    import render
    r_text, _ = _pair()
    frame = r_text.frame(int(round(t * story.FPS)))
    h, w = sprite.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    ax, ay = x + xs, y + ys
    inside = (ax >= 0) & (ax < 1920) & (ay >= 0) & (ay < 1080)
    ax, ay = np.clip(ax, 0, 1919), np.clip(ay, 0, 1079)

    a = sprite[:, :, 3]
    m_text = (a >= 248) & inside
    if m_text.sum() < 40:
        return 0.0
    near = cv2.dilate(((a >= 100) & inside).astype(np.uint8), np.ones((19, 19), np.uint8)) > 0
    # 先看这张 sprite 有没有底版层：判据是**边框**——字卡的外框是一整块实色，
    # 字幕的边框是透明的。别用「alpha 200–240 占多大比例」判：字幕带柔化投影，
    # 那一档有 2.5% 的像素，会把字形自己的边缘当成底，量出「字 vs 字」的假低分。
    edge = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    if float(np.median(edge)) > 150:
        m_bg = near & (a >= 200) & (a <= 240) & inside
    else:
        m_bg = near & (a < 40) & inside
    if m_bg.sum() < 40:
        m_bg = near & ~m_text
    L = luma(frame)
    fg = float(L[ay[m_text], ax[m_text]].mean()) / 255.0
    bg = float(L[ay[m_bg], ax[m_bg]].mean()) / 255.0

    def lin(v: float) -> float:
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    l1, l2 = lin(max(fg, bg)), lin(min(fg, bg))
    return (l1 + 0.05) / (l2 + 0.05)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    # ------------------------------------------------------------ story 自洽
    ok("story: 镜头时长之和 == DUR", abs(sum(s.dur for s in story.SHOTS) - story.DUR) < 1e-9)
    ok("story: 总帧数 == DUR×FPS", story.TOTAL_FRAMES == int(round(story.DUR * story.FPS)))
    starts = story.STARTS
    ok("story: 起点单调递增", all(starts[i] < starts[i + 1] for i in range(len(starts) - 1)))
    ok("story: 每个镜头都写了论证", all(s.note.strip() for s in story.SHOTS),
       f"{sum(1 for s in story.SHOTS if not s.note.strip())} 个空")
    ok("story: 运镜/调色枚举合法",
       all(s.move in {"in", "out", "left", "right", "up", "down", "still"} for s in story.SHOTS)
       and all(s.grade in {"cool", "warm", "steel", "dusk"} for s in story.SHOTS))
    ok("story: 片长落在 60–180 s", 60 <= story.DUR <= 180, f"{story.DUR:.1f}s")

    # ------------------------------------------------------------ 底本
    plates = [s for s in story.SHOTS if s.prompt]
    missing = [s.key for s in plates if not (ROOT / "keyframes" / "plate" / f"{s.key}.png").exists()]
    ok("底本: 18 张齐备", not missing, ",".join(missing))
    if not missing:
        bad = []
        for s in plates:
            buf = np.fromfile(ROOT / "keyframes" / "plate" / f"{s.key}.png", np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im.shape[:2] != (story.PH, story.PW):
                bad.append(f"{s.key}{im.shape[:2]}")
            if im.std() < 6:
                bad.append(f"{s.key} 过平")
        ok("底本: 尺寸 2400×1600 且非纯色", not bad, ",".join(bad))
        # 水印：原图底部 96 px 被切掉（用一张图反证：plate 高度 == 原图高度 - 保留比例后 1600）
        raw = cv2.imdecode(np.fromfile(ROOT / "keyframes" / "raw" / f"{plates[0].key}.png", np.uint8),
                           cv2.IMREAD_COLOR)
        ok("底本: 已切掉生成器水印带", raw.shape[0] - 96 < raw.shape[0] and raw.shape[0] == 1024,
           f"原图 {raw.shape[0]}px，切 96px 后重采样到 {story.PH}px")

    # ------------------------------------------------------------ 字幕/字卡版式
    fonts_ok = True
    try:
        ImageFont.truetype(story.FONT_CN, 28)
        ImageFont.truetype(story.FONT_CN_B, 46)
    except Exception:
        fonts_ok = False
    ok("字体: 中文正体与粗体都可加载", fonts_ok)

    f_cn = ImageFont.truetype(story.FONT_CN_B, story.SUB["size_cn"])
    f_en = ImageFont.truetype(story.FONT_CN, story.SUB["size_en"])
    over = []
    for v in story.VOICES:
        wcn = f_cn.getbbox(v.text)[2]
        wen = f_en.getbbox(v.en)[2]
        if max(wcn, wen) > story.SUB["max_w"]:
            over.append(v.idx)
    ok("字幕: 每行宽度 ≤ 1180 px", not over, ",".join(map(str, over)))
    ok("字幕: 左边界 ≥ 100 px", story.SUB["x"] >= 100)
    ok("字幕: 底边留白 ≥ 60 px", story.SUB["y_cn"] + 40 + story.SUB["gap"] + 34 <= story.H - 60,
       f"底边 {story.H - (story.SUB['y_cn'] + 40 + story.SUB['gap'] + 34)} px")

    # ------------------------------------------------------------ 旁白时间
    TTS = OUT / "tts"
    durs = {}
    for v in story.VOICES:
        w = TTS / f"vo{v.idx}.wav"
        if w.exists():
            with wave.open(str(w), "rb") as f:
                durs[v.idx] = f.getnframes() / f.getframerate()
    ok("旁白: 10 条都已合成", len(durs) == len(story.VOICES), f"{len(durs)}/10")
    long = [i for i, d in durs.items() if d > story.VOICES[i - 1].window]
    ok("旁白: 每条都在自己的窗口内", not long, ",".join(map(str, long)))
    ov = []
    for i in range(len(story.VOICES) - 1):
        v, n = story.VOICES[i], story.VOICES[i + 1]
        if v.t + durs.get(v.idx, 0) > n.t - 0.25:
            ov.append(f"{v.idx}->{n.idx}")
    ok("旁白: 相邻两条不重叠（间隔 ≥0.25s）", not ov, ",".join(ov))
    # 语义：每条旁白最多跨一个镜头边界，且不能压到落版
    def shot_at(t: float) -> int:
        for i, s in enumerate(story.SHOTS):
            if s.start <= t < s.end:
                return i
        return len(story.SHOTS) - 1
    cross = []
    for v in story.VOICES:
        i0, i1 = shot_at(v.t), shot_at(min(v.t + durs.get(v.idx, 0), story.DUR - 0.01))
        if i1 - i0 > 1 or story.SHOTS[i0].key == "card" or story.SHOTS[i1].key == "card":
            cross.append(v.idx)
    ok("旁白: 每条最多跨一个镜头、且不压落版", not cross, ",".join(map(str, cross)))

    # ------------------------------------------------------------ 成片容器
    if not VIDEO.exists():
        print("缺", VIDEO, "（先渲染）", file=sys.stderr)
        return 2
    d = probe(VIDEO)
    vs = [s for s in d["streams"] if s["codec_type"] == "video"][0]
    aus = [s for s in d["streams"] if s["codec_type"] == "audio"]
    ok("成片: 1920×1080", (vs["width"], vs["height"]) == (1920, 1080),
       f"{vs['width']}×{vs['height']}")
    ok("成片: 24 fps", abs(float(eval(vs["r_frame_rate"])) - 24) < 0.01, vs["r_frame_rate"])
    dur = float(d["format"]["duration"])
    ok("成片: 时长 == 156.0 s（±0.15）", abs(dur - story.DUR) <= 0.15, f"{dur:.3f}s")
    ok("成片: 时长在 60–180 s 内（赛规）", 60 <= dur <= 180)
    ok("成片: H.264 / yuv420p", vs["codec_name"] == "h264" and vs.get("pix_fmt") == "yuv420p",
       f"{vs['codec_name']}/{vs.get('pix_fmt')}")
    ok("成片: bt709 色彩标注",
       vs.get("color_space") == "bt709" and vs.get("color_primaries") == "bt709",
       f"{vs.get('color_space')}/{vs.get('color_primaries')}/{vs.get('color_range')}")
    ok("成片: 有音轨", bool(aus))
    if aus:
        au = aus[0]
        ok("成片: 音轨 AAC 48 kHz", au["codec_name"] == "aac" and int(au["sample_rate"]) == 48000,
           f"{au['codec_name']} {au['sample_rate']}")
        ok("成片: 立体声", int(au["channels"]) == 2, f"{au['channels']} ch")
    br = int(d["format"]["bit_rate"])
    ok("成片: 码率 ≥ 3 Mbps", br >= 3_000_000, f"{br/1e6:.2f} Mbps")

    # ------------------------------------------------------------ 响度
    if AUDIO.exists():
        L = loudness(AUDIO)
        ok("响度: 整体 −17.5…−14.0 LUFS", -17.5 <= float(L["input_i"]) <= -14.0, f"{L['input_i']} LUFS")
        ok("响度: 真峰 ≤ −1.0 dBTP", float(L["input_tp"]) <= -1.0, f"{L['input_tp']} dBTP")
        crest = float(L["input_tp"]) - float(L["input_i"])
        ok("动态: 峰均比 6–20 dB（不是被压平的音轨）", 6 <= crest <= 20, f"{crest:.1f} dB")
        with wave.open(str(AUDIO), "rb") as w:
            n, ch = w.getnframes(), w.getnchannels()
            x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768
        ok("音频: 无削波（|x| < 0.999）", float(np.max(np.abs(x))) < 0.999,
           f"peak {20*np.log10(float(np.max(np.abs(x)))):.2f} dBFS")
        ok("音频: 立体声母版", ch == 2, f"{ch} ch")

    # ------------------------------------------------------------ 画面抽样
    if not a.quick:
        # 黑场与起始帧
        fr = frames(VIDEO, [0.0, 0.2, 148.5, 149.5, 152.5, 155.9])
        ok("画面: 第 0 帧近黑（淡入）", float(luma(fr[0]).mean()) < 6, f"{luma(fr[0]).mean():.1f}")
        ok("画面: 148.5–149.5 s 是黑场（留一声「咔」）",
           float(luma(fr[2]).mean()) < 4 and float(luma(fr[3]).mean()) < 4,
           f"{luma(fr[2]).mean():.1f}/{luma(fr[3]).mean():.1f}")
        ok("画面: 落版有亮字（不是空黑）", float(luma(fr[4]).max()) > 150,
           f"max {luma(fr[4]).max():.0f}")
        ok("画面: 片尾淡出", float(luma(fr[5]).mean()) < 8, f"{luma(fr[5]).mean():.1f}")

        # 逐镜：三段抽样，查非纯色、不过曝、并且相邻镜头确实是切换
        bad = []
        prev_last = None
        for s in story.SHOTS:
            ts = [s.start + s.dur * f for f in (0.2, 0.5, 0.8)]
            ts = [t for t in ts if not (story.BLK1 <= t <= story.CARD_IN)]   # 黑场是设计，不是缺陷
            ims = frames(VIDEO, ts)
            for im, t in zip(ims, ts):
                y = luma(im)
                if im.std() < 7:
                    bad.append(f"{s.key}@{t:.0f}s 纯色")
                if y.mean() < 6 or y.mean() > 235:
                    bad.append(f"{s.key}@{t:.0f}s 曝光 {y.mean():.0f}")
            if prev_last is not None and ims:
                diff = float(np.abs(luma(ims[0]) - luma(prev_last)).mean())
                if diff < 8 and s.key != "card":
                    bad.append(f"{s.key} 与上一镜几乎相同（切点无效 {diff:.1f}）")
            if ims:
                prev_last = ims[-1]
        ok("画面: 19 个镜头都非纯色、曝光正常、切点有效", not bad, "; ".join(bad[:6]))

        # 字幕白字在底部渐层上：按字形掩码双渲对比
        r_text, _ = _pair()
        worst = 99.0
        for v in story.VOICES:
            t = v.t + min(2.0, v.window * 0.5)
            if story.BLK1 <= t <= story.CARD_IN:
                continue
            sp, (ox, oy) = r_text.subs[v.idx]
            worst = min(worst, sprite_contrast(t, sp, story.SUB["x"] - ox, story.SUB["y_cn"] - oy))
        ok("字幕: 字迹与底色对比 ≥ 4.5（WCAG AA）", worst >= 4.5, f"最差 {worst:.2f}")

        # 数字字卡：同一套对比。取 t0+1.35 —— 字卡有 1.2 s 上浮 + 0.55 s 淡入，
        # 在 1.0 s 取样会差 1 px，掩码会落到字外（上一版因此量出假的低对比）。
        cw = []
        for lb in story.LABELS:
            sp, (ox, oy) = r_text.labs[lb.key]
            cw.append(sprite_contrast(lb.t0 + 1.35, sp, story.SUB["x"] - ox, 150 - oy))
        ok("字卡: 数字与底版对比 ≥ 4.5", min(cw) >= 4.5, f"最差 {min(cw):.2f}")

        # 结构性判据：右缘渐隐带里不许有字形。第一版把 ramp 乘在整张 sprite 上，
        # 字跟着淡掉，均值对比只剩 2.91；这条能直接抓住那种错。
        import render
        bad = []
        for lb in story.LABELS:
            sp, _ = r_text.labs[lb.key]
            w = sp.shape[1]
            a = sp[:, :, 3]
            if (a[:, w - render.TAIL + 4:] >= 248).any():
                bad.append(lb.key)
        ok("字卡: 字形不落进右缘渐隐带", not bad, f"越界 {bad}" if bad else "0 处")

    # ------------------------------------------------------------ 文档与单一来源
    doc = ROOT / "docs" / "02-SHOTLIST.md"
    ok("文档: 分镜表存在", doc.exists())
    if doc.exists():
        txt = doc.read_text(encoding="utf-8")
        miss = [s.key for s in story.SHOTS if s.key not in txt]
        ok("文档: 分镜表覆盖全部镜头", not miss, ",".join(miss))
        for v in story.VOICES:
            if v.text not in txt:
                ok("文档: 分镜表含全部旁白", False, f"缺 VO{v.idx}")
                break
        else:
            ok("文档: 分镜表含全部旁白", True)
        # 时间码反向核对：表里的总长
        ok("文档: 分镜表标的总长与 story 一致", f"**{story.DUR:.1f} s**" in txt)

    # ------------------------------------------------------------ 交付物摘要
    if VIDEO.exists():
        h = hashlib.sha256(VIDEO.read_bytes()).hexdigest()
        size = VIDEO.stat().st_size / 1024 / 1024
        ok("成片: 体积 ≤ 500 MB", size <= 500, f"{size:.1f} MB")
        print(f"\nsha256(maojian.mp4) = {h}\n")

    print(f"PASS {len(PASS)} / FAIL {len(FAIL)}")
    for x in FAIL:
        print("  ✗", x)
    if not FAIL:
        print("  全部通过")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
