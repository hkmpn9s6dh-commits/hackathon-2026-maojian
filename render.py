"""把 18 张底本渲成 156 秒《第五个轮子》。

用法：
  py -3 render.py --prep                 # 去水印、放大、分级缓存
  py -3 render.py --stills 5 40 120      # 出指定秒数的静帧（核对用）
  py -3 render.py --pipe                 # 直接喂 ffmpeg 出成片
  py -3 render.py --seg 1/4 --silent     # 分段（并行用）；--concat 拼接
  py -3 render.py --shotlist             # 生成 docs/02-SHOTLIST.md（单一来源反向核对）

原则：画面只由 story.py 决定；这里的每个函数都必须是纯函数（同输入同输出），
否则 verify.py 的重渲断言会失败。
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import story

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "keyframes" / "raw"
PLATE = ROOT / "keyframes" / "plate"
OUT = ROOT / "out"
FONTS = {
    "cn": story.FONT_CN_B,
    "cn_light": story.FONT_CN,
    "num": story.FONT_NUM,
    "serif": "C:/Windows/Fonts/Source Han Serif SC Heavy (TrueType).ttf",
}

W, H = story.W, story.H
FPS = story.FPS
PW, PH = story.PW, story.PH
PRESET = "fast"
THREADS = "3"
WATERMARK_H = 96          # 底部水印高度（原始 1024 高里的像素）


# ---------------------------------------------------------------- 基础工具
def imread_u(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不出图片：{path}")
    return img


def imwrite_u(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit(f"写不出图片：{path}")
    buf.tofile(str(path))


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease(x: float) -> float:
    """镜头运动用：头尾更慢，中间更匀。"""
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, x)))


# ---------------------------------------------------------------- 前处理
def prep_one(key: str) -> Path:
    dst = PLATE / f"{key}.png"
    if dst.exists():
        return dst
    img = imread_u(RAW / f"{key}.png")
    h, w = img.shape[:2]
    # 1) 切掉底部生成器水印
    img = img[: h - WATERMARK_H, :, :]
    # 2) 等比放大到高 = PH，再中心裁到 PW 宽（保住构图中心，两侧只裁一点）
    s = PH / img.shape[0]
    img = cv2.resize(img, (int(round(img.shape[1] * s)), PH), interpolation=cv2.INTER_CUBIC)
    if img.shape[1] < PW:
        pad = (PW - img.shape[1]) // 2 + 1
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_REFLECT)
    x0 = (img.shape[1] - PW) // 2
    img = img[:, x0:x0 + PW, :]
    # 3) 轻微锐化（放大后的补偿），不改变色
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    img = cv2.addWeighted(img, 1.28, blur, -0.28, 0)
    imwrite_u(dst, img)
    return dst


# ---------------------------------------------------------------- 分级
def grade(img: np.ndarray, kind: str) -> np.ndarray:
    """lift / gamma / gain + 一点 S 曲线。输入输出都是 BGR uint8。"""
    x = img.astype(np.float32) / 255.0
    # BGR
    if kind == "cool":
        lift = np.array([0.030, 0.016, 0.004], np.float32)     # 抬蓝、压红
        gain = np.array([1.03, 1.00, 0.98], np.float32)
        sat = 0.88
    elif kind == "steel":
        lift = np.array([0.012, 0.012, 0.010], np.float32)
        gain = np.array([1.00, 1.00, 1.00], np.float32)
        sat = 0.80
    elif kind == "warm":
        lift = np.array([0.006, 0.014, 0.028], np.float32)     # 抬红
        gain = np.array([0.99, 1.02, 1.06], np.float32)
        sat = 0.95
    elif kind == "dusk":
        lift = np.array([0.020, 0.010, 0.028], np.float32)
        gain = np.array([1.04, 0.99, 1.00], np.float32)
        sat = 0.92
    else:
        lift = np.zeros(3, np.float32)
        gain = np.ones(3, np.float32)
        sat = 1.0

    x = x * gain + lift
    # gamma
    x = np.clip(x, 0, 1) ** 1.04
    # S 曲线（只压两端，中间不动）
    x = x * x * (3 - 2 * x) * 0.35 + x * 0.65
    # 饱和度
    lum = (x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299)[..., None]
    x = lum + (x - lum) * sat
    # 冷部偏移：暗部再偏蓝一点，高光偏暖一点
    if kind in ("cool", "steel", "dusk"):
        w = np.clip(1.0 - lum * 2.2, 0, 1)[..., 0]
        x[..., 0] = np.clip(x[..., 0] + 0.012 * w, 0, 1)
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- 镜头运动
MOVES = {
    #        z0    z1    dx0  dx1  dy0  dy1
    "in":    (1.00, 1.085, 0.0, 0.0, 0.0, 0.0),
    "out":   (1.085, 1.00, 0.0, 0.0, 0.0, 0.0),
    "left":  (1.07, 1.07, 0.9, -0.9, 0.0, 0.0),
    "right": (1.07, 1.07, -0.9, 0.9, 0.0, 0.0),
    "up":    (1.07, 1.07, 0.0, 0.0, 0.8, -0.8),
    "down":  (1.07, 1.07, 0.0, 0.0, -0.8, 0.8),
    "still": (1.035, 1.035, 0.0, 0.0, 0.0, 0.0),
}


def window(plate: np.ndarray, move: str, p: float) -> np.ndarray:
    z0, z1, dx0, dx1, dy0, dy1 = MOVES[move]
    e = ease(p)
    z = z0 + (z1 - z0) * e
    w = PW / z
    h = w * H / W
    mx = (PW - w) / 2
    my = (PH - h) / 2
    cx = PW / 2 + mx * (dx0 + (dx1 - dx0) * e)
    cy = PH / 2 + my * (dy0 + (dy1 - dy0) * e)
    x0 = int(round(min(max(cx - w / 2, 0), PW - w)))
    y0 = int(round(min(max(cy - h / 2, 0), PH - h)))
    x1 = int(round(x0 + w))
    y1 = int(round(y0 + h))
    crop = plate[y0:y1, x0:x1]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 粒子/氛围
_GRAIN: list[np.ndarray] = []


def grain_tiles() -> list[np.ndarray]:
    if not _GRAIN:
        for s in range(6):
            rng = np.random.default_rng(1000 + s)
            g = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.6)
            _GRAIN.append(g)
    return _GRAIN


_VIG: np.ndarray | None = None


def vignette() -> np.ndarray:
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W / 2, H / 2
        r = np.sqrt(((xx - cx) / (W * 0.62)) ** 2 + ((yy - cy) / (H * 0.62)) ** 2)
        v = np.clip(1.0 - 0.30 * np.clip(r - 0.55, 0, 2) ** 1.6, 0.0, 1.0)
        _VIG = v.astype(np.float32)
    return _VIG


_SCRIM_B: np.ndarray | None = None
_SCRIM_L: np.ndarray | None = None


def scrim_bottom() -> np.ndarray:
    """字幕自带纸底：从 y=640 渐到 0.66，最后 120 px 再压深一点。"""
    global _SCRIM_B
    if _SCRIM_B is None:
        yy = np.arange(H, dtype=np.float32)
        a = np.clip((yy - 640) / (1080 - 640), 0, 1) ** 1.5 * 0.62
        a += np.clip((yy - 960) / 120, 0, 1) * 0.12
        _SCRIM_B = np.clip(a, 0, 0.72)[:, None, None].astype(np.float32)
        _SCRIM_B = np.repeat(_SCRIM_B, W, axis=1)
    return _SCRIM_B


def scrim_left() -> np.ndarray:
    """数字字卡自带纸底：左上角一块横向渐隐。"""
    global _SCRIM_L
    if _SCRIM_L is None:
        xx = np.arange(W, dtype=np.float32)
        a = np.clip(1.0 - xx / 760, 0, 1) ** 1.35 * 0.52
        yy = np.arange(H, dtype=np.float32)
        v = np.clip((yy - 80) / 120, 0, 1) * np.clip((430 - yy) / 120, 0, 1)
        _SCRIM_L = (a[None, :] * v[:, None])[:, :, None].astype(np.float32)
    return _SCRIM_L


# ---------------------------------------------------------------- 文字精灵
def text_sprite(lines: list[tuple[str, int, str, float]], pad: int = 12,
                shadow: int = 6) -> tuple[np.ndarray, tuple[int, int]]:
    """lines: (文字, 字号, 字体键, 透明度)"""
    fonts = [ImageFont.truetype(FONTS[k], s) for _, s, k, _ in lines]
    widths, heights = [], []
    for (t, s, _k, _a), f in zip(lines, fonts):
        b = f.getbbox(t)
        widths.append(b[2] - b[0])
        heights.append(b[3] - b[1])
    w = max(widths) + pad * 2
    line_gap = [int(s * 0.55) for _, s, _, _ in lines]
    h = sum(heights) + sum(line_gap[1:]) + pad * 2
    img = Image.new("RGBA", (w + shadow * 2, h + shadow * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = pad
    for i, ((t, s, _k, a), f) in enumerate(zip(lines, fonts)):
        # 阴影：近黑，偏移 (2,3)，模糊
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad + 2, y + 3), t, font=f, fill=(0, 0, 0, int(150 * a)))
        sh = sh.filter(ImageFilter.GaussianBlur(4))
        img = Image.alpha_composite(img, sh)
        d = ImageDraw.Draw(img)
        d.text((pad, y), t, font=f, fill=(242, 237, 228, int(255 * a)))
        y += heights[i] + (line_gap[i] if i < len(line_gap) - 1 else 0)
    return np.array(img)[:, :, :], (pad, pad)


def paste_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int, alpha: float = 1.0) -> np.ndarray:
    """src 是 PIL 出来的 RGBA（RGB 序），dst 是 cv2 的 BGR——必须换序。
    不换的时候，近中性的白字看不出来，带色的一眼就错：底版左缘那条「暖色细边」
    BGR 实测 (178,138,96)、也就是蓝色，设计意图是 RGB (202,150,92) 的琥珀色。
    """
    if alpha <= 0.001:
        return dst
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return dst
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (s[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    roi = roi * (1 - a) + s[:, :, 2::-1].astype(np.float32) * a
    dst[y0:y1, x0:x1] = roi.astype(np.uint8)
    return dst


TAIL = 104  # 底版右缘渐隐带宽度：字右边必须留出这么宽，渐隐不许压到字上


def label_sprite(big: str, small: str) -> tuple[np.ndarray, tuple[int, int]]:
    """数字字卡自带一块深色底版（左缘一道暖色细边）：深底上单靠字重压不住，
    底版一给，字卡就变成「版式」而不是「贴纸」。

    渐隐只作用在**底版**上，且整条渐隐带留在字右边。第一版把 ramp 乘在整张 sprite 的
    alpha 上，字自己也跟着淡掉了——`suppliers` 的右端基本看不见（verify 量到 2.91）。
    """
    f_big = ImageFont.truetype(FONTS["num"], 76)
    f_sm = ImageFont.truetype(FONTS["cn_light"], 23)
    pad_x, pad_y = 34, 24
    wb = f_big.getbbox(big)
    ws = f_sm.getbbox(small)
    tw = max(wb[2] - wb[0], ws[2] - ws[0])
    w = pad_x * 2 + 8 + tw + TAIL
    hb, hs = wb[3] - wb[1], ws[3] - ws[1]
    h = pad_y * 2 + hb + 10 + hs

    plate = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dp = ImageDraw.Draw(plate)
    dp.rounded_rectangle([0, 0, w - 1, h - 1], radius=3, fill=(8, 9, 11, 224))
    dp.rectangle([0, 0, 3, h - 1], fill=(202, 150, 92, 236))
    arr = np.array(plate).astype(np.float32)
    ramp = np.ones(w, np.float32)
    ramp[w - TAIL:] = np.linspace(1.0, 0.0, TAIL) ** 1.5
    arr[:, :, 3] *= ramp[None, :]
    plate = Image.fromarray(arr.astype(np.uint8))

    d = ImageDraw.Draw(plate)
    d.text((pad_x + 8, pad_y - wb[1]), big, font=f_big, fill=(240, 235, 226, 255))
    d.text((pad_x + 10, pad_y + hb + 8 - ws[1]), small, font=f_sm, fill=(208, 202, 192, 255))
    return np.array(plate), (0, 0)


def spaced(d: ImageDraw.ImageDraw, xy, text: str, font, fill, track: int) -> int:
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + track
    return int(x - xy[0])


# ---------------------------------------------------------------- 落版
_CARD: np.ndarray | None = None


def card_frame() -> np.ndarray:
    global _CARD
    if _CARD is not None:
        return _CARD
    img = Image.new("RGB", (W, H), (7, 8, 10))
    d = ImageDraw.Draw(img)
    # 极淡的横向渐层，避免死平
    for i in range(H):
        v = int(7 + 5 * (1 - abs(i - H * 0.42) / (H * 0.58)))
        d.line([(0, i), (W, i)], fill=(v, v + 1, v + 3))
    # 一枚几乎看不见的「牵引座」圆盘：与主题同构，只做结构，不做装饰
    ring = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    cx, cy, r0 = 1570, 250, 470
    for k, (rr, aa, wd) in enumerate([(r0, 58, 3), (r0 - 46, 34, 2), (r0 - 96, 20, 1)]):
        rd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(214, 158, 96, aa), width=wd)
    rd.line([(cx, cy - r0 - 40), (cx, cy - r0 + 30)], fill=(214, 158, 96, 30), width=2)
    rd.line([(120, 286), (120, 792)], fill=(214, 158, 96, 64), width=2)
    img = Image.alpha_composite(img.convert("RGBA"), ring).convert("RGB")
    d = ImageDraw.Draw(img)

    f_big = ImageFont.truetype(FONTS["cn"], 148)
    f_mid = ImageFont.truetype(FONTS["cn_light"], 38)
    f_sub = ImageFont.truetype(FONTS["cn"], 54)
    f_en = ImageFont.truetype(FONTS["num"], 24)
    f_small = ImageFont.truetype(FONTS["cn_light"], 22)

    x = 168
    d.text((x, 286), story.CARD["line1"], font=f_big, fill=(242, 236, 226))
    d.text((x + 4, 464), story.CARD["line2"], font=f_mid, fill=(200, 192, 180))
    d.line([(x, 542), (x + 520, 542)], fill=(148, 96, 56), width=2)
    d.text((x, 574), story.CARD["line3"], font=f_sub, fill=(216, 160, 98))
    spaced(d, (x + 2, 676), story.CARD["line4"], f_en, (176, 172, 164), 7)
    spaced(d, (x + 2, 714), story.CARD["line5"], f_en, (128, 126, 120), 7)
    d.text((x, H - 82), story.CARD["foot"], font=f_small, fill=(132, 132, 132))
    # 整张卡是 PIL 画的（RGB），但 Renderer 的其它帧都是 cv2 的 BGR。
    # 不换序的话，卡上的琥珀色（214,158,96）会渲成蓝色——圆盘环、竖线、片名都跟着偏色。
    arr = np.array(img)[:, :, ::-1]
    _CARD = arr
    return arr


# ---------------------------------------------------------------- 渲染器
class Renderer:
    def __init__(self, no_text: bool = False):
        self.no_text = no_text
        self.plates: dict[str, np.ndarray] = {}
        for s in story.SHOTS:
            if s.prompt:
                # 调色是逐点运算：先对整张底本做一次，再裁窗口，结果等价而便宜一百倍。
                self.plates[s.key] = grade(imread_u(prep_one(s.key)), s.grade)
        self.bounds = list(zip([s.key for s in story.SHOTS], story.STARTS,
                               [s.end for s in story.SHOTS]))
        # 字幕精灵
        self.subs = {}
        for v in story.VOICES:
            sp, off = text_sprite([(v.text, story.SUB["size_cn"], "cn", 1.0),
                                   (v.en, story.SUB["size_en"], "cn_light", 0.80)], shadow=5)
            self.subs[v.idx] = (sp, off)
        # 数字字卡
        self.labs = {}
        for lb in story.LABELS:
            self.labs[lb.key] = label_sprite(lb.text, lb.sub)
        self.lab_t0 = {lb.key: lb.t0 for lb in story.LABELS}

    def shot_at(self, t: float) -> tuple[story.Shot, float]:
        for s in story.SHOTS:
            if s.start <= t < s.end:
                return s, (t - s.start) / s.dur
        return story.SHOTS[-1], 1.0

    def frame(self, i: int) -> np.ndarray:
        t = i / FPS
        s, p = self.shot_at(t)
        if s.key == "card":
            return card_frame().copy()
        plate = self.plates[s.key]
        img = window(plate, s.move, p)
        # 颗粒 + 暗角（两次全画幅运算，合并成一次乘加）
        g = grain_tiles()[i % 6] * (2.6 if s.grade != "still" else 2.0)
        f = img.astype(np.float32)
        f *= vignette()[..., None]
        f += g[..., None]
        img = np.clip(f, 0, 255).astype(np.uint8)

        if self.no_text:
            return img

        # 数字字卡
        lab_alpha = {}
        for lb in story.LABELS:
            if lb.t0 - 0.55 <= t <= lb.t1 + 0.55:
                lab_alpha[lb.key] = (smoothstep((t - lb.t0 + 0.55) / 0.55)
                                     * smoothstep((lb.t1 + 0.55 - t) / 0.55))
        sub_alpha = {}
        for v in story.VOICES:
            if v.t - 0.25 <= t <= v.t + v.window:
                sub_alpha[v.idx] = (smoothstep((t - v.t + 0.25) / 0.25)
                                    * smoothstep((v.t + v.window - t) / 0.30))

        # 纸底：字幕自带一条底部渐层（2026-11-14 的教训：深底上单靠字重压不住）
        if sub_alpha:
            a = max(sub_alpha.values())
            img = (img.astype(np.float32) * (1 - scrim_bottom() * a)).astype(np.uint8)

        for key, a in lab_alpha.items():
            sp, (ox, oy) = self.labs[key]
            dy = int(round(14 * (1 - smoothstep((t - self.lab_t0[key]) / 1.2))))
            img = paste_rgba(img, sp, story.SUB["x"] - ox, 150 - oy + dy, a)

        for idx, a in sub_alpha.items():
            sp, (ox, oy) = self.subs[idx]
            img = paste_rgba(img, sp, story.SUB["x"] - ox, story.SUB["y_cn"] - oy, a)
        return img


def fade_frames(fr: np.ndarray, t: float) -> np.ndarray:
    """片头 0.6s 从黑起、片尾 147.6→148.1 落黑、落版 150→150.6 起。"""
    a = 1.0
    if t < 0.6:
        a = smoothstep(t / 0.6)
    if story.BLK0 <= t < story.BLK1:
        a = 1.0 - smoothstep((t - story.BLK0) / (story.BLK1 - story.BLK0))
    if story.BLK1 <= t < story.CARD_IN:
        a = 0.0
    if story.CARD_IN <= t < story.CARD_IN + 0.75:
        a = smoothstep((t - story.CARD_IN) / 0.75)
    if t >= 155.3:
        a = 1.0 - smoothstep((t - 155.3) / 0.7)
    if a >= 0.999:
        return fr
    return (fr.astype(np.float32) * a).astype(np.uint8)


def retag(src: Path, dst: Path) -> None:
    """把 bt709 三件套补进容器。

    `encode()` 已经给编码器带了 `-color_primaries/-color_trc/-colorspace`，但走 rawvideo 直编时
    libx264 只写进了 matrix 与 range，primaries/trc 丢掉（ffprobe 读出来是 bt709/None/tv）。
    改走一次 `-c copy` 重新封装才落得进去——原来的分段拼接路径正好是这么做的，所以一直是对的。
    """
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                    "-c", "copy", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-colorspace", "bt709", "-color_range", "tv",
                    "-movflags", "+faststart", str(dst)], check=True)


def encode(audio: Path | None, out: Path, renderer: Renderer,
           n: int | None = None, start: int = 0, total: int | None = None) -> None:
    n = n or story.TOTAL_FRAMES
    total = total if total is not None else n
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        *(["-i", str(audio)] if audio else []),
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", PRESET, "-crf", "17", "-pix_fmt", "yuv420p",
        "-threads", THREADS, "-profile:v", "high", "-level", "4.2", "-tune", "film",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-color_range", "tv",
        *(["-c:a", "aac", "-b:a", "192k", "-ar", "48000"] if audio else []),
        "-movflags", "+faststart", *(["-shortest"] if audio else []), str(out),
    ]
    # stderr 必须落文件，不能挂 PIPE：不读的 PIPE 写满 64 KB 就把 ffmpeg 堵死，
    # python 再被 stdin 反压堵死——两条进程都「不占 CPU 但一动不动」。
    errpath = out.with_suffix(".fflog")
    errf = open(errpath, "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
    step = max(1, n // 12)
    for k in range(n):
        i = start + k
        fr = fade_frames(renderer.frame(i), i / FPS)
        proc.stdin.write(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).tobytes())
        if k % step == 0:
            print(f"  frame {i}/{total} ({i * 100 // max(1, total)}%)", flush=True)
    proc.stdin.close()
    rc = proc.wait()
    errf.close()
    err = errpath.read_text(encoding="utf-8", errors="replace")
    if rc != 0:
        print(err[-3000:], file=sys.stderr)
        raise SystemExit(f"ffmpeg 失败 rc={rc}")
    print(f"wrote {out} ({out.stat().st_size // 1024 // 1024} MB)")


# ---------------------------------------------------------------- 分镜表
def shotlist() -> None:
    lines = ["# 02 分镜表（由 render.py --shotlist 生成，勿手改）", "",
             f"总长 **{story.DUR:.1f} s**｜{story.W}×{story.H}｜{story.FPS} fps｜"
             f"{story.TOTAL_FRAMES} 帧｜{len(story.SHOTS)} 个镜头", "",
             "| # | 时间码 | 时长 | 镜头 | 运镜 | 调色 | 承担的论证 |",
             "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(story.SHOTS, 1):
        tc = f"{int(s.start)//60:02d}:{s.start%60:05.2f}–{int(s.end)//60:02d}:{s.end%60:05.2f}"
        lines.append(f"| {i} | {tc} | {s.dur:.1f}s | `{s.key}` | {s.move} | {s.grade} | {s.note} |")
    lines += ["", "## 旁白", "", "| # | 入点 | 窗口 | 中文 | English |", "|---|---|---|---|---|"]
    for v in story.VOICES:
        lines.append(f"| {v.idx} | {v.t:.1f}s | {v.window:.1f}s | {v.text} | {v.en} |")
    lines += ["", "## 屏幕数字", "", "| 键 | 入–出 | 数字 | 说明 |", "|---|---|---|---|"]
    for lb in story.LABELS:
        lines.append(f"| {lb.key} | {lb.t0:.1f}–{lb.t1:.1f}s | {lb.text} | {lb.sub} |")
    lines += ["", "## 音效点", "", "| 时间 | 音 | 增益 |", "|---|---|---|"]
    for c in story.CUES:
        lines.append(f"| {c.t:.1f}s | {c.kind} | {c.gain:.2f} |")
    p = ROOT / "docs" / "02-SHOTLIST.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--pipe", action="store_true")
    ap.add_argument("--seg", default="")
    ap.add_argument("--concat", action="store_true")
    ap.add_argument("--silent", action="store_true")
    ap.add_argument("--shotlist", action="store_true")
    ap.add_argument("--out", default=str(OUT / "maojian.mp4"))
    ap.add_argument("--audio", default=str(OUT / "audio-master.wav"))
    ap.add_argument("--no-text", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    PLATE.mkdir(parents=True, exist_ok=True)

    if a.shotlist:
        shotlist()
        return 0
    if a.force_prep:
        for f in PLATE.glob("*.png"):
            f.unlink()
    if a.prep:
        for s in story.SHOTS:
            if s.prompt:
                print("prep", s.key, "->", prep_one(s.key).name)
        return 0

    r = Renderer(no_text=a.no_text)

    if a.stills is not None:
        (OUT / "frames").mkdir(exist_ok=True)
        for t in a.stills or []:
            i = int(round(t * FPS))
            fr = fade_frames(r.frame(i), i / FPS)
            p = OUT / "frames" / f"t{t:06.2f}.jpg"
            ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            buf.tofile(str(p))
            print("still", p.name)
        return 0

    audio = None if a.silent else Path(a.audio)
    if audio and not audio.exists():
        print("（没有音轨，先跑 audio.py）", file=sys.stderr)
        return 2

    if a.seg:
        k, n = (int(x) for x in a.seg.split("/"))
        per = math.ceil(story.TOTAL_FRAMES / n)
        s0 = (k - 1) * per
        cnt = min(per, story.TOTAL_FRAMES - s0)
        encode(audio, OUT / f"seg-{k}.mp4", r, n=cnt, start=s0, total=story.TOTAL_FRAMES)
        return 0

    if a.concat:
        segs = sorted(OUT.glob("seg-*.mp4"), key=lambda p: int(p.stem.split("-")[1]))
        if not segs:
            raise SystemExit("没有 seg-*.mp4")
        # 分段一律先剥掉音轨再拼：每段都带着「从 0 开始」的音轨，直接拼会串音。
        bare = []
        for p in segs:
            q = OUT / f"v-{p.stem.split('-')[1]}.mp4"
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(p),
                            "-an", "-c:v", "copy", str(q)], check=True)
            bare.append(q)
        lst = OUT / "segs.txt"
        lst.write_text("".join("file '{}'\n".format(p.as_posix()) for p in bare), encoding="utf-8")
        silent = OUT / "video-only.mp4"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
                        "-safe", "0", "-i", str(lst), "-c", "copy", str(silent)], check=True)
        if Path(a.audio).exists():
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent),
                            "-i", str(a.audio), "-map", "0:v:0", "-map", "1:a:0",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                            "-colorspace", "bt709", "-color_primaries", "bt709",
                            "-color_trc", "bt709", "-color_range", "tv",
                            "-movflags", "+faststart", "-shortest", str(OUT / "maojian.mp4")], check=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent),
                            "-c", "copy", "-movflags", "+faststart", str(OUT / "maojian.mp4")],
                           check=True)
        print("wrote", OUT / "maojian.mp4")
        return 0

    if a.pipe:
        dst = Path(a.out)
        tmp = dst.with_name("_pipe-tmp.mp4")
        encode(audio, tmp, r)
        retag(tmp, dst)
        tmp.unlink(missing_ok=True)
        print("wrote", dst, f"({dst.stat().st_size // 1024 // 1024} MB)")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
