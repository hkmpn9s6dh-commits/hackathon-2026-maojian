"""声音：全片自产。没有一条素材是拿来的。

结构：
  · 底噪   —— 低频 drone（42 / 63 Hz 失谐）+ 厂房空气声，全片铺底
  · 心跳   —— 45 s 锁钩合上之后进来的机械脉冲（52 BPM），到港口撤掉
  · 垫音   —— 116 s 山口进来一个五度，135 s 海外加一层微光
  · 事件声 —— 16 个点，全部按 story.CUES 落位
  · 旁白   —— edge-tts，普通话男声，−14% 语速

输出 out/audio.wav（48k 单声道）；母带另跑 scripts/master_audio.py（−16 LUFS）。
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

import story

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
TTS_DIR = OUT / "tts"
SR = 48000
N = int(story.DUR * SR)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def env_adsr(n: int, a: float, d: float, s: float, r: float, sus: float = 0.7) -> np.ndarray:
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    s_n = max(0, n - a_n - d_n - r_n)
    e = np.concatenate([np.linspace(0, 1, a_n, endpoint=False),
                        np.linspace(1, sus, d_n, endpoint=False),
                        np.full(s_n, sus),
                        np.linspace(sus, 0, r_n)])
    if len(e) < n:
        e = np.pad(e, (0, n - len(e)))
    return e[:n].astype(np.float32)


def band(x: np.ndarray, lo: float, hi: float, soft: float = 0.3) -> np.ndarray:
    """FFT 域的软带通；soft 越大边越软。"""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    mask = np.ones_like(f)
    if lo > 0:
        mask *= 1 / (1 + (lo / np.maximum(f, 1e-6)) ** (2 / soft))
    if hi < SR / 2:
        mask *= 1 / (1 + (f / hi) ** (2 / soft))
    return np.fft.irfft(X * mask, n).astype(np.float32)


def norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return (x / m * peak).astype(np.float32)


def add(buf: np.ndarray, sig: np.ndarray, t: float, gain: float) -> None:
    i = int(t * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


# ---------------------------------------------------------------- 素材
def s_drone(dur: float, seed: int = 1) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    x = (0.55 * np.sin(2 * np.pi * 42.0 * t)
         + 0.30 * np.sin(2 * np.pi * 63.1 * t + 0.7)
         + 0.14 * np.sin(2 * np.pi * 84.6 * t + 1.9))
    x *= 1 + 0.09 * np.sin(2 * np.pi * 0.07 * t)
    air = band(r.normal(0, 1, n).astype(np.float32), 120, 900, 0.5)
    x = x.astype(np.float32) + 0.22 * norm(air)
    return norm(x, 0.55)


def s_pulse(seed: int) -> np.ndarray:
    """机械心跳：80 Hz 低鼓 + 一点点金属。"""
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * (74 - 26 * t / 0.55) * t) * env_adsr(n, 0.004, 0.09, 0.0, 0.40, 0.12)
    click = norm(rng(seed).normal(0, 1, n).astype(np.float32)) * np.exp(-t * 90) * 0.25
    ring = np.sin(2 * np.pi * 1180 * t) * np.exp(-t * 26) * 0.06
    return norm(body + click + ring, 0.9)


def s_tyre(seed: int) -> np.ndarray:
    n = int(2.6 * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    wet = band(r.normal(0, 1, n).astype(np.float32), 900, 7000, 0.4)
    wet *= np.exp(-t * 2.2) * (1 + 0.4 * np.sin(2 * np.pi * 13 * t))
    body = band(r.normal(0, 1, n).astype(np.float32), 60, 400, 0.5) * np.exp(-t * 5)
    return norm(wet * 0.8 + body * 0.9, 0.8)


def s_metal(seed: int) -> np.ndarray:
    n = int(1.8 * SR)
    t = np.arange(n) / SR
    x = np.zeros(n, np.float32)
    for f, g, dcy in [(880, 0.5, 3.0), (1320, 0.35, 4.2), (2470, 0.22, 6.0), (3910, 0.12, 8.5)]:
        x += g * np.sin(2 * np.pi * f * t) * np.exp(-t * dcy)
    x += 0.4 * norm(band(rng(seed).normal(0, 1, n).astype(np.float32), 2000, 9000, 0.4)) * np.exp(-t * 40)
    return norm(x, 0.55)


def s_grease(seed: int) -> np.ndarray:
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    x = band(r.normal(0, 1, n).astype(np.float32), 300, 2200, 0.45)
    x *= np.exp(-t * 2.0) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t))
    return norm(x, 0.35)


def s_pin(seed: int) -> np.ndarray:
    """锁钩咬住牵引销：低频撞击 + 金属余振 + 一点点油脂。"""
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    thump = np.sin(2 * np.pi * (58 - 20 * t / 0.4) * t) * np.exp(-t * 7.5)
    crack = norm(band(r.normal(0, 1, n).astype(np.float32), 400, 9000, 0.35)) * np.exp(-t * 120) * 0.55
    ring = np.zeros(n, np.float32)
    for f, g, dcy in [(612, 0.30, 6.0), (1465, 0.20, 9.0), (2310, 0.10, 12.0)]:
        ring += g * np.sin(2 * np.pi * f * t) * np.exp(-t * dcy)
    return norm(thump * 1.0 + crack + ring, 0.95)


def s_unlock(seed: int) -> np.ndarray:
    n = int(1.4 * SR)
    x = s_pin(seed)[: n]
    t = np.arange(len(x)) / SR
    # 先一声小的「嗒」，再放掉
    lead = np.zeros(len(x), np.float32)
    k = int(0.06 * SR)
    lead[:k] = norm(rng(seed).normal(0, 1, k).astype(np.float32)) * np.exp(-np.arange(k) / SR * 260) * 0.4
    return norm(lead + x * 0.8 * np.exp(-t * 2.5), 0.8)


def s_press(seed: int) -> np.ndarray:
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    hit = np.sin(2 * np.pi * (48 - 14 * t / 0.6) * t) * np.exp(-t * 4.0)
    hiss = band(r.normal(0, 1, n).astype(np.float32), 1500, 10000, 0.4)
    hiss *= np.exp(-np.maximum(0, t - 0.25) * 1.6) * (t > 0.2)
    return norm(hit * 1.0 + norm(hiss) * 0.35, 0.9)


def s_weld(seed: int) -> np.ndarray:
    n = int(2.4 * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    buzz = np.sign(np.sin(2 * np.pi * 100 * t)) * 0.35 + np.sin(2 * np.pi * 200 * t) * 0.2
    buzz = band(buzz.astype(np.float32), 200, 5000, 0.4)
    crackle = (r.random(n) > 0.9992) * r.normal(0, 1, n).astype(np.float32) * 1.5
    crackle = band(crackle, 1500, 12000, 0.3)
    e = np.minimum(1, t / 0.08) * np.minimum(1, (2.4 - t) / 0.5)
    return norm((buzz * 0.7 + crackle * 0.5) * e, 0.55)


def s_spray(seed: int) -> np.ndarray:
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    x = band(rng(seed).normal(0, 1, n).astype(np.float32), 2500, 14000, 0.5)
    return norm(x * np.exp(-t * 1.2) * 0.5, 0.4)


def s_hoist(seed: int) -> np.ndarray:
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    imp = np.zeros(n, np.float32)
    idx = r.integers(0, n, 90)
    imp[idx] = r.normal(0, 1, 90)
    chain = band(imp, 1200, 6500, 0.4)
    low = np.sin(2 * np.pi * 55 * t) * 0.25 * np.exp(-t * 3)
    return norm(chain * 0.8 + low.astype(np.float32), 0.45)


def s_gear(seed: int) -> np.ndarray:
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    x = band(r.normal(0, 1, n).astype(np.float32), 400, 3500, 0.45)
    x *= (0.5 + 0.5 * np.sin(2 * np.pi * 18 * t)) * np.exp(-t * 1.8)
    return norm(x, 0.4)


def s_caliper(seed: int) -> np.ndarray:
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 2100 * t) * np.exp(-t * 80) * 0.5
    x += norm(rng(seed).normal(0, 1, n).astype(np.float32)) * np.exp(-t * 300) * 0.3
    return norm(x, 0.3)


def s_engine(seed: int) -> np.ndarray:
    n = int(4.0 * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    base = np.sin(2 * np.pi * 34 * t) + 0.5 * np.sin(2 * np.pi * 68 * t) + 0.28 * np.sin(2 * np.pi * 102 * t)
    rumble = band(r.normal(0, 1, n).astype(np.float32), 40, 600, 0.5)
    x = base.astype(np.float32) * 0.6 + norm(rumble) * 0.4
    e = np.minimum(1, t / 0.8) * np.minimum(1, (4.0 - t) / 1.4)
    return norm(x * e, 0.5)


def s_wind(dur: float, seed: int, lo: float = 200, hi: float = 4000) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = band(rng(seed).normal(0, 1, n).astype(np.float32), lo, hi, 0.55)
    x *= 0.55 + 0.45 * np.sin(2 * np.pi * 0.09 * t + 1.1)
    x *= np.minimum(1, t / 1.0) * np.minimum(1, (dur - t) / 1.0)
    return norm(x, 0.5)


def s_ship_horn(seed: int) -> np.ndarray:
    n = int(3.6 * SR)
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * 110 * t) + 0.5 * np.sin(2 * np.pi * 165 * t)
         + 0.3 * np.sin(2 * np.pi * 220 * t))
    e = np.minimum(1, t / 0.35) * np.minimum(1, (3.6 - t) / 1.2)
    return norm(x.astype(np.float32) * e, 0.45)


def s_pad(dur: float, root: float, seed: int) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n, np.float32)
    for k, g in [(1, 0.5), (1.5, 0.28), (2, 0.18), (3, 0.09)]:
        for det in (-0.6, 0.0, 0.7):
            ph = rng(seed).random() * 6.28
            x += g / 3 * np.sin(2 * np.pi * (root * k + det) * t + ph)
    x *= 0.55 + 0.45 * np.sin(2 * np.pi * 0.05 * t)
    x *= np.minimum(1, t / 6.0) * np.minimum(1, (dur - t) / 6.0)
    return norm(x, 0.35)


# ---------------------------------------------------------------- 旁白
async def _tts(text: str, voice: str, rate: str, pitch: str, dst: Path) -> None:
    import edge_tts
    c = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await c.save(str(dst))


def tts_wav(v: story.Voice) -> Path:
    TTS_DIR.mkdir(exist_ok=True)
    mp3 = TTS_DIR / f"vo{v.idx}.mp3"
    wav = TTS_DIR / f"vo{v.idx}.wav"
    if not mp3.exists():
        asyncio.run(_tts(v.text, v.voice, v.rate, v.pitch, mp3))
    if not wav.exists():
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp3),
                        "-ac", "1", "-ar", str(SR), str(wav)], check=True)
    return wav


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == SR
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def loudness(x: np.ndarray) -> float:
    return 20 * np.log10(max(1e-9, float(np.sqrt(np.mean(x ** 2)))))


# ---------------------------------------------------------------- 主进程
def main() -> int:
    OUT.mkdir(exist_ok=True)
    bed = np.zeros(N, np.float32)

    # 底：drone
    add(bed, s_drone(story.DUR, 1), 0.0, 0.55)

    # 心跳：45.4 锁钩之后 到 125 港口之前
    t = 45.4
    k = 0
    while t < 124.0:
        add(bed, s_pulse(300 + k), t, 0.30 if t > 60 else 0.22)
        t += 60 / 52
        k += 1

    # 垫音
    add(bed, s_pad(46.0, 55.0, 7), 112.0, 0.30)
    add(bed, s_pad(30.0, 82.5, 8), 130.0, 0.22)
    add(bed, s_pad(26.0, 110.0, 9), 134.0, 0.12)

    # 事件
    for c in story.CUES:
        g = c.gain
        if c.kind == "tyre":
            add(bed, s_tyre(int(c.t)), c.t, 0.55 * g)
        elif c.kind == "metal":
            add(bed, s_metal(int(c.t)), c.t, 0.30 * g)
        elif c.kind == "grease":
            add(bed, s_grease(int(c.t)), c.t, 0.35 * g)
        elif c.kind == "pin":
            add(bed, s_pin(int(c.t)), c.t, 0.85 * g)
        elif c.kind == "unlock":
            add(bed, s_unlock(int(c.t)), c.t, 0.70 * g)
        elif c.kind == "press":
            add(bed, s_press(int(c.t)), c.t, 0.60 * g)
        elif c.kind == "weld":
            add(bed, s_weld(int(c.t)), c.t, 0.35 * g)
        elif c.kind == "spray":
            add(bed, s_spray(int(c.t)), c.t, 0.45 * g)
        elif c.kind == "hoist":
            add(bed, s_hoist(int(c.t)), c.t, 0.40 * g)
        elif c.kind == "gear":
            add(bed, s_gear(int(c.t)), c.t, 0.32 * g)
        elif c.kind == "caliper":
            add(bed, s_caliper(int(c.t)), c.t, 0.45 * g)
        elif c.kind == "engine":
            add(bed, s_engine(int(c.t)), c.t, 0.45 * g)
        elif c.kind == "wind":
            add(bed, s_wind(10.0, int(c.t), 260, 3600), c.t, 0.30 * g)
        elif c.kind == "wind2":
            add(bed, s_wind(11.0, int(c.t), 300, 2600), c.t, 0.26 * g)
        elif c.kind == "horn":
            add(bed, s_ship_horn(int(c.t)), c.t, 0.55 * g)

    # 旁白
    vo_sum = np.zeros(N, np.float32)
    spans = []
    ok = True
    for v in story.VOICES:
        w = tts_wav(v)
        sig = load_wav(w)
        sig = band(sig, 85, 13000)
        sig = norm(sig, 0.95)
        d = len(sig) / SR
        print(f"VO{v.idx}: {d:5.2f}s / window {v.window:4.1f}s  "
              f"{'OK' if d <= v.window else 'TOO LONG'}  {v.text}")
        ok = ok and d <= v.window
        spans.append((v.t, v.t + d))
        add(vo_sum, sig, v.t, 0.92)
    if not ok:
        print("！有旁白超出窗口，改 story.py 的 window 或文案", file=sys.stderr)

    # 旁白期间压床
    duck = np.ones(N, np.float32)
    for a, b in spans:
        ia, ib = int(a * SR), int(b * SR)
        atk, rel = int(0.22 * SR), int(0.65 * SR)
        duck[max(0, ia - atk):ia] = np.minimum(
            duck[max(0, ia - atk):ia], np.linspace(1, 0.55, min(atk, ia)))
        duck[ia:ib] = 0.48
        duck[ib:min(N, ib + rel)] = np.minimum(
            duck[ib:min(N, ib + rel)], np.linspace(0.48, 1, min(rel, N - ib)))

    # 148.1–150 的黑场：把床压到很低，只留最后一声「咔」
    lo, hi = int(148.0 * SR), int(150.2 * SR)
    if hi > lo:
        duck[lo:hi] *= np.linspace(0.25, 0.10, hi - lo)

    mix = bed * duck * 0.62 + vo_sum
    mix = band(mix, 32, 16000)
    # 只压峰、不压动态：0.70 以下原样，以上做软拐点（tanh 会把手感压平，LRA 掉到 3.6）
    a = np.abs(mix)
    over = a > 0.70
    if over.any():
        mix[over] = np.sign(mix[over]) * (0.70 + 0.29 * np.tanh((a[over] - 0.70) / 0.29))
    pk = float(np.max(np.abs(mix)))
    if pk > 0.995:
        mix *= 0.995 / pk
    print(f"mix rms {loudness(mix):.1f} dBFS, peak {20*np.log10(np.max(np.abs(mix))):.2f} dBFS")

    p = OUT / "audio.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    print("wrote", p)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
