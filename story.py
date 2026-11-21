"""《第五个轮子》/ THE FIFTH WHEEL — 单一来源。

片子的一切都从这里出：镜头、时长、构图、旁白、字幕、音效、落版。
改片子只改这个文件；docs/02-SHOTLIST.md 由 `render.py --shotlist` 生成，
`verify.py` 会拿它反过来核对，防止「改了时长忘了重生成文档」。

赛道 AS · 十堰市茅箭区首届 Token 产业 IP 创意大赛 · AI 产业宣传片

v2（闸 D 之后改的，三路盲读各指一遍）：
  · 第 5、6 镜换成「牵引车倒进挂车底下」「从地面仰视挂车底盘落下来」——
    原来那个抽象的圆盘法兰，三位评委都说「这不是牵引座」，改「拍事件」而不是「拍零件」。
  · 第 7 镜由冲压改成钢卷库（原图的钣金像布，被指违反物理）。
  · 第 11 镜由啮合齿轮改成「货架上成排的同步器」（原来那张齿圈不啮合，被指结构错）。
  · 英文全部重写：原稿五处直译，`THE FIFTH WHEEL` 在英语里是「多余的人」的习语，
    与中文意思相反，落版改成 `THE FIFTH WHEEL COUPLING`（行内正式名）。
  · 首句「四个轮子走路」会被画面里的多轴车反驳，改成「四只轮子先落地」——只说不落地的那个。
"""
from __future__ import annotations

from dataclasses import dataclass

W, H = 1920, 1080
FPS = 24
DUR = 156.0

# 底本尺寸（3:2，比 16:9 多出取景余量；生成器出 1536×1024 再放大到这里）
PW, PH = 2400, 1600

FONT_CN = "C:/Windows/Fonts/Noto Sans SC (TrueType).otf"
FONT_CN_B = "C:/Windows/Fonts/Noto Sans SC Bold (TrueType).otf"
FONT_NUM = "C:/Windows/Fonts/Deng.ttf"


@dataclass
class Shot:
    key: str
    dur: float
    kind: str          # wide / macro / inside
    move: str          # in / out / left / right / up / down / still
    grade: str         # cool / warm / steel / dusk
    prompt: str
    note: str = ""     # 这一镜承担哪一句论证

    @property
    def start(self) -> float:
        t = 0.0
        for s in SHOTS:
            if s.key == self.key:
                return t
            t += s.dur
        raise KeyError(self.key)

    @property
    def end(self) -> float:
        return self.start + self.dur


# ---------------------------------------------------------------- 镜头
# 视觉总纲：工业微距 + 冷钢蓝灰 + 唯一暖色（弧光/钠灯）。
# 卡车一律不挂牌照、不露商标；远景靠雾、逆光、剪影，避开生成器的车头比例软肋。
STYLE = (
    "cinematic industrial brand-film still, shot on ARRI Alexa with anamorphic 40mm, "
    "volumetric haze, desaturated steel blue-grey palette with one warm amber accent, "
    "deep crushed shadows, soft highlight rolloff, fine 35mm grain, photorealistic, "
    "no text, no lettering, no logos, no watermark, no people looking at camera"
)

SHOTS: list[Shot] = [
    Shot("dawn_road", 11.0, "wide", "in", "cool",
         "a two-lane mountain highway at first light cutting through the Qinling foothills, "
         "low mist between dark ridges, a single heavy truck far away with headlights on, "
         "wet asphalt reflecting cold sky, wide establishing shot"),
    Shot("wheel_close", 8.0, "macro", "left", "steel",
         "extreme close-up of a heavy truck front wheel rolling through a shallow puddle on wet asphalt, "
         "water and grit thrown up, tyre tread filling the frame, motion blur at the rim"),
    Shot("four_wheels", 8.0, "wide", "right", "cool",
         "low angle under a parked heavy truck at dawn showing four wheels in a row on wet concrete, "
         "chassis rails and air tanks above, backlit haze, symmetric composition"),
    Shot("under_body", 8.0, "inside", "up", "steel",
         "inside the chassis of a heavy truck looking along the drivetrain: propeller shaft, "
         "axle housing, air lines, greased metal, a thin shaft of light from above"),
    Shot("couple_wide", 10.0, "wide", "right", "dusk",
         "a tractor unit slowly reversing under a parked semi-trailer in a freight yard at dusk, "
         "low three-quarter view from the rear corner of the tractor, the gap between tractor and "
         "trailer narrowing, silhouettes, sodium lamps in haze, wet asphalt"),
    Shot("couple_low", 8.0, "inside", "up", "warm",
         "extreme low angle from the ground beneath a semi-trailer: the trailer's steel underside and "
         "landing legs overhead, the tractor's rear frame entering from the right, thick grease and "
         "worn steel, one amber work lamp, deep shadow"),
    Shot("coil", 8.0, "wide", "right", "cool",
         "cold-rolled steel coils stacked in rows inside a dark plant, an overhead crane hook and sling "
         "above them, shafts of daylight from high windows, dust in the air, industrial scale"),
    Shot("weld", 8.0, "wide", "in", "cool",
         "rows of welding robots on a truck cab assembly line, blue-white arc flashes, sparks falling, "
         "cab shells in jigs, dark hall with haze"),
    Shot("paint", 7.0, "wide", "left", "steel",
         "a truck cab shell moving through a paint booth, electrostatic mist, soft light bars overhead, "
         "glossy white surface, sterile industrial atmosphere"),
    Shot("assembly", 9.0, "wide", "down", "cool",
         "a truck cab being lowered onto a chassis on the final assembly line, workers in dark uniforms "
         "guiding it by hand, cables and hoists above, dramatic top light, silhouettes only"),
    Shot("rack", 8.0, "macro", "in", "steel",
         "rows of identical machined synchroniser assemblies standing in a factory rack, oiled steel "
         "surfaces, cold overhead light, a repeating pattern receding into darkness, shallow depth of field"),
    Shot("caliper", 7.0, "macro", "left", "warm",
         "a gloved hand holding a digital micrometer against a machined steel part, measurement jaws "
         "touching the surface, shallow depth of field, warm task light in a dark inspection room"),
    Shot("screen", 7.0, "wide", "in", "cool",
         "a dark control room with a wall of screens showing abstract production data and wireframe "
         "vehicle geometry, cold blue glow on empty chairs, shallow depth of field"),
    Shot("gate_out", 9.0, "wide", "out", "warm",
         "an unbranded white heavy truck rolling out of a factory gate at sunrise, long shadows across "
         "wet concrete, guard booth silhouette, backlit dust, hopeful morning light"),
    Shot("highway", 9.0, "wide", "right", "cool",
         "aerial view of a convoy of heavy trucks crossing a high viaduct between mountains, morning fog "
         "in the valley below, cold blue and pale gold, sweeping drone shot"),
    Shot("port", 10.0, "wide", "in", "dusk",
         "a heavy truck driving up the ramp into a ro-ro vehicle carrier at dusk, ship hull towering, "
         "port cranes and containers in silhouette, sodium lights reflecting on wet dock"),
    Shot("overseas", 9.0, "wide", "left", "warm",
         "a cab-over-engine heavy truck with a box trailer on a two-lane highway through tropical "
         "highlands at dawn, palms and bamboo at the roadside, mist lying in the valleys, "
         "three-quarter side view, long lens compression"),
    Shot("couple_open", 6.0, "wide", "in", "warm",
         "a semi-trailer being released from a tractor unit at dawn: the trailer standing on its landing "
         "legs, the tractor pulling away, the coupling gap opening, backlit dust, wide three-quarter view"),
    Shot("card", 6.0, "wide", "still", "steel", ""),   # 落版由代码画，不用底图
]

TOTAL_FRAMES = int(round(DUR * FPS))


def _starts() -> list[float]:
    out, t = [], 0.0
    for s in SHOTS:
        out.append(t)
        t += s.dur
    return out


STARTS = _starts()
assert abs(sum(s.dur for s in SHOTS) - DUR) < 1e-6, sum(s.dur for s in SHOTS)


# ---------------------------------------------------------------- 旁白
@dataclass
class Voice:
    idx: int
    t: float
    text: str
    en: str
    window: float
    voice: str = "zh-CN-YunxiNeural"
    rate: str = "-14%"
    pitch: str = "-2Hz"


VOICES: list[Voice] = [
    Voice(1, 1.6, "一台重卡，四只轮子先落地。",
          "Four wheels take the road first.", 8.2),
    Voice(2, 27.4, "第五个轮子不落地。它只做一件事——接住重量。",
          "The fifth never touches it. It does one thing: it carries the load.", 9.6),
    Voice(3, 45.4, "牵引座。行内叫它，第五个轮子。",
          "The coupling plate. The trade calls it the fifth wheel.", 8.4),
    Voice(4, 54.0, "在茅箭，一千一百五十家企业，做的是同一件事。",
          "In Maojian, 1,150 suppliers do the same thing.", 8.4),
    Voice(5, 63.6, "把别人的重量，接住。",
          "They carry someone else's load.", 7.2),
    Voice(6, 86.0, "一套同步器，国内市场九成。",
          "Its synchronisers hold nine tenths of the domestic market.", 8.8),
    Voice(7, 100.4, "四十万辆的产能，从一个山口出发。",
          "Capacity for 400,000 vehicles a year, out of a mountain pass.", 9.4),
    Voice(8, 116.6, "这座城没有海。",
          "This city has no sea.", 7.0),
    Voice(9, 125.6, "于是它把海，装进了车轮。",
          "So it carried the sea out on wheels.", 8.6),
    Voice(10, 137.0, "四只轮子带它走。第五个轮子，让它接住世界。",
          "Four wheels carry it. The fifth lets it hold the world.", 9.4),
]

assert all(v.t + v.window <= DUR for v in VOICES)


# ---------------------------------------------------------------- 屏幕字（除旁白外的定位字）
@dataclass
class Label:
    key: str
    t0: float
    t1: float
    text: str
    sub: str = ""
    pos: str = "left"     # left / center


LABELS: list[Label] = [
    Label("n1150", 56.6, 62.8, "1,150", "汽车及零部件企业｜auto & parts suppliers", "left"),
    Label("n90", 88.4, 93.2, "90%", "同步器 · 国内市场份额｜synchroniser share, China", "left"),
    Label("n40", 102.6, 107.6, "400,000", "整车及专用车年产能｜annual capacity", "left"),
]


# ---------------------------------------------------------------- 音效
@dataclass
class Cue:
    t: float
    kind: str
    gain: float = 1.0


CUES: list[Cue] = [
    Cue(11.0, "tyre", 0.8),        # 轮子压水
    Cue(19.2, "metal", 0.5),       # 底盘金属
    Cue(35.2, "grease", 0.4),      # 倒车前的液压
    Cue(45.4, "pin", 1.0),         # 牵引销落座（全片第一声「咔」）
    Cue(53.0, "press", 0.8),       # 钢卷落垛
    Cue(61.0, "weld", 0.85),       # 焊接
    Cue(69.0, "spray", 0.5),
    Cue(76.0, "hoist", 0.7),
    Cue(85.0, "gear", 0.5),
    Cue(93.0, "caliper", 0.35),
    Cue(107.0, "engine", 0.8),     # 出厂
    Cue(116.0, "wind", 0.55),
    Cue(125.0, "horn", 0.6),       # 港口
    Cue(135.0, "wind2", 0.5),
    Cue(144.2, "unlock", 0.9),     # 锁钩打开
    Cue(147.8, "pin", 1.0),        # 最后一声「咔」，画面已黑
]


# ---------------------------------------------------------------- 落版
CARD = {
    "line1": "茅箭",
    "line2": "中国商用车之都",
    "line3": "第五个轮子",
    "line4": "MAOJIAN · SHIYAN",
    "line5": "THE FIFTH WHEEL COUPLING",
    "foot": "本片由生成式 AI 制作｜MADE WITH GENERATIVE AI　·　2026",
}

# 字幕安全区（像素）
SUB = {"x": 132, "y_cn": 884, "size_cn": 46, "size_en": 28, "gap": 52, "max_w": 1180}


# ---------------------------------------------------------------- 每个镜头承担的论证
# 判据是「去掉这一镜，论证塌不塌」；写不出来就说明这个镜头可以删。
NOTES = {
    "dawn_road": "开局交代地理：山、路、第一辆车",
    "wheel_close": "「轮子先落地」——拍到物理层面",
    "four_wheels": "四只轮子成排，给「第五个」留出空位",
    "under_body": "钻到车底，视线开始找那个不落地的轮子",
    "couple_wide": "把「接住」拍成事件：牵引车倒进挂车底下",
    "couple_low": "从地面仰视，重量真的从上面压下来",
    "coil": "产业链起点：一块钢",
    "weld": "车身成形",
    "paint": "涂装＝出厂的体面",
    "assembly": "合装＝一台车的完成",
    "rack": "成排的同一个零件＝「九成」的视觉论证",
    "caliper": "精度：千分尺上的那一格",
    "screen": "数字化：智慧工厂的另一半",
    "gate_out": "出厂，第一次朝向外面",
    "highway": "山口与桥——「从一个山口出发」",
    "port": "第一次见海",
    "overseas": "海的那边，同一种车",
    "couple_open": "反向动作：分开，才说明刚才接住过",
    "card": "落版",
}
for _s in SHOTS:
    _s.note = NOTES.get(_s.key, "")


# ---------------------------------------------------------------- 黑场（留给最后一声「咔」）
# 从 BLK0 开始 0.6 s 落黑，到 CARD_IN 一直是黑；落版 CARD_IN 起淡入。
# verify.py 用同一对数字判「这一段本来就该是黑的」，不再把设计当缺陷。
BLK0 = 147.55
BLK1 = 148.15
CARD_IN = 150.0
