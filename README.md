# 《第五个轮子》/ THE FIFTH WHEEL

十堰市茅箭区首届 Token 产业 IP 创意大赛 · **AI 产业宣传片**赛道
156 秒 / 1920×1080 / 24fps / H.264

> 一台重卡有四只轮子走路。第五个轮子不走路——它是牵引座，行内叫它「第五个轮子」，
> 只做一件事：接住重量。茅箭有 1,150 家汽车及零部件企业，绝大多数不造整车；
> 它们做的正是这一件事。
>
> 这座城没有海。于是它把海，装进了车轮。

作品页：`site/index.html`　｜　成片：`out/maojian.mp4`

## 一条命令验收

```bash
py -3 verify.py          # 44 条断言，退出码即结论
```

## 复现

```bash
py -3 render.py --prep           # 底图去水印、放大、锐化 → keyframes/plate/
py -3 audio.py                   # 旁白 + 合成音效 → out/audio.wav
py -3 scripts/master_audio.py    # −16 LUFS 母带 → out/audio-master.wav
py -3 render.py --seg 1/4        # 任意分段并行（渲染约 0.11 s/帧/进程）
py -3 render.py --seg 2/4
py -3 render.py --seg 3/4
py -3 render.py --seg 4/4
py -3 render.py --concat         # 剥音轨 → 拼 → 挂母带 → out/maojian.mp4
py -3 render.py --shotlist       # docs/02-SHOTLIST.md（由 story.py 生成）
py -3 build_site.py              # site/ + 字体子集 + 封面 + 六帧
py -3 verify.py
```

依赖：Python 3.8+、`opencv-python`、`numpy`、`pillow`、`edge-tts`、`fonttools`、`brotli`、ffmpeg。
底图不随仓库提供（生成物，体积大）；`keyframes/raw/*.png` 是 18 张 1536×1024 的生成底本，
按 `story.SHOTS[*].prompt` 可重新生成。

## 结构

```
story.py          单一来源：19 镜、时长、运镜、调色、10 句旁白、字卡、16 个音效点、落版
render.py         前处理 / 窗口运镜 / 调色 / 颗粒 / 字卡 / 字幕 / 分段编码 / 拼接挂音 / 分镜表
audio.py          旁白（edge-tts）+ numpy 合成的 drone / 脉冲 / 垫音 / 16 个事件音
verify.py         44 条断言（容器、时长、色彩、响度、逐镜抽检、版式、文档一致性）
build_site.py     作品页 + 中文子集字体 + 封面 + 六帧
scripts/          master_audio.py（两遍 loudnorm）
docs/             00 赛场 / 01 概念 / 02 分镜表（生成）/ 03 数据出处 / 04 做法 / 05 提交
submission/       作品简介、创作说明（可直接粘贴到平台表单）
site/             作品页（含成片、六帧、分镜、创作说明）
```

## 两条规矩

1. **数字必须有出处。** 片子里只有 1,150 家、90%、400,000 辆产能三个数字，
   每条都能点到官方原文（`docs/03-SOURCES.md`）。查不到的一律不写。
2. **每一镜都要能说出「去掉它，论证塌不塌」。** 19 条论证写在 `story.NOTES`，
   生成到分镜表，验收脚本反向核对。

## 版权

画面由生成式 AI 制作；旁白为语音合成；音乐与音效由 `audio.py` 现场合成，无外部素材；
字体使用系统授权的 Noto Sans SC（作品页内嵌子集）。无第三方商标、车牌、可识别品牌。
