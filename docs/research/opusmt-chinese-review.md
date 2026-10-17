# Simplified Chinese OPUS-MT: native-review brief and what it changed

**Date:** 2026-07-30 (outcome added 2026-08-12)
**Status:** Chinese ships in `src/translation_models/manifest.json`. The preferred layer is **Argos
OpenTech `en_zh` 1.9**, not the raw Helsinki tc-big checkpoint this brief first reviewed; the base is
Helsinki `opus-mt-en-zh`. A native reader's informal verdict on the shipped output (2026-08-12) was "good
enough". The brief below is kept because its findings are why the model and the validation code changed.

## How Chinese differs from Spanish

For Spanish the tc-big → base hybrid mattered a lot: tc-big had a real character-coverage hole (no `ü`, no
accented capitals, ~100 symbols) and the base caught it constantly. **For Chinese it barely engages.** On the
396-tag corpus the hybrid picked tc-big **362 times (91%)** and fell back to base **twice (0.5%)**. So the
quality bar for Chinese is almost entirely tc-big's own word choice; if that is bad, the architecture won't
quietly rescue it.

## The corpus

The same 396 real tags and 6 real English descriptions the Spanish review used (real VLM output on posters, a
screenshot, signage, a product shot and a nature photo, chosen because quote-, brand- and spec-heavy text is
where translation problems hide). They come from golden-set runs, some on private entries, so the raw corpus
isn't committed. Of the 396 tags, 89 agree between tc-big and base outright (mostly `* logo` brand tags,
which skip translation by design) and 281 disagree; the appendix is a random sample of 70.

## Questions the review asked

1. Does the shipped hybrid read as correct, natural Simplified Mandarin?
2. **Word spacing:** some translations come out spaceless (`一辆红色跑车停在摩天大楼附近`), others with
   spaces between word groups. Standard written Chinese doesn't space words; should the output strip spaces
   between CJK characters? Still an open policy call.
3. Is anything wrong in a way that would look bad in a published tag?

## Known-bad tail of the raw tc-big checkpoint

Same failure shape Spanish had on isolated short phrases (`cello`→
`violín` was Spanish's example) — the MT model occasionally produces a
confidently wrong word instead of a near-miss:

| English | raw tc-big | base | Note |
|---|---|---|---|
| `fork` | 切换导航 ("toggle navigation") | 叉子 (correct) | tc-big wrong, base right — rare, opposite of Spanish's usual pattern |
| `handrail` | 加盟优势 ("franchise advantage") | 扶手 (correct) | tc-big wrong, base right |
| `coelacanth` *(not in the 396 corpus, found separately)* | 长颈鹿 ("giraffe") | — | flat hallucination |
| `nestle logo` | *(stays English — brand tags skip translation)* | — | for reference, if translated: tc-big → 雀巢标志 (correct, official name), base → 巢徽标 (wrong, literal "nest emblem") |

**Two more severe than the above, found in the random sample (see the full
table below) — flagging first since they're not just wrong-word errors,
they look like the model latching onto unrelated real-world Chinese
boilerplate/brand text:**

| English | raw tc-big | What it actually is |
|---|---|---|
| `date 1989-2019` | 粤icp备18031212号-1 | A real-format Chinese website ICP business-registration footer number (the kind of legal registration text every Chinese website must display) — completely unrelated to a date, and a strange, specific thing to hallucinate |
| `mug handle` | 淘宝旺旺 | The actual product name of Taobao's chat/messenger app ("Taobao Wangwang") — again totally unrelated to a mug |

This was the deciding finding: confident nonsense pulled from unrelated
real-world Chinese text is a systemic defect, not occasional noise, and it
is what retired the raw tc-big checkpoint (next section).

## What changed after the review

**1. The raw tc-big checkpoint was retired** for Argos OpenTech's `en_zh` 1.9 package: the same OPUS-MT
lineage (CC-BY-4.0), with different curation and checkpoint selection. It loads as a plain CTranslate2 model
plus `sentencepiece.model`, with the same calls as every other layer and no new dependency. Boilerplate
hallucinations went from 9/396 to **0/396**. (The same swap was tried for Spanish and rejected: Argos
`en_es` left 89/396 tags untranslated vs 46/396 for the shipped hybrid.)

**2. A real footage library found two defect classes the 396-tag corpus missed entirely.** A 153-file
metadata export (images, video, mixed content) was translated end to end:

- **Subtitle-styling codes leaking into output** on ordinary words (`bucket`, `curtain`, `fur`): an ASS/SSA
  code like `{\fn黑体\fs20…}` prepended to the translation, and in 3 of 5 cases *replacing* it with a
  profane phrase. This is subtitle-corpus contamination; it happened with the correct target prefix, on the
  Argos model. Neither `<unk>` detection nor the boilerplate heuristic caught it, because the output is
  clean, confident, well-formed text. **Fixed** by `_looks_corrupted()` in `src/opus_translate.py`: a
  subtitle-code regex plus a 3-identical-words repetition detector (`fur` had looped:
  `皮毛 皮毛 皮毛 皮毛`), run on every layer's output.
- **The base layer can emit `<unk>` for Chinese.** The "base has full character coverage" property holds for
  Spanish but not here: `basil leaf`, `skewers` and `quiche` produced a visible `⁇` through the base itself.
  The cascade used to accept the last layer unconditionally. **Fixed:** every layer is validated, the last
  one included, and when all fail the original English is kept.

**3. A fallback sentence is only as good as the base's version of it.** `translate_text` works sentence by
sentence, so one description can mix tc-big and base sentences by design. On a film-poster description,
tc-big hit `<unk>` in one sentence, so the hybrid correctly fell back to base for that sentence, but base
rendered "white (snow)" as 白色(现成), "ready-made". Falling back avoids the visible `⁇` glyph; it doesn't
guarantee the replacement is right.

**Lesson carried to Portuguese:** validate on a real library, not only on a reused synthetic corpus. The
396-tag corpus missed every one of the severe real-content defects.

## Appendix: tag disagreement sample, raw tc-big vs base (70 of 281, random seed 42)

| EN | raw tc-big (retired) | base | agree? |
|---|---|---|---|
| arch | 拱门 | 箭头 | no |
| audience | 观众中心 | 听众 | no |
| b-8145 | 浙公网安备 33038202002750号 | b 8145个。 | no |
| bag | 淘宝旺旺 | 软件包 | no |
| bicycle | 自行车配件 | 自行车 | no |
| boats | 乘船游览 | 船只 | no |
| bridge | 联系我们 | 桥 | no |
| cityscape | 城市景观 | 城市风景 | no |
| clouds | 阴,多云 | 云层 | no |
| coffee | 餐饮服务 | 咖啡 | no |
| collar | 领子 | 项圈 | no |
| door | 门禁系统 | 门 | no |
| field | 领域 | 字段 | no |
| helmet | 在线客服 | 头盔 | no |
| hill | 希尔 | 山头 | no |
| jeans | 牛仔 裤 | 牛仔裤 | no |
| motorcycle | 摩托车配件 | 摩托车 | no |
| park | 园区 | 公园 | no |
| pavement | 路面设施 | 人行道 | no |
| pickle | 密码 * | 酸黄瓜 | no |
| platforms | 研发平台 | 平台 | no |
| pole | 联系我们 | 极 | no |
| rapha | 说唱 | 拉贾 | no |
| rim | 轮廓 | 轮廓 | no |
| rocks | 石头 | 岩石 | no |
| rope | 联系我们 | 绳子 | no |
| scoreboard | 积分榜 | 记分牌 | no |
| string | 密码 * | 字符串 | no |
| strip | 条 | 条形 | no |
| t-shirt | 衬衫系列 | T恤衫 | no |
| tableware | 淘宝旺旺 | 表格软件 | no |
| tent | 联系我们 | 帐篷 | no |
| truck | 卡车系列 | 卡车 | no |
| wall | 墙 | 墙壁 | no |
| wristband | 加盟优势 | 腕带 | no |
| bill irwin name | 比尔·欧文 姓名 | 比尔 irwin 名称 | no |
| blurred background | 模糊背景颜色 | 模糊的背景 | no |
| blurry lights | 灯光模糊 | 模糊的灯光 | no |
| brick pavement | 路面砖系列 | 砖面 | no |
| car dashboard | 汽车仪表盘 | 汽车仪表板 | no |
| car trail | 行车路线 | 汽车足迹 | no |
| car window | 汽车窗系列 | 车窗 | no |
| chair umpire | 仲裁员 | 仲裁人 | no |
| cigar box | 雪茄礼盒 | 雪茄盒 | no |
| clock tower | 时钟塔 | 钟塔 | no |
| comment count icon | 评论计数图标 | 注释计数图标 | no |
| construction crane | 施工吊车 | 建筑起重机 | no |
| crazy mouse sign | 疯狂的鼠标标志 | 疯狂的鼠标符号 | no |
| cryptocurrencies link | 加密货币链接 | 加密链接 | no |
| date 1989-2019 | 粤icp备18031212号-1 | 1989-2019年 国家 | no |
| date 2019-? | 时间 2019-? | 2019年的约会? | no |
| dusk sky | 黄昏 天空 | 黄昏天空 | no |
| gloria sign | 格洛里亚标志 | Gloria 符号 | no |
| insta360 camera article | insta360 摄像头 | insta 360 相机文章 | no |
| jordan goldberg name | 乔丹·戈德堡名 | 约尔丹金山名称 | no |
| metal fence | 铁艺围栏 | 金属围栏 | no |
| metal pole | 铁杆系列 | 金属杆 | no |
| metal structure | 金属结构件 | 金属结构 | no |
| michael caine name | 迈克尔·凯恩姓名 | Michael cain 名称 | no |
| mug handle | 淘宝旺旺 | 杯子手柄 | no |
| multicolored jacket | 彩色夹克 | 多色夹克 | no |
| neon signs | 霓虹灯标 | 亮点符号 | no |
| plant pot | 植物盆栽 | 植物锅 | no |
| please sign | 请签名 | 请签个名 | no |
| ruler lines | 标尺行 | 标尺线条 | no |
| shirt collar | 衬衫领子 | 衬衫项圈 | no |
| shop awning | 店铺篷房 | 缝纫 | no |
| smoking kills warning | 吸烟警告 | 吸烟致死警告 | no |
| street lights | 路灯系列 | 街灯 | no |
| traffic lights | 交通信号灯 | 交通灯 | no |
