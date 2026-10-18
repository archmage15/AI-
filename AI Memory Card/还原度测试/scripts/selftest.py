# -*- coding: utf-8 -*-
r"""还原度测试 · 自检（不跑模型，只验算法）

为什么要它
----------
项目零测试，但算法改过12 轮。历史上多次"静默错误"——数字照出、报告照出，
但结论是错的：
  · 长音频不画图，PNG 里是上一次的旧图（10-17）
  · 陈旧的 节奏对比.json 混进长音频报告，显示"净时长 3.70s"（10-17）
  · history 把单句 63.2 和长音频 67.2 混着比，写出"↑+4.0"假进步（10-18/P1）
  · 半截检测用 delete 段判漏头 → 抓不到（10-16）
  · 幂平均 p=1 时"10 块坏 2 块"被摊平，比"只是慢了"分还高（09-28）
这些都不是"代码崩了"，是"代码好好地算出了一个没意义的数"。单元测试挡不住
崩溃，但**能挡住口径被改坏**——所以每个用例都锚定一次真实踩过的坑。

怎么用
------
    python scripts\selftest.py

约1 秒跑完（纯计算，不加载 torch / 模型 / 音频）。
退出码 0 = 全过；1 = 有失败。可直接挂 git 钩子或改完代码先跑一遍。

设计约束
--------
1. **只用纯函数**。不碰模型、不碰音频、不写 output/ —— 自检不该产生副作用，
   否则它自己就成了干扰源（这正是 10-17 那几个 bug 的形态）。
2. **每个用例写明锚定哪个历史坑**，见各用例上方注释。将来有人想删某个用例，
   能看出它护的是什么。
3. 断言用精确值或窄容差；只在"数学上本该有浮点误差"处用 toFixed 风格比较。
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compare_long as cl
import compare_phoneme as cp
import compare_rhythm as cr
import overall_score as os_

# ---- 小工具 --------------------------------------------------------------

_PASS = []
_FAIL = []
_group = ""


def group(name):
    global _group
    _group = name
    print("\n%s" % name)


def check(desc, got, want):
    """断言 got == want；不等就记失败并打印差异。"""
    ok = got == want
    mark = "OK  " if ok else "FAIL"
    line = "  [%s] %s" % (mark, desc)
    if not ok:
        line += "\n         期望 %r\n         实际 %r" % (want, got)
    print(line)
    (_PASS if ok else _FAIL).append((_group, desc))
    return ok


def check_near(desc, got, want, tol):
    """数值断言（带容差）。用于浮点计算结果。"""
    ok = got is not None and abs(got - want) <= tol
    mark = "OK  " if ok else "FAIL"
    line = "  [%s] %s" % (mark, desc)
    if not ok:
        line += "\n         期望 %.4f ±%.4f\n         实际 %r" % (want, tol, got)
    print(line)
    (_PASS if ok else _FAIL).append((_group, desc))
    return ok


def check_true(desc, got):
    return check(desc, bool(got), True)


# ==========================================================================
# 1. 半截检测 detect_partial_coverage
#    锚定：10-16 踩过的坑 —— 用 opcodes 的 delete 段判漏头**抓不到**
#    （跳开头时 difflib 判成 replace 不是 delete）。现在靠匹配块跨度。
# ==========================================================================
group("1. 半截检测（锚定 10-16 漏头抓不到的坑）")

# 参考序列：**非周期**的音素串。
# ⚠️ 别用 ["s","oʊ","w"]*N 这种循环节律——截取 REF[20:40] 得到的片段
# 与 REF[0:20] 完全相同，difflib 会把它对到 index 0，"漏头"就测不出来了
# （真实代码没错，是测试数据自欺）。非周期串保证任意两段都不会撞。
REF40 = (["s", "oʊ", "w", "ɛ", "r", "t", "n", "i", "æ", "b"]
         + ["k", "d", "m", "ɛ", "l", "p", "h", "oʊ", "g", "v"]
         + ["θ", "z", "f", "ʃ", "j", "u", "aɪ", "d", "b", "l"]
         + ["f", "eɪ", "t", "ʃ", "ɔ", "r", "m", "h", "k", "w"])


def slice_ref(start, length):
    """取参考序列的连续片段（模拟"只跟读了这一段"）。

    ⚠️ 片段必须**就是参考序列本身**，才能产生真实匹配块。
    若换成全新的假音素（与参考无交集）→ 匹配块为空 → 直接 return empty，
    测的就不是漏尾逻辑而是"完全没读"的另一个分支了。
    """
    return REF40[start:start + length]


# 完整模仿：长度接近参考，不该判半截
r = cp.detect_partial_coverage(REF40, slice_ref(0, 38))
check("完整模仿不判半截", r["半截"], False)
check("完整模仿覆盖比例=1.0", r["覆盖比例"], 1.0)

# 漏尾：只读前半（20/40 = 50%，明显短一截且尾漏 20 音）
r = cp.detect_partial_coverage(REF40, slice_ref(0, 20))
check("漏尾判半截", r["半截"], True)
check("漏尾头漏=0", r["头部漏读音数"], 0)
check_near("漏尾覆盖比例≈0.5", r["覆盖比例"], 0.5, 0.05)

# 漏头：只读后半（跳开头）—— 这是 10-16 用 delete 段抓不到的那个
r = cp.detect_partial_coverage(REF40, slice_ref(20, 20))
check("漏头判半截", r["半截"], True)
check("漏头覆盖起始=20", r["覆盖起始位置"], 20)
check("漏头尾部漏读=0", r["尾巴漏读音数"], 0)

# 头尾都漏（只读中间一小段）→ 也该判半截
r = cp.detect_partial_coverage(REF40, slice_ref(18, 12))
check("掐头去尾只读中间 → 判半截", r["半截"], True)

# 只漏 1 个音（零星漏读不算整段没读）
r = cp.detect_partial_coverage(REF40, slice_ref(0, 39))
check("零星漏 1 音不判半截", r["半截"], False)

# ⚠️ 三个门槛里**只有 LEN_RATIO 真正生效**（变异测试 + 边界扫描实测得出）：
#   LEN_RATIO=0.75 要求 mine <= 30/40；MIN_TAIL_RATIO=0.10 只要漏 ≥4 音就够；
#   MIN_TAIL=3 更宽松。逐点扫描 40 音参考的真实边界：
#     mine=30（漏10 音/25%）→ 判半截   mine=31（漏9 音/22.5%）→ 不判
#   边界恰好等于 LEN_RATIO，与 10% 那条线无关 —— MIN_TAIL 与 MIN_TAIL_RATIO
#   在短句上都是死门槛（真要生效得漏 ≥25%，而那时 LEN_RATIO 早就拦下了）。
#   所以只锁 LEN_RATIO 这条真实边界，别为 MIN_TAIL 写用例（改了测不出差异）。
#
# ⚠️ 参考数据必须**全局互异**。变异测试踩过两次坑：
#   ① ["s","oʊ","w"]*13 循环 → slice(20,20) 片段与开头完全相同，difflib 对到
#      index 0，"漏头"根本测不出来；
#   ② REF40+REF40 拼 100 音 → 尾部与前段重复，尾漏被算成 0。
#   两次都是**用例自己骗自己**，不是代码 bug。
REF_UNIQ = ["ph%02d" % i for i in range(40)]
r = cp.detect_partial_coverage(REF_UNIQ, REF_UNIQ[:30])
check("mine=30(正好75%) → 判半截（LEN_RATIO 边界内）", r["半截"], True)
r = cp.detect_partial_coverage(REF_UNIQ, REF_UNIQ[:31])
check("mine=31(超过75%) → 不判半截（未短一截）", r["半截"], False)
r = cp.detect_partial_coverage(REF_UNIQ, REF_UNIQ[:35])
check("mine=35(87.5%，只漏5音) → 不判半截", r["半截"], False)

# LEN_RATIO 前提：模仿未短到 75% → 根本不该进半截判断（哪怕尾部漏很多）
r = cp.detect_partial_coverage(REF_UNIQ, REF_UNIQ[:36])
check("漏4音但长度仍90% → 不判半截（未短一截）", r["半截"], False)

# 模仿比参考长（读多了）→ 不可能是半截
r = cp.detect_partial_coverage(REF40, REF40 + ["n", "e", "w"])
check("模仿更长不判半截", r["半截"], False)

# 中间读错几个音但整体长度够 → 不该判半截
mixed = list(slice_ref(0, 38))
for i in (5, 20, 33):
    mixed[i] = "WRONG"
r = cp.detect_partial_coverage(REF40, mixed)
check("中间错音不判半截", r["半截"], False)

# 模仿为空 → 不崩，返回"未覆盖任何"
r = cp.detect_partial_coverage(REF40, [])
check("空模仿 → 不判半截且不崩", r["半截"], False)


# ==========================================================================
# 2. 节奏分score_blocks
#    锚定：09-28 的两个坑 —— ①不归一化则"整体慢 15%"被判成"每块都错"
#    ②算术平均把"坏 2 块"摊平。幂平均 p=3 是为此引入的。
# ==========================================================================
group("2. 节奏分（锚定 09-28 归一化 + 幂平均的坑）")


def pairs_from(durations_r, durations_m):
    """按两侧时长列表造score_blocks 的输入。"""
    return [{"序号": i + 1, "块": "w%d" % (i + 1), "连读": False, "角色": "A",
             "原音频时长": dr, "我的时长": dm}
            for i, (dr, dm) in enumerate(zip(durations_r, durations_m))]


# ① 整体等比例变慢：归一化后应接近 100 分（不惩罚"就是语速慢"）
items, sc = cr.score_blocks(pairs_from([0.40] * 10, [0.46] * 10))
check_near("整体慢 15% → 节奏分≈100（归一化生效）", sc, 100.0, 0.5)
check("整体慢 15% → 无块被判'要读慢一些'", len([i for i in items if i["判定"] != "差不多"]), 0)

# ② 10 块里坏 2 块：幂平均必须比算术平均扣得多
base = [0.40] * 10
slow_blocks = [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.18, 0.22, 0.40, 0.40]
items, sc_bad = cr.score_blocks(pairs_from(base, slow_blocks))
check_true("10 块坏 2 块 → 分明显低于 100", sc_bad < 85)
# 时长被读短 → 相对差为负 → 判「要读慢一些」（不是"读快"，方向别搞反）
check("读短的2 块判'要读慢一些'",
      sorted(i["序号"] for i in items if i["判定"] == "要读慢一些"), [7, 8])

# 反向：时长被读长 → 判「要读快一些」
fast_blocks = [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.62, 0.58, 0.40, 0.40]
items, _ = cr.score_blocks(pairs_from(base, fast_blocks))
check("读长的 2 块判'要读快一些'",
      sorted(i["序号"] for i in items if i["判定"] == "要读快一些"), [7, 8])

# 小幅偏差（±10%）不该被判为错误
mild = [0.40 * 1.1] * 10
items, _ = cr.score_blocks(pairs_from(base, mild))
check("小幅偏差 10% → 全部判'差不多'",
      len([i for i in items if i["判定"] != "差不多"]), 0)

# 幂平均确实比算术平均更敏感（p=3 vs p=1）
# ⚠️ 必须直接调 aggregate_penalty 并显式传 power：aggregate_penalty 的默认参数
# `power=AGG_POWER` 是**函数定义时**绑定的，改AGG_POWER 变量它不读。
# 换句话说：用 score_blocks 测不出 AGG_POWER 被改坏（变异测试实测踩过）。
diffs = [0.0] * 8 + [0.55, 0.62]
check_true("幂平均 p=3 比算术平均更敏感",
           cr.aggregate_penalty(diffs, power=3.0) > cr.aggregate_penalty(diffs, power=1.0))

# 锁定 AGG_POWER 必须是 3.0 —— 直接读模块常量（score_blocks 读不到，见上）
check("AGG_POWER 仍为 3.0（改它会退回09-28 的'坏块被摊平'）",
      cr.AGG_POWER, 3.0)

# p=3 在"8好2坏"上应显著低于 p=1（这是选p=3 的唯一理由，必须锁住）
check_true("p=3 对坏块比 p=1 扣得更狠",
           cr.aggregate_penalty(diffs, power=3.0) > cr.aggregate_penalty(diffs, power=1.0) + 0.1)

# 空输入不崩
items, sc_empty = cr.score_blocks([])
check("空输入返回空列表+0分", (items, sc_empty), ([], 0.0))


# ==========================================================================
# 3. 停顿判定 find_boundary_pauses
#    锚定：§5.5 连带坑 —— 换人处的停顿是**预期停顿**，不计入处数。
#    实测原音频换人 gap 880ms、模仿 680ms，旧口径会全记成停顿错误。
# ==========================================================================
group("3. 停顿判定（锚定 §5.5 换人处不应算停顿错误）")

blocks_same_role = [
    {"词": ["where"], "start_ms": 0, "end_ms": 400, "角色": "A"},
    {"词": ["are"], "start_ms": 900, "end_ms": 1200, "角色": "A"},   # gap 500ms 同人
]
result, counted = cr.find_boundary_pauses(blocks_same_role)
check("同人 gap 500ms → 记为停顿", len(counted), 1)
check("同人停顿 非换人处", counted[0]["换人处"], False)

blocks_turn = [
    {"词": ["terry"], "start_ms": 0, "end_ms": 400, "角色": "A"},
    {"词": ["oh", "im"], "start_ms": 1280, "end_ms": 1600, "角色": "B"},  # gap 880ms 换人
]
result, counted = cr.find_boundary_pauses(blocks_turn)
check("换人 gap 880ms → 出现但不计入处数", len(counted), 0)
check("换人处被标记出来", result[0]["换人处"], True)

# gap 不足阈值 → 不算停顿
blocks_tight = [
    {"词": ["a"], "start_ms": 0, "end_ms": 400, "角色": "A"},
    {"词": ["b"], "start_ms": 450, "end_ms": 800, "角色": "A"},   # gap 50ms
]
_, counted = cr.find_boundary_pauses(blocks_tight)
check("gap 50ms → 不算停顿", len(counted), 0)


# ==========================================================================
# 4. 词级错误 compute_word_errors
#    锚定：09-23 定的基准 —— 以**原音频实际音素**为准，不是参考文本。
#    连读吞音若原音频也吞、学生也吞 → 不算错（反例：正确连读成 wanna 会被误报）。
# ==========================================================================
group("4. 词级错误（锚定 09-23 基准=原音频实际音素）")

wb = [{"word": "want", "角色": "A"}, {"word": "to", "角色": "A"}]

# 原音频吞音（wanna 只有一个音），学生也吞成一样 → 不算错
ref_owner = [0, 1]
my_owner = [0, 1]
errs = cp.compute_word_errors(wb, ["w", "n"], ref_owner, ["w", "n"], my_owner)
check("原音频也吞音、学生同样吞 → 不报错", len(errs), 0)

# 学生确实读错→ 要报
# ⚠️ owner 数组长度必须等于音素数（compare_word_errors 按 len(音素) 索引 owner）
errs = cp.compute_word_errors(wb, ["w", "n"], [0, 1], ["w", "t", "o"], [0, 1, 1])
check_true("学生读错 → 报出问题词", len(errs) >= 1)

# 只错1 个音、且占该词比例低（4 音里错 1 = 25% < 40%，且错音数 1 < 2）
# → 低于双阈值，不报（避免把吹毛求疵的差异全标成错）
errs = cp.compute_word_errors(
    [{"word": "bank", "角色": "A"}],
    ["b", "a", "n", "k"], [0, 0, 0, 0],
    ["b", "a", "n", "x"], [0, 0, 0, 0])
check("只错 1 个音(25%) → 不报（低于双阈值）", len(errs), 0)

# 只错 1 个音但该词很短（2 音里错 1 = 50% ≥ 40%）→ 要报
errs = cp.compute_word_errors(
    [{"word": "at", "角色": "A"}],
    ["æ", "t"], [0, 0],
    ["æ", "b"], [0, 0])
check("2音词错1个(50%) → 报（比例超阈值）", len(errs), 1)

# 原音频与模仿都是空段 → 跳过，不崩不报
errs = cp.compute_word_errors([{"word": "x", "角色": "A"}], [], [], [], [])
check("双方全空 → 不报不崩", len(errs), 0)


# ==========================================================================
# 5. 长音频句子分配 assign_mine_to_sentences
#    锚定：长音频模式的核心 —— 全局对齐后把模仿音素分配进各句，
#    区间内一个都没对上的句 = 漏句（不计分）。
# ==========================================================================
group("5. 长音频逐句分配与漏句判定")

text_ph = ["a", "b", "c", "d", "e", "f", "g", "h"]
sentences = [
    {"句号": 1, "文本": "ab", "a": 0, "b": 2},
    {"句号": 2, "文本": "cd", "a": 2, "b": 4},
    {"句号": 3, "文本": "ef", "a": 4, "b": 6},
    {"句号": 4, "文本": "gh", "a": 6, "b": 8},
]
import difflib

# 完整模仿：每句 2 音全对上 → 无漏句
sm = difflib.SequenceMatcher(a=text_ph, b=list(text_ph), autojunk=False)
assign = cl.assign_mine_to_sentences(sm.get_opcodes(), sentences, 8, 8)
check("完整模仿 → 4 句都有分配", sorted(assign.keys()), [1, 2, 3, 4])
check("完整模仿 → 无空句", sum(1 for v in assign.values() if not v), 0)

# 只读前 2 句 → 后 2 句空 → 判漏句
mine_half = text_ph[:4]
sm = difflib.SequenceMatcher(a=text_ph, b=mine_half, autojunk=False)
assign = cl.assign_mine_to_sentences(sm.get_opcodes(), sentences, 8, 4)
check("只读前2句 → 句3无分配", assign.get(3), [])
check("只读前2句 → 句4无分配", assign.get(4), [])

# 逐句评测：完整句不该判漏句
sent = {"句号": 1, "文本": "ab", "a": 0, "b": 2, "_mine": [0, 1]}
wb_local = [{"word": "ab", "a": 0, "b": 2, "角色": "A"}]
res = cl.evaluate_sentence(sent, text_ph, list(text_ph), [0.0, 0.1], wb_local)
check("完整句不判漏句", res["漏句"], False)
check("完整句匹配=2", res["匹配"], 2)

# 空句（一个模仿音素都没分到）→ 判漏句
# ⚠️ 句子区间必须 >= MIN_SENT_PH(3) 音，否则会被判成「碎句-忽略」而不判漏句。
# 这里用 a=2,b=6（4 音），确认是"有内容的句但完全没跟读"→ 漏句。
sent = {"句号": 3, "文本": "cdef", "a": 2, "b": 6, "_mine": []}
res = cl.evaluate_sentence(sent, text_ph, list(text_ph), [0.0], [])
check("空句(4音)判漏句", res["漏句"], True)

# 少于 3 音的空句 → 碎句忽略，不是漏句（10-17踩过：讯飞把 "There. Are" 拆两句）
sent = {"句号": 9, "文本": "g", "a": 6, "b": 7, "_mine": []}
res = cl.evaluate_sentence(sent, text_ph, list(text_ph), [0.0], [])
check("碎句(1音)空 → 标忽略不判漏句", (res["忽略"], res["漏句"]), (True, False))


# ==========================================================================
# 6. 综合分公式
# ==========================================================================
group("6. 综合分公式与等级")

stat = {"原生音数": 100, "模仿音数": 100, "替换音数": 20, "缺失音数": 5, "多余音数": 5}
check_near("发音分 (1-替换/基准)×100", os_.score_pronunciation(stat), 80.0, 0.01)
check_near("音变分 (1-(缺失+多余)/基准)×100", os_.score_phoneme_change(stat), 90.0, 0.01)
check("空输入 → None（不当0 分处理）", os_.score_pronunciation(None), None)
check("等级边界 90=优", os_.grade(90), "优")
check("等级边界 59.9=待加强", os_.grade(59.9), "待加强")
check("None → —", os_.grade(None), "—")


# ==========================================================================
# 7. history 模式隔离（P1 · 10-18）
#    锚定：63.2（4秒单句）与 67.2（72秒长音频）被写成"↑+4.0"假进步。
# ==========================================================================
group("7. history 跨模式隔离（锚定 P1 假进步）")

hist_mixed = [
    {"时间": "t0", "综合分": 63.2},                                # 旧记录：无模式
    {"时间": "t1", "综合分": 55.0, "模式": "单句"},
    {"时间": "t2", "综合分": 67.2, "模式": "长音频"},
]
check("长音频只跟长音频比（命中 67.2）",
      os_.find_prev_same_mode(hist_mixed, "长音频")["综合分"], 67.2)
check("单句只跟单句比（命中 55.0，不被 67.2 污染）",
      os_.find_prev_same_mode(hist_mixed, "单句")["综合分"], 55.0)
check("全旧记录无模式 → 拒绝出对比",
      os_.find_prev_same_mode(hist_mixed[:1], "长音频"), None)
check("空历史 → None", os_.find_prev_same_mode([], "单句"), None)
check("模式判定：有长音频段", os_.detect_mode({"长音频": {}}), "长音频")
check("模式判定：无长音频段", os_.detect_mode({"相似度": 63.2}), "单句")
# 空 dict 也算长音频（存在即长音频）—— 防止 compare_long 写空段时被误判成单句
check("模式判定：长音频段存在但为空 → 仍算长音频",
      os_.detect_mode({"长音频": {}}), "长音频")


# ==========================================================================
# 汇总
# ==========================================================================
total = len(_PASS) + len(_FAIL)
print("\n" + "=" * 62)
print("自检结果：%d/%d 通过" % (len(_PASS), total))
if _FAIL:
    print("失败用例：")
    for g, d in _FAIL:
        print("   · [%s] %s" % (g, d))
    print("=" * 62)
    sys.exit(1)
print("全部通过。")
print("=" * 62)
sys.exit(0)