# -*- coding: utf-8 -*-
"""还原度对比 · 总装：综合评分 + 纠音报告

把前两步的产物合起来算一个"综合还原度"：
  ① 音素对比.json  → 发音分（音素替换）+ 音变分（缺音/多音）
  ② 节奏对比.json  → 节奏分 + 语调分

权重（用户定调：发音最重要，其余三项差不多）：
  发音 40%  ｜  音变 20%  ｜  语调 20%  ｜  节奏 20%

产出（两份，职责分开，都写在 output\\ 里）：
  - 还原度综合评分.json   给机器/AI：各项分值 + 权重 + 明细
  - 还原度综合报告.txt    给人看：综合分 + 分项分 + 问题清单 + 改进建议

目录约定（本脚本在 scripts\\ 里，用 ROOT 指回上一层）：
  audio\\   放录音     output\\  放结果（前两步写这里，本脚本从这里读）

依赖：只用标准库（读 json、写 txt），不需要 torch —— 秒出结果。
用法：先跑 音素对比 / 节奏对比（或直接双击 一键跑还原度.bat），再跑本脚本。
"""
import datetime
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 目录约定 ----
# 本脚本位于 <项目>\scripts\ 下，ROOT = 项目根（上一层）
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

PHONEME_JSON = os.path.join(OUT_DIR, "音素对比.json")
RHYTHM_JSON = os.path.join(OUT_DIR, "节奏对比.json")
OUT_JSON = os.path.join(OUT_DIR, "还原度综合评分.json")
OUT_REPORT = os.path.join(OUT_DIR, "还原度综合报告.txt")
HISTORY_JSON = os.path.join(OUT_DIR, "history.json")

# ---- 维度定义：全局唯一来源 ----
# 顺序 = 展示顺序；权重必须和为 1。
# 本文件不再出现第二处维度列表 —— 终端简报、报告、改进建议、payload 全部迭代它。
# 加/删维度：改这里 + collect_parts()，两处即可。
DIMENSIONS = [
    ("发音", 0.40),
    ("音变", 0.25),
    ("节奏", 0.20),
    ("重音位置", 0.15),
]
# 注：语调暂不出分（计划书 §4）—— 不进 DIMENSIONS、不参与加权；
#    语调分仍从 rhythm["语调分"] 读，仅供报告显示。

# ---- 评测模式（history.json 必须记，否则跨模式比分会误导）----
# 单句 = 原音频 ≤20s；长音频 = 逐句评测。两者的素材长度、评分口径都不同，
# 综合分**不可横向比**。混在一起比会得出"我进步了"的错误结论
# （实测踩过：63.2 是 4 秒单句、67.2 是 72 秒长音频，被写成"↑+4.0"）。
MODE_LONG = "长音频"
MODE_SINGLE = "单句"

# ---- 口径版本（history.json 必须记，否则跨版本比分会误导）----
# 为什么需要它（10-28 补的欠账）：
#   同一个「模式」下，**评分口径本身也会变**。10-26 修了「espeak 连读合并
#   导致词边界错位」的 P0 bug —— 修之前，报告里「单词 X 实读 Y」的指控
#   指错了词（"the" 头上装的是 lesson 的音）。修之后归因口径变了。
#   但 10-21 那条 60.6 分是**错位口径**算出来的，跟修复后的 60.3 分
#   直接相减，就会得出"我进步了"或"我退步了"的错误结论 ——
#   其实只是口径变了，跟练得好不好没关系。
#
# 纪律同「模式」隔离：**跨口径版本一律不对比**。
#   旧记录没有「口径版本」字段 → 视为未知版本 → 一律跳过。
#   宁可不给对比，也不给错的对比。
#
# 什么时候要 +1：
#   改了任何影响「分数怎么算」或「差异怎么归因」的东西时。
#   纯 UI 改动、纯文案改动 → 不用 +1。
SCORE_VERSION = 2
SCORE_VERSION_NOTE = (
    "2=10-26 修词边界错位（espeak 连读合并 → DP 对齐重建词边界）；"
    "1=修之前（词边界错位，词级指控不可信）"
)


def detect_mode(phoneme):
    """从音素对比.json 判定本次是长音频还是单句模式。

    判据用 `is not None` 而非真值判断：compare_long.py 一定会写「长音频」段
    （内含 模式/句子/四个分项等 7 个键），但若哪天它写了空段，
    `phoneme.get("长音频") or None` 会把"存在但为空"误判成单句。
    """
    return MODE_LONG if phoneme.get("长音频") is not None else MODE_SINGLE


# ---- 日志（统一模块，带时间戳；详细模式见 logkit.py 顶部说明）----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logkit import log, out, log_setup, log_done, log_step, log_error, VERBOSE


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_history():
    """读 history.json（历史分数列表）；不存在/损坏返回空列表。"""
    if not os.path.exists(HISTORY_JSON):
        return []
    try:
        with open(HISTORY_JSON, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def find_prev_same_mode(hist, mode, version=None):
    """在历史里**往回找**最近一条同模式、**同口径版本**的记录。

    为什么不能直接用 hist[-2]（旧版做法）：
      单句（4秒）与长音频（72秒）的综合分口径不同，跨模式比会得出
      "进步了 +4.0" 的假结论 —— 分数变化其实来自素材变化，不是练得更好。
      宁可不给对比，也不给错的对比。
    旧记录没有「模式」字段 → 一律跳过（无法判断模式 = 不敢用来比）。

    口径版本隔离（10-28 加）：
      同一模式下**评分/归因口径也会变**。10-26 修了词边界错位 bug，
      修之前的记录是「错位口径」算的分数，跟修之后的直接相减会得出
      "我进步了"的假结论 —— 其实只是口径变了。
      所以：版本不同 → 跳过；旧记录没有「口径版本」字段 → 也跳过
      （不知道是哪个版本 = 不敢用来比）。

    参数 version 传 None → 只按模式找（用于"能不能比"的诊断，不用于出对比）。
    """
    for entry in reversed(hist):
        if not isinstance(entry, dict):
            continue
        if entry.get("模式") != mode:
            continue
        if version is not None and entry.get("口径版本") != version:
            continue
        return entry
    return None


def count_comparable(hist, mode, version=None):
    """统计历史里有几条能用于对比（同模式 + 同口径版本）。"""
    n = 0
    for entry in hist:
        if not isinstance(entry, dict):
            continue
        if entry.get("模式") != mode:
            continue
        if version is not None and entry.get("口径版本") != version:
            continue
        n += 1
    return n


def append_history(overall, parts, mode):
    """把本次综合分追加进 history.json，供报告显示「上次 vs 这次」的进步。

    mode 必须记（单句/长音频）—— 否则报告会把两种模式混着比。
    口径版本也必须记 —— 否则改过评分/归因口径后，新旧分数会被直接相减，
    得出"我进步了"的假结论（10-28 踩过：修词边界 bug 前后 60.6 vs 60.3）。
    """
    entry = {
        "时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "综合分": round(overall, 1),
        "模式": mode,
        "口径版本": SCORE_VERSION,
        "分项": {k: (round(parts[k], 1) if parts.get(k) is not None else None)
                  for k, _w in DIMENSIONS},
    }
    hist = read_history()
    hist.append(entry)
    try:
        with open(HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log("（写 history.json 失败，不影响评分：%s）" % e, show=True)


def score_pronunciation(stat):
    """发音分：音素序列里"替换"越少越高。
    替换 = 音读错了（θ→s 之类）—— 这是"发音"问题。
    用均值口径：替换数同时占原/模仿音数的比例，取较宽松的一侧，避免长音串放大。"""
    if not stat:
        return None
    ref_n = stat.get("原生音数") or 0
    my_n = stat.get("模仿音数") or 0
    rep = stat.get("替换音数") or 0
    base = max(ref_n, my_n, 1)
    return max(0.0, (1 - rep / base) * 100)


def score_phoneme_change(stat):
    """音变分：缺音（吞音/漏读）+ 多音（连读黏出来的多余音）越少越高。"""
    if not stat:
        return None
    ref_n = stat.get("原生音数") or 0
    my_n = stat.get("模仿音数") or 0
    dele = stat.get("缺失音数") or 0
    ins = stat.get("多余音数") or 0
    base = max(ref_n, my_n, 1)
    return max(0.0, (1 - (dele + ins) / base) * 100)


def grade(score):
    if score is None:
        return "—"
    if score >= 90:
        return "优"
    if score >= 80:
        return "良"
    if score >= 70:
        return "中"
    if score >= 60:
        return "及格"
    return "待加强"


# 各维度低于门槛时的针对性建议（键须与 DIMENSIONS 对齐；新增维度时在这里补文案）
ADVICE = {
    "发音": "发音是当前短板：对照报告里【替换】的音，逐个纠音（注意长短音、θ/ð、r/l）。",
    "音变": "音变（连读/弱化/吞音）偏弱：留意原音频里被“吃掉”或黏连的音，先慢读再连读。",
    "语调": "语调（音高起伏）偏弱：模仿原音频的升降调，疑问句尾音上扬、陈述句尾音下压。",
    "节奏": "节奏偏弱：跟着原音频打拍子，注意停顿位置和每句的重音落点。",
    "重音位置": "重音落点偏弱：原音频强调的词你没强调，留意原音频哪几个词读得重、跟着加重。",
}


# ---- 读 节奏对比.json 里「真实存在」的键 ----
# 旧字段 总时长秒 / 对齐 / 停顿 / 子项分 已失效（现网 JSON 里不存在）。
# 若继续用 .get(..., 0) 兜底，会静默变成 0 —— 所以这里取不到就告警。
def rhythm_section(rhythm, key):
    """取 节奏.<key>；取不到则记日志告警并返回 None。"""
    if not rhythm:
        return None
    sec = (rhythm.get("节奏") or {}).get(key)
    if sec is None:
        log("⚠ 节奏对比.json 里没有 节奏.%s（字段失效或脚本未更新），该项显示为「—」" % key,
            show=True)
    return sec


def collect_parts(phoneme, rhythm):
    """按维度取分：键须与 DIMENSIONS 对齐。加维度时改这里（全局唯一取数处）。"""
    stat = phoneme.get("对齐统计")
    # 长音频模式：只信 compare_long.py 写的 节奏对比.json（数据源带标记），
    # 防止把上一次单句运行的陈旧数据混进长音频报告（实测踩过）。
    long_mode = bool(phoneme.get("长音频"))
    src = ((rhythm or {}).get("节奏") or {}).get("数据源") or ""
    rhythm_ok = bool(rhythm) and (not long_mode or "compare_long" in src)
    return {
        "发音": score_pronunciation(stat),
        "音变": score_phoneme_change(stat),
        "语调": rhythm.get("语调分") if rhythm_ok else None,
        "节奏": rhythm.get("节奏分") if rhythm_ok else None,
        "重音位置": score_stress_position(rhythm) if rhythm_ok else None,
    }


def score_stress_position(rhythm):
    """重音位置分 = (1 − 漏数 / max(原音频突出块数,1)) × 100（计划书 §4 公式）。
    漏 = 原音频突出、模仿没突出的块。⚠️ 只罚「漏」，不罚「多」——
    多突出可能是感情色彩，漏原音频强调的词才是没还原。"""
    if not rhythm:
        return None
    sig = rhythm.get("重音_信号") or {}
    miss = sig.get("漏（原音频有）") or []
    ref = sig.get("原音频突出块") or []
    if not ref:
        return None       # 原音频一个突出块都没判出来 → 没基准，不出分
    return round(max(0.0, (1.0 - len(miss) / float(max(len(ref), 1))) * 100.0), 1)


def build_advice(parts, phoneme, rhythm):
    """根据分项分与明细生成"下一步怎么练"的建议。维度顺序来自 DIMENSIONS。"""
    tips = []
    lows = [k for k, _w in DIMENSIONS if parts.get(k) is not None and parts[k] < 70]
    if not lows:
        tips.append("各维度都在 70 分以上，整体还原度不错，可以练更长、更快的句子。")
    for k, _w in DIMENSIONS:
        v = parts.get(k)
        if v is not None and v < 75 and k in ADVICE:
            tips.append(ADVICE[k])
    # 从停顿明细再补一条针对性提示（真实键：节奏.句子边界停顿）
    ps = rhythm_section(rhythm, "句子边界停顿")
    if ps and ps.get("原音频处数") != ps.get("我的处数"):
        tips.append("停顿数量与原音频不一致（原音频 %s 处 / 你 %s 处），注意不要多停或少停。"
                    % (ps.get("原音频处数"), ps.get("我的处数")))
    return tips


def main():
    log_setup("综合评分")
    out("=" * 62)
    out("   还原度综合评分")
    out("=" * 62)
    out("")

    log_step("读取前两步的产出")
    phoneme = read_json(PHONEME_JSON)
    rhythm = read_json(RHYTHM_JSON)
    if phoneme is None:
        out("X 缺少 音素对比.json —— 请先跑 compare_phoneme.py（或一键 bat）。")
        log_done("（缺前置数据，提前退出）")
        return
    long_mode = bool(phoneme.get("长音频"))
    if rhythm is None:
        if long_mode:
            # 长音频模式但没跑出 节奏对比.json：用空 dict 占位
            # （collect_parts 里 rhythm_ok 会把它判为不可信，相关项不参与加权）
            rhythm = {}
        else:
            out("X 缺少 节奏对比.json —— 请先跑 compare_rhythm.py（或一键 bat）。")
            log_done("（缺前置数据，提前退出）")
            return
    if long_mode:
        log("   长音频模式：发音/音变/节奏/重音 均由 compare_long.py 逐句评测汇总", verbose=True)
    log("   音素对比.json：相似度 %s，差异 %d 条"
        % (phoneme.get("相似度"), len(phoneme.get("差异") or [])), verbose=True)
    log("   节奏对比.json：节奏 %s ｜ 语调 %s ｜ 重音位置 %s"
        % (rhythm.get("节奏分"), rhythm.get("语调分"),
           ((rhythm.get("重音_信号") or {}).get("原音频突出块") and "有" or "—")), verbose=True)

    log_step("计算分项分与综合分")
    stat = phoneme.get("对齐统计")
    parts = collect_parts(phoneme, rhythm)

    # 综合分：只有拿到的项按权重归一（缺项不参与，避免误判为 0）
    avail = {k: parts[k] for k, _w in DIMENSIONS if parts.get(k) is not None}
    wsum = sum(w for k, w in DIMENSIONS if k in avail)
    overall = sum(w * parts[k] for k, w in DIMENSIONS if k in avail) / wsum if wsum else 0.0
    # 有效权重 = 额定权重在「缺项」后归一的结果（与 payload.权重 的额定值口径不同）
    eff_weights = ({k: round(w / wsum, 4) for k, w in DIMENSIONS if k in avail}
                   if wsum else {})

    # ---- 终端简报 ----
    out("综合还原度：%.1f / 100   （等级：%s）" % (overall, grade(overall)))
    out("")
    out("分项：")
    for k, w in DIMENSIONS:
        v = parts.get(k)
        out("   %s  %s  / 100   (权重 %d%%)"
            % (k, ("%.1f" % v) if v is not None else "  —", int(w * 100)))
    out("")
    out("明细参考：")
    if stat:
        out("   音素：替换 %d ｜ 缺失 %d ｜ 多余 %d ｜ 匹配 %d（原生共 %d 音）"
            % (stat.get("替换音数", 0), stat.get("缺失音数", 0), stat.get("多余音数", 0),
               stat.get("匹配音数", 0), stat.get("原生音数", 0)))
    if rhythm and not phoneme.get("长音频"):
        net = rhythm_section(rhythm, "整句净时长") or {}
        ps = rhythm_section(rhythm, "句子边界停顿") or {}
        st = rhythm.get("重音_信号") or {}
        if net:
            out("   节奏：净时长 原 %.2fs / 我 %.2fs（相对差 %+.1f%%）"
                % (net.get("原音频", 0), net.get("我的", 0), (net.get("相对差") or 0) * 100))
        if ps:
            out("   停顿：原音频 %s 处 / 我 %s 处"
                % (ps.get("原音频处数", "—"), ps.get("我的处数", "—")))
        if st:
            out("   重音：突出块 原 %d / 我 %d（命中 %d ｜ 漏 %d ｜ 多 %d）"
                % (len(st.get("原音频突出块") or []), len(st.get("我的突出块") or []),
                   len(st.get("命中") or []), len(st.get("漏（原音频有）") or []),
                   len(st.get("多（我的）") or [])))

    # ---- 写 JSON（给机器）----
    log_step("写综合评分与报告")
    # 注意：这里叫 payload 不叫 out —— out 是 logkit 的结果行函数，
    # 若用同名局部变量会遮蔽它，导致 main() 前面调用 out() 报 UnboundLocalError。
    payload = {
        "综合还原度": round(overall, 1),
        "等级": grade(overall),
        "权重": dict(DIMENSIONS),
        "权重(有效)": eff_weights,
        "分项分": {k: (round(parts[k], 1) if parts.get(k) is not None else None)
                   for k, _w in DIMENSIONS},
        "音素明细": stat,
        "节奏明细": rhythm,
        "音素相似度(原始)": phoneme.get("相似度"),
        "母语者基准": (phoneme.get("长音频") or {}).get("母语者基准"),
    }
    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log("（写 json 失败：%s）" % e, show=True)

    # ---- 记历史（供报告显示进步）----
    append_history(overall, parts, detect_mode(phoneme))

    # ---- 写报告（给人）----
    try:
        write_report(overall, parts, phoneme, rhythm)
    except Exception as e:
        log("（写报告失败：%s）" % e, show=True)

    out("")
    out("数据（机器）：还原度综合评分.json")
    out("报告（人看）：还原度综合报告.txt")
    log_done("综合评分")


def write_report(overall, parts, phoneme, rhythm):
    L = []
    L.append("=" * 60)
    L.append("   还原度综合报告")
    L.append("=" * 60)
    L.append("")
    from audio_paths import find_audio_pair, short_name
    _ref, _my = find_audio_pair()
    L.append("%s  vs  %s（跟读还原度）" % (short_name(_ref), short_name(_my)))
    L.append("")
    # 半截模仿提示：模仿只覆盖原音频一部分时，明确告诉用户本次只按已模仿部分评分
    coverage = (phoneme.get("半截模仿") or {})
    if coverage.get("半截"):
        long_info = phoneme.get("长音频") or {}
        if long_info:
            # 长音频格式：按句报
            L.append("⚠ 检测到「半截模仿」：你只跟读到 %d / %d 句（%.0f%%）。"
                     % (coverage.get("已读句数", 0), coverage.get("总句数", 0),
                        coverage.get("覆盖比例", 0) * 100))
            missed = coverage.get("漏句") or []
            if missed:
                L.append("   未读到的句子（不计分）：")
                for m in missed:
                    L.append("     · 句%s：%s" % (m.get("句号"), m.get("文本", "")))
        else:
            head = coverage.get("头部漏读词") or []
            tail = coverage.get("尾巴漏读词") or []
            L.append("⚠ 检测到「半截模仿」：你只跟读到原音频的 %.1f%%。" % (coverage.get("覆盖比例", 0) * 100))
            if head:
                L.append("   开头没读到的部分：%s" % " ".join(head))
            if tail:
                L.append("   结尾没读到的部分：%s" % " ".join(tail))
        L.append("   本次只按已模仿到的部分评分，未读到的部分不算错。")
        L.append("")
    L.append("综合还原度：%.1f / 100     等级：%s" % (overall, grade(overall)))
    # 母语者基准（10-28 加）：让分数有参照。
    #   没有它，用户只能跟满分比 —— 60 分看着像失败，
    #   但母语者读同一段在同一口径下也只有 88.4，真实差距是 28 分不是 40 分。
    _bl = (phoneme.get("长音频") or {}).get("母语者基准")
    if _bl and _bl.get("相似度") is not None:
        _sim = phoneme.get("相似度")
        if _sim is not None:
            L.append("（母语者基准 %.1f 分 —— 原音频自己跟参考音素比，"
                     "是工具的误差底线；你的真实差距 %.1f 分）"
                     % (_bl["相似度"], max(0.0, _bl["相似度"] - _sim)))
        else:
            L.append("（母语者基准 %.1f 分 —— 原音频自己跟参考音素比，"
                     "是工具的误差底线）" % _bl["相似度"])
    # 进步对比：**只跟同模式 + 同口径版本的历史记录比**。
    #   单句(≤20s) 与长音频(逐句评测) 的素材长度、评分口径都不同，
    #   跨模式比会写出"进步 +4.0"这种假结论（分数变化来自素材变化）。
    #   同理，改过归因口径后（10-26 修词边界错位）新旧分数也不能直接比。
    #   找不到可比的记录就不显示，并说明原因 —— 宁可不给对比，也不给错的对比。
    mode = detect_mode(phoneme)
    L.append("（本次：%s模式 · 口径版本 %d）" % (mode, SCORE_VERSION))
    hist = read_history()
    prev = find_prev_same_mode(hist, mode, SCORE_VERSION)
    if prev and prev.get("综合分") is not None:
        prev_score = prev["综合分"]
        diff = round(overall, 1) - prev_score
        if abs(diff) < 0.05:
            delta = "持平"
        else:
            delta = ("↑ %+.1f" if diff > 0 else "↓ %+.1f") % diff
        L.append("   上次%s模式 %.1f 分（%s）→ 这次 %.1f 分，%s"
                 % (mode, prev_score, prev.get("时间", "?"), overall, delta))
    else:
        same_mode = count_comparable(hist, mode)          # 同模式（不限版本）
        same_ver = count_comparable(hist, mode, SCORE_VERSION)
        if same_mode > 0 and same_ver == 0:
            L.append("   暂无可比的历史记录 —— 已跑过 %d 次%s模式，"
                     "但都是**旧口径版本**（评分/归因方式已改），"
                     "直接比会得出错的结论，所以不比。"
                     % (same_mode, mode))
            L.append("   （下次跑就有基准了）")
        else:
            L.append("   暂无同模式（%s）的历史记录可对比 —— 已跑过 %d 次，"
                     "其中能用于对比的是 %d 次。"
                     % (mode, len(hist), same_ver))
    L.append("")
    L.append("-" * 60)
    L.append("一、分项得分（发音最重，其余持平）")
    L.append("-" * 60)
    for k, w in DIMENSIONS:
        v = parts.get(k)
        bar = "—" if v is None else "%.1f" % v
        L.append("   %s  %6s / 100    权重 %d%%    %s"
                 % (k, bar, int(w * 100), grade(v) if v is not None else ""))
    L.append("")

    # 二、发音 / 音变问题（复用音素对比的问题清单）
    L.append("-" * 60)
    L.append("二、发音 / 音变问题（哪些音不像原音频）")
    L.append("-" * 60)
    diffs = phoneme.get("差异") or []
    # 分类计数（10-28）：三类问题行动不同，先给总览
    _cats = (phoneme.get("长音频") or {}).get("问题分类")
    if _cats:
        L.append("   口音型错误 %d 处 —— 学习者最典型的音变，优先练发音"
                 % _cats.get("口音型", 0))
        L.append("   未读出的音 %d 处 —— 读太轻/漏读，练清晰度"
                 % _cats.get("未读出", 0))
        L.append("   多读出的音 %d 处 —— 连读黏出来的，不扣分"
                 % _cats.get("多读出", 0))
        if _cats.get("其他替换"):
            L.append("   其他替换 %d 处" % _cats.get("其他替换", 0))
        L.append("")
    if diffs:
        # 按类别分组呈现：先给口音型（该练发音），再给其他
        _groups = {"口音": [], "没读出": [], "多读出": [], "替换": []}
        for d in diffs:
            _s = d.strip()
            for _k in _groups:
                if _s.startswith("[%s]" % _k):
                    _groups[_k].append(_s)
                    break
            else:
                _groups["替换"].append(_s)
        for _k, _title in (("口音", "口音型（优先练这些）"),
                           ("没读出", "没读出来的音（练清晰度）"),
                           ("多读出", "多读出的音（不扣分）"),
                           ("替换", "其他替换")):
            if not _groups[_k]:
                continue
            L.append("   【%s】%d 处" % (_title, len(_groups[_k])))
            for d in _groups[_k]:
                L.append("    " + d)
            L.append("")
        L.append("    ※ [口音] 是学习者典型音变（θ→f、ð→d 之类），"
                 "母语者几乎不犯；[没读出] 是读太轻或漏读；")
        L.append("      [多读出] 是连读黏出来的，不扣分；[替换] 是其他读错。")
    else:
        L.append("    （音素序列完全一致，很棒）")
    L.append("")
    # 原句
    words = phoneme.get("词表") or []
    problem_words = set()
    long_problem = phoneme.get("问题词")
    if long_problem:
        # 长音频模式：用 compare_long 挑好的"问题严重词"（错音≥2 或比例≥40%）。
        # 不能改成解析差异行的标签——那样满句都是框，等于没框（实测踩过）。
        problem_words |= set(long_problem)
    else:
        for d in diffs:
            for seg in d.split("【"):
                if "】" not in seg:
                    continue
                inner = seg.split("】")[0]
                # inner 形如：单词 "so"  或  单词 "Where" → "are" 之间
                for q in inner.split('"')[1::2]:        # 取所有被双引号括起的词
                    w = q.strip()
                    if w:
                        problem_words.add(w)
    if words:
        sent = " ".join(("【%s】" % w["word"]) if w["word"] in problem_words else w["word"]
                       for w in words)
        L.append("-" * 60)
        L.append("三、原句（有问题的词已用【】框出）")
        L.append("-" * 60)
        L.append("    " + sent)
        L.append("")

    # 四、节奏 / 语调
    L.append("-" * 60)
    L.append("四、节奏 / 语调")
    L.append("-" * 60)
    long_src = "compare_long" in (((rhythm or {}).get("节奏") or {}).get("数据源") or "")
    if phoneme.get("长音频") and not long_src:
        L.append("    长音频模式但缺 节奏对比.json —— 请先跑 compare_long.py。")
    elif rhythm:
        net = rhythm_section(rhythm, "整句净时长")
        ps = rhythm_section(rhythm, "句子边界停顿")
        st = rhythm.get("重音_信号") or {}
        if net:
            L.append("    净时长：原音频 %.2fs   |   你 %.2fs   |   相对差 %+.1f%%"
                     % (net.get("原音频", 0), net.get("我的", 0), (net.get("相对差") or 0) * 100))
        else:
            L.append("    净时长：（数据缺失）")
        if (rhythm.get("节奏") or {}).get("序号对齐风险"):
            L.append("    ⚠ 两侧词块序号可能未对齐，以下节奏明细仅供参考")
        if ps:
            L.append("    停顿：原音频 %s 处  |  你 %s 处"
                     % (ps.get("原音频处数", "—"), ps.get("我的处数", "—")))
            mine = ps.get("我的") or []
            if mine:
                L.append("          你多停的位置：" + "、".join(
                    "%s 后（%+.2fs）" % (p.get("after_word"), p.get("gap", 0)) for p in mine))
        else:
            L.append("    停顿：（数据缺失）")
        if st:
            L.append("    重音：原音频突出 %d 块  |  你 %d 块  |  命中 %d  |  漏 %d  |  多 %d"
                     % (len(st.get("原音频突出块") or []), len(st.get("我的突出块") or []),
                        len(st.get("命中") or []), len(st.get("漏（原音频有）") or []),
                        len(st.get("多（我的）") or [])))
            miss = st.get("漏（原音频有）") or []
            if miss:
                L.append("          漏掉的重音：" + "、".join(x.get("块", "") for x in miss))
        # 语调：阶段 5 起降级为"仅可视化"，不出分
        L.append("    语调：仅作可视化参考（音高曲线 / 绝对响度见 节奏对比.json），不出分")
    L.append("")

    # 五、改进建议
    L.append("-" * 60)
    L.append("五、下一步怎么练")
    L.append("-" * 60)
    for tip in build_advice(parts, phoneme, rhythm):
        L.append("    · " + tip)
    L.append("")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log_error(traceback.format_exc())
    try:
        input("按回车退出 ...")
    except Exception:
        pass
