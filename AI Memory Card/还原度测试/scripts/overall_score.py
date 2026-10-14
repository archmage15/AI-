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


def append_history(overall, parts):
    """把本次综合分追加进 history.json，供报告显示「上次 vs 这次」的进步。"""
    entry = {
        "时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "综合分": round(overall, 1),
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
    return {
        "发音": score_pronunciation(stat),
        "音变": score_phoneme_change(stat),
        "语调": rhythm.get("语调分"),
        "节奏": rhythm.get("节奏分"),
        "重音位置": score_stress_position(rhythm),
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
    if rhythm is None:
        out("X 缺少 节奏对比.json —— 请先跑 compare_rhythm.py（或一键 bat）。")
        log_done("（缺前置数据，提前退出）")
        return
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
    if rhythm:
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
    }
    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log("（写 json 失败：%s）" % e, show=True)

    # ---- 记历史（供报告显示进步）----
    append_history(overall, parts)

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
    L.append("原音频.wav  vs  模仿.wav（跟读还原度）")
    L.append("")
    L.append("综合还原度：%.1f / 100     等级：%s" % (overall, grade(overall)))
    hist = read_history()
    if len(hist) >= 2:
        prev = hist[-2]
        prev_score = prev.get("综合分")
        if prev_score is not None:
            diff = round(overall, 1) - prev_score
            if abs(diff) < 0.05:
                delta = "持平"
            else:
                delta = ("↑ %+.1f" if diff > 0 else "↓ %+.1f") % diff
            L.append("   上次 %.1f 分（%s）→ 这次 %.1f 分，%s"
                     % (prev_score, prev.get("时间", "?"), overall, delta))
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
    if diffs:
        for d in diffs:
            L.append("    " + d.strip())
        L.append("")
        L.append("    ※ [替换] 属发音问题；[缺音]/[多音] 属音变（连读/弱化/吞音）")
    else:
        L.append("    （音素序列完全一致，很棒）")
    L.append("")
    # 原句
    words = phoneme.get("词表") or []
    problem_words = set()
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
    if rhythm:
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
