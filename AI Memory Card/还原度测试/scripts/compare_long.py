# -*- coding: utf-8 -*-
r"""还原度对比 · 长音频模式：按内容对齐，逐句评测

为什么单独一个脚本（而不是改 compare_phoneme.py）：
  单句链路已经 12 块排期打磨验证过（综合分 72.7 稳定），不能动。
  长音频是**新能力**，单独入口，零回归风险；单句音频走原链路不变。

核心思路（避开 §7.3「两段录音时间轴不匹配」的坑）：
  1. 参考音素序列（文本生成）按标点切成 N 句，每句有确定音素区间 [a,b)
  2. 模仿的识别音素 vs 完整参考序列做一次 difflib 对齐 —— **按内容对齐**，
     自动处理：读快读慢、跳过某句、多读、半截（只读前半）
  3. 全局对齐结果把「模仿音素」分配进各句区间；每句内部再各自对齐
     （对齐范围回到单句级别 → 词块不再崩坏）
  4. 区间内一个模仿音素都没对上的句 = **漏句**（不扣分，单独列出）
  5. 发音/音变按句汇总（求和天然按音素数加权）

产出：
  output\音素对比.json     与单句版同 schema（overall_score.py 直接可用），
                           外加「长音频」段（逐句明细）
  output\音素对比_报告.txt  人读：逐句明细 + 漏句清单 + 半截提示

依赖：与 compare_phoneme.py 相同（torch/transformers/soundfile/numpy）
用法：python scripts\compare_long.py
"""
import datetime
import difflib
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(ROOT, "audio")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

REF_FILE = None      # 运行时按格式探测（见 audio_paths），可能是 .mp3 / .wav / .flac
MY_FILE = None
REF_PHONEME_JSON = os.path.join(SRC_DIR, "参考音素.json")
OUT_JSON = os.path.join(OUT_DIR, "音素对比.json")
OUT_REPORT = os.path.join(OUT_DIR, "音素对比_报告.txt")
RHYTHM_JSON = os.path.join(OUT_DIR, "节奏对比.json")   # 同单句版 schema，overall_score 直接读

sys.path.insert(0, HERE)
from logkit import log, out, log_setup, log_done, log_step, log_error
from audio_paths import find_audio_pair, missing_hint, short_name

REF_FILE, MY_FILE = find_audio_pair(SRC_DIR)

# 复用单句链路的函数（import 不会触发 main，有 __main__ 保护）
import compare_phoneme as cp
import compare_rhythm as cr

# 少于这么多音素的句不判漏句（讯飞转写会把一句话拆成碎句，如 "There. Are"，
# 2 音的 "There" 若判漏句会虚增漏句数 —— 实测踩过）
MIN_SENT_PH = 3


def assign_mine_to_sentences(opcodes, sentences, n_ref, n_mine):
    """把全局对齐的「模仿音素下标」分配到各句。

    opcodes: difflib.get_opcodes() 的结果（a=text_ph 参考序列, b=mine 模仿序列）
    返回：dict 句号 -> [mine 音素下标,...]

    规则：
      - equal/replace：参考区间 [i1,i2) 落在哪句，对应的模仿 [j1,j2) 就归哪句；
        跨句时按重叠比例拆分（极少发生）。
      - insert（模仿多读、参考没有）：归到前一个参考位置所在句（i1-1）。
    """
    # 参考位置 -> 句号
    pos2sent = {}
    for s in sentences:
        for i in range(s["a"], min(s["b"], n_ref)):
            pos2sent[i] = s["句号"]

    assign = {s["句号"]: [] for s in sentences}
    for op, i1, i2, j1, j2 in opcodes:
        if op == "insert":
            # 多读的音：归到前一个参考位置所在句
            prev = i1 - 1
            sent = pos2sent.get(prev)
            if sent is not None:
                assign[sent].extend(range(j1, j2))
            continue
        # equal / replace / delete：按参考区间归属
        if i2 <= i1:
            continue
        # 统计该区间内每个句号的重叠量
        overlap = {}
        for i in range(i1, i2):
            s = pos2sent.get(i)
            if s is not None:
                overlap[s] = overlap.get(s, 0) + 1
        if not overlap:
            continue
        total = sum(overlap.values())
        mine_ids = list(range(j1, j2))
        if op == "delete" or not mine_ids:
            continue        # delete 没有模仿音素可分配
        # 按重叠比例把模仿音素分给各句
        cursor = 0
        for s in sorted(overlap):
            k = round(len(mine_ids) * overlap[s] / total)
            if s == sorted(overlap)[-1]:
                k = len(mine_ids) - cursor      # 最后的句拿剩下的，避免舍入丢失
            assign[s].extend(mine_ids[cursor:cursor + k])
            cursor += k
    return assign


def classify_diffs(scored):
    """把逐句差异按「性质」分类，返回 {类别: [(句号, 差异条目), ...]}。

    分类（10-28 加）：这几类要采取的行动完全不同，混在一起报用户不知道该练什么：
      口音型（L2 音变）→ 练发音：θ→f、ð→d、ɹ→l 这类，母语者几乎不犯
      未读出           → 读太轻/漏读/被吞了，练清晰度
      多读出           → 连读黏出来的，不扣分，注意别抢话
      替换             → 其他读错

    ⚠️ 抽成独立函数有两个原因：
       1) main() 写 payload 和人读报告**都要用** —— 之前只在报告段算，
          main() 用不到 → UnboundLocalError（真实跑一次才发现，自检测不到）。
       2) 抽出来才能被自检覆盖。
    """
    cats = {"口音型": [], "未读出": [], "多读出": [], "替换": []}
    for r in scored or []:
        for d in r.get("差异") or []:
            c = d.get("类别") or ("未读出" if d["类型"] == "缺失"
                                  else "多读出" if d["类型"] == "多余" else "替换")
            cats.setdefault(c, []).append((r["句号"], d))
    return cats


def compute_baseline(text_ph, ref):
    """算「母语者基准」—— 工具自己的误差底线。

    目的：让用户知道「这个工具自己会错多少」。
          否则 60 分看着像失败，其实母语者读同一段也只有这个水平，
          用户会以为差 40 分，实际只差 28 分。

    口径必须**与本次完全一致**（同一套 difflib、同一份参考音素），
    才可比 —— 拿别处的数字来对比是自欺欺人。

    怎么算：把**原音频自己的识别结果**当成"模仿"去跟参考音素比。
          原音频 = 母语者 + 干净录音 = 工具的误差底线。
          它也判错的那些音 → 工具的固有误差（不是任何人的问题）。

    参数
      text_ph : 参考音素序列
      ref     : 原音频的识别音素序列
    返回 dict；输入为空时返回 None。

    ⚠️ 抽成独立函数是为了能自检 —— 放在 main() 里的话，
       变异测试根本测不到「基准算错」这种错（10-28 踩过）。
    """
    if not text_ph or not ref:
        return None
    try:
        sm_b = difflib.SequenceMatcher(a=text_ph, b=ref, autojunk=False)
        b_stat = {"匹配": 0, "替换": 0, "缺失": 0, "多余": 0}
        for op, i1, i2, j1, j2 in sm_b.get_opcodes():
            if op == "equal":
                b_stat["匹配"] += i2 - i1
            elif op == "replace":
                b_stat["替换"] += max(i2 - i1, j2 - j1)
            elif op == "delete":
                b_stat["缺失"] += i2 - i1
            else:
                b_stat["多余"] += j2 - j1
        b_ph = b_stat["匹配"] + b_stat["替换"] + b_stat["缺失"]
        return {
            "口径": "原音频自己跟参考音素比（母语者+干净录音 = 工具误差底线）",
            "相似度": round(sm_b.ratio() * 100, 1),
            "替换": b_stat["替换"], "缺失": b_stat["缺失"], "多余": b_stat["多余"],
            "原生音数": b_ph,
            "替换率": round(100.0 * b_stat["替换"] / max(b_ph, 1), 1),
        }
    except Exception:
        return None


def evaluate_sentence(sent, text_ph, mine, mine_frames, word_bounds):
    """单句评测：句内各自做 difflib 对齐 + 词归属。

    返回 dict：句号/文本/匹配/替换/缺失/多余/漏句/忽略/问题词列表
    “忽略”= 参考音素少于 MIN_SENT_PH 的碎句（转写瑕疵），不判漏句也不计分。
    """
    sa, sb = sent["a"], sent["b"]
    sent_ref = text_ph[sa:sb]
    # 句内词边界（下标平移到句内）
    sent_words = [dict(wb, a=wb["a"] - sa, b=wb["b"] - sa)
                  for wb in word_bounds if wb["a"] >= sa and wb["b"] <= sb]

    # 全局对齐分配给本句的模仿音素（由调用方填进 sent["_mine"]）
    mine_ids = sent["_mine"]
    sent_mine = [mine[j] for j in mine_ids]
    sent_frames = [mine_frames[j] for j in mine_ids]

    # 句内对齐（短序列，对齐质量回到单句级别）
    sm = difflib.SequenceMatcher(a=sent_ref, b=sent_mine, autojunk=False)
    stat = {"匹配": 0, "替换": 0, "缺失": 0, "多余": 0}
    diff_items = []

    def words_overlapping(i1, i2):
        """句内参考区间 [i1,i2) 压到的词（用于差异行挂词标签）。
        契约（见 compare_phoneme.format_words 注释）：overall_score.py 会按
        双引号反向解析这些标签取"问题词"，所以引号必须成对、包单个词。"""
        hit = [wb["word"] for wb in sent_words if wb["a"] < i2 and i1 < wb["b"]]
        if not hit:
            return ""
        if len(hit) == 1:
            return "【单词 \"%s\"】" % hit[0]
        return "【单词 \"%s\" → \"%s\" 之间】" % (hit[0], hit[-1])

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            stat["匹配"] += (i2 - i1)
        elif op == "replace":
            stat["替换"] += max(i2 - i1, j2 - j1)
            _ref = " ".join(sent_ref[i1:i2])
            _got = " ".join(sent_mine[j1:j2])
            # 分类（10-28 加）：口音型 vs 其他替换。
            #   口音型 = 命中 L2 对照表（θ→f、ð→d、ɹ→l 这类）→ 给针对性改法
            #   其他替换 → 就是普通的读错，给通用要领
            #
            # ⚠️ 要在**大块**里逐音配对 —— difflib 会把连续几个音的变化挤成一块
            #   （如「θ æ ŋ」→「f ɛ」其实是 θ→f、æ→ɛ 两个音变），
            #   只做整块一对一判定会漏掉绝大部分（真实跑出来发现：只认出 2 处）。
            _l2 = cp.l2_tip(_ref, _got)          # 整块一对一
            if not _l2:
                _pairs = cp.align_l2_pairs(_ref, _got)   # 块内逐音配对
                if _pairs:
                    _l2 = "；".join(p[2] for p in _pairs)
            diff_items.append({"类型": "替换", "词": words_overlapping(i1, i2),
                               "应读": _ref, "实读": _got,
                               "类别": "口音型" if _l2 else "替换",
                               "要领": _l2 or cp.phoneme_tips(_ref)})
        elif op == "delete":
            stat["缺失"] += (i2 - i1)
            # 缺失 = 参考里有、实际没发出来 → **跟"读错"是两回事**，
            # 要领必须先说"没读出来"，再补通用发音要领。
            # ⚠️ 原来写成 `phoneme_tips(ref) or "没读出来"` —— or 是短路的，
            #    通用要领一旦有值，"没读出来"就被挤掉了，用户还是不知道
            #    该练的是清晰度而不是发音（变异测试发现，已踩）。
            _ref = " ".join(sent_ref[i1:i2])
            _tip = ("【要领】这个音没读出来（可能读太轻、漏读或被吞掉了）")
            _general = cp.phoneme_tips(_ref)
            if _general:
                _tip += "　" + _general
            diff_items.append({"类型": "缺失", "词": words_overlapping(i1, i2),
                               "应读": _ref, "实读": "",
                               "类别": "未读出",
                               "要领": _tip})
        else:
            stat["多余"] += (j2 - j1)
            # 多音没有参考区间 → 挂到前一个音所在的词
            prev = words_overlapping(max(0, i1 - 1), max(1, i1))
            diff_items.append({"类型": "多余", "词": prev,
                               "应读": "", "实读": " ".join(sent_mine[j1:j2]),
                               "类别": "多读出",
                               "要领": ""})

    covered = stat["匹配"] + stat["替换"]
    # 碎句（转写把一句拆成 "There."+"Are..."）：不判漏句、不计分
    ignored = len(sent_ref) < MIN_SENT_PH
    is_missed = (not ignored) and ((len(sent_mine) == 0) or (covered == 0))

    # 词级错误：供"原句框出"用。只挑**问题严重**的词（错音≥2 或 占该词比例≥40%，
    # 阈值同单句版 compute_word_errors）——否则满句都是框，等于没框（实测踩过）。
    word_problems = []
    for wb in sent_words:
        wa, wb_ = wb["a"], wb["b"]
        wrong = 0
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            ov = max(0, min(i2, wb_) - max(i1, wa))
            if op == "insert":
                # 多音归到前一个音所在的词
                if wa <= max(0, i1 - 1) < wb_:
                    wrong += (j2 - j1)
            elif ov > 0:
                wrong += ov
        if wrong == 0:
            continue
        ratio = wrong / float(max(wb_ - wa, 1))
        if wrong >= 2 or ratio >= 0.40:
            word_problems.append({"词": wb["word"], "错音数": wrong,
                                  "错误比例": round(ratio, 3)})

    return {
        "句号": sent["句号"],
        "文本": sent["文本"],
        "角色": sent.get("角色"),
        "音素区间": [sa, sb],
        "参考音素数": len(sent_ref),
        "模仿音素数": len(sent_mine),
        "匹配": stat["匹配"], "替换": stat["替换"],
        "缺失": stat["缺失"], "多余": stat["多余"],
        "漏句": is_missed,
        "忽略": ignored,
        "差异": diff_items,
        "问题词": word_problems,
    }


def build_sentence_blocks(sent, text_ph, ref, ref_frames, mine, mine_frames, word_bounds):
    """给单句构造两侧对称的词块（用于节奏/重音）。

    原音频侧：识别音素按 参考序列↔原音频识别序列 的全局对齐分配进本句区间；
    模仿侧：同上（用 参考序列↔模仿识别序列 的对齐）。
    两侧词块数与词序天然对称（同一批 sent_words）→ 满足 compare_rhythm 的序号对齐前提。

    返回 (blocks_r, blocks_m)，元素 {"词","start_ms","end_ms","连读","角色"}。
    """
    sa, sb = sent["a"], sent["b"]
    sent_ref = text_ph[sa:sb]
    sent_words = [dict(wb, a=wb["a"] - sa, b=wb["b"] - sa)
                  for wb in word_bounds if wb["a"] >= sa and wb["b"] <= sb]
    if not sent_words:
        return [], []

    def side(key, audio_ph, frames):
        ids = sent["_%s" % key]
        ph = [audio_ph[j] for j in ids]
        fr = [frames[j] for j in ids]
        if not ph:
            return []
        owner = cp.map_words_to_audio(sent_ref, ph, sent_words)
        spans = cp.build_text_spans(sent_words, owner, [f * cp.FRAME_MS for f in fr])
        # ⚠️ 词 必须是**列表**（与单句版词块契约一致）：compare_rhythm 里
        # "+".join(b["词"]) 按列表拼接；传字符串会按字符拆成 "W+h+e+r+e"（实测踩过）
        return [{"词": [sp["word"]], "start_ms": sp["start_ms"], "end_ms": sp["end_ms"],
                 "连读": False, "角色": sent.get("角色")} for sp in spans]

    return side("ref", ref, ref_frames), side("mine", mine, mine_frames)


def main():
    log_setup("长音频评测")
    out("=" * 62)
    out("   还原度对比 · 长音频（按内容对齐·逐句评测）")
    out("=" * 62)

    if not (REF_FILE and MY_FILE):
        out("X 没找到音频文件：")
        out(missing_hint([("原音频", REF_FILE), ("模仿", MY_FILE)], SRC_DIR))
        log_done("（缺音频，提前退出）")
        return

    log_step("加载音素识别模型")
    model_path = cp.ensure_model()
    fe = cp.load_feature_extractor(model_path)
    model = cp.AutoModelForCTC.from_pretrained(model_path) if hasattr(cp, "AutoModelForCTC") else None
    if model is None:
        from transformers import AutoModelForCTC
        model = AutoModelForCTC.from_pretrained(model_path)
    model.eval()
    id2tok = cp.load_vocab(model_path)

    log_step("识别两段音频")
    import torch
    results, frames_map = {}, {}
    for tag, path in (("原音频", REF_FILE), ("我的模仿", MY_FILE)):
        audio = cp.load_mono(path)
        inputs = fe(audio, sampling_rate=cp.SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)[0].tolist()
        toks, frames = cp.ctc_decode(ids, id2tok)
        results[tag] = toks
        frames_map[tag] = frames
        log("   %s：%d 音素（%.1f 秒）" % (tag, len(toks), len(audio) / cp.SR))

    ref = results["原音频"]
    mine = results["我的模仿"]

    log_step("读参考音素表（含句子边界）")
    if not os.path.exists(REF_PHONEME_JSON):
        out("X 缺少 参考音素.json —— 请先跑 gen_ref_phonemes.py 生成")
        log_done("（缺参考音素表，提前退出）")
        return
    with open(REF_PHONEME_JSON, encoding="utf-8") as f:
        ref_data = json.load(f)
    text_ph = ref_data.get("音素") or []
    word_bounds = ref_data.get("词边界") or []
    sentences = ref_data.get("句子边界") or []
    if not sentences:
        out("X 参考音素.json 里没有 句子边界 —— 请用新版 gen_ref_phonemes.py 重新生成")
        log_done("（缺句子边界，提前退出）")
        return
    log("   参考 %d 音素 / %d 词 / %d 句" % (len(text_ph), len(word_bounds), len(sentences)))

    # ---- 全局对齐（按内容，不用时间戳）----
    log_step("全局对齐（按内容）")
    sm = difflib.SequenceMatcher(a=text_ph, b=mine, autojunk=False)
    opcodes = sm.get_opcodes()
    cov = sum(i2 - i1 for op, i1, i2, j1, j2 in opcodes if op == "equal") / max(1, len(text_ph))
    log("   音素级覆盖率 %.1f%%" % (cov * 100))

    assign = assign_mine_to_sentences(opcodes, sentences, len(text_ph), len(mine))
    for s in sentences:
        s["_mine"] = assign.get(s["句号"], [])
    # 原音频侧识别音素也要分配进各句（节奏/重音需要两侧对称词块）
    sm_ref = difflib.SequenceMatcher(a=text_ph, b=ref, autojunk=False)
    assign_ref = assign_mine_to_sentences(sm_ref.get_opcodes(), sentences,
                                          len(text_ph), len(ref))
    for s in sentences:
        s["_ref"] = assign_ref.get(s["句号"], [])

    # ---- 逐句评测 ----
    log_step("逐句评测")
    sent_results = [evaluate_sentence(s, text_ph, mine, frames_map["我的模仿"], word_bounds)
                    for s in sentences]
    scored = [r for r in sent_results if not r["漏句"] and not r["忽略"]]
    missed = [r for r in sent_results if r["漏句"]]
    ignored = [r for r in sent_results if r["忽略"]]
    log("   %d 句中：已模仿 %d 句，漏读 %d 句，碎句忽略 %d 句"
        % (len(sentences), len(scored), len(missed), len(ignored)))

    # ---- 半截检测（句子级）----
    tail_missed = 0
    for r in reversed(sent_results):
        if r["漏句"]:
            tail_missed += 1
        else:
            break
    coverage_ratio = (len(sent_results) - tail_missed) / float(len(sent_results))
    is_partial = tail_missed >= 2 and coverage_ratio < 0.95
    if is_partial:
        log("   ⚠ 半截模仿：只读到第 %d/%d 句，后 %d 句整句未读"
            % (len(sent_results) - tail_missed, len(sent_results), tail_missed), show=True)

    # ---- 汇总（按句求和，天然按音素数加权）----
    total = {"匹配": 0, "替换": 0, "缺失": 0, "多余": 0}
    for r in scored:                   # 漏句/碎句不计入（没读到的不算错）
        for k in total:
            total[k] += r[k]
    total_ph = total["匹配"] + total["替换"] + total["缺失"]
    pron = max(0.0, (1 - total["替换"] / max(total_ph, 1))) * 100
    change = max(0.0, (1 - (total["缺失"] + total["多余"]) / max(total_ph, 1))) * 100
    sim = sm.ratio() * 100

    # ---- 母语者基准（10-28 加）----
    baseline = compute_baseline(text_ph, ref)
    if baseline:
        log("   母语者基准：相似度 %.1f ｜ 替换率 %.1f%%（工具固有误差）"
            % (baseline["相似度"], baseline["替换率"]), verbose=True)

    # ---- 节奏 / 重音（逐句两侧对称词块，整段汇总）----
    log_step("节奏 / 重音（逐句对称词块）")
    blocks_r_all, blocks_m_all, pairs = [], [], []
    env_r = cr.frame_rms(cp.load_mono(REF_FILE))
    env_m = cr.frame_rms(cp.load_mono(MY_FILE))
    f0_r = cr.pitch_contour(cp.load_mono(REF_FILE))
    f0_m = cr.pitch_contour(cp.load_mono(MY_FILE))
    for s in sentences:
        r = next(x for x in sent_results if x["句号"] == s["句号"])
        if r["漏句"] or r["忽略"]:
            continue
        br, bm = build_sentence_blocks(s, text_ph, ref, frames_map["原音频"],
                                       mine, frames_map["我的模仿"], word_bounds)
        if not br or not bm:
            continue
        blocks_r_all.extend(br)
        blocks_m_all.extend(bm)
        for i in range(min(len(br), len(bm))):
            pairs.append({"序号": len(pairs) + 1,
                          "块": "+".join(br[i]["词"]),
                          "连读": br[i].get("连读", False),
                          "角色": br[i].get("角色"),
                          "原音频时长": cr.block_span(br[i]),
                          "我的时长": cr.block_span(bm[i])})
    # 节奏分：把所有已读句的词块对合成一次 score_blocks（归一化+幂平均，同单句口径）
    items, rhythm_score = cr.score_blocks(pairs) if pairs else ([], 0.0)
    # 重音：两侧各自邻域法（整段已读部分当一个整体，同单句口径），再比对集合
    feats_r = cr.block_signal_features(env_r, f0_r, blocks_r_all)
    feats_m = cr.block_signal_features(env_m, f0_m, blocks_m_all)
    for i, b in enumerate(blocks_r_all):
        feats_r[i]["词"] = b["词"]
    for i, b in enumerate(blocks_m_all):
        feats_m[i]["词"] = b["词"]
    roles_r = [b.get("角色") for b in blocks_r_all]
    roles_m = [b.get("角色") for b in blocks_m_all]
    marked_r, detail_r = (cr.mark_stress_neighborhood(feats_r, blocks_r_all, cr.NEIGH_R,
                                                      cr.NEIGH_K, roles_r)
                          if blocks_r_all else ([], []))
    marked_m, detail_m = (cr.mark_stress_neighborhood(feats_m, blocks_m_all, cr.NEIGH_R,
                                                      cr.NEIGH_K, roles_m)
                          if blocks_m_all else ([], []))
    miss_s = sorted(set(marked_r) - set(marked_m))
    extra_s = sorted(set(marked_m) - set(marked_r))
    hit_s = sorted(set(marked_r) & set(marked_m))
    ref_prom = len(marked_r)
    stress_pos = (round(max(0.0, (1.0 - len(miss_s) / float(max(ref_prom, 1))) * 100.0), 1)
                  if ref_prom else None)
    log("   节奏分 %.1f ｜ 重音位置 %s（原突出 %d / 我突出 %d / 漏 %d / 多 %d）"
        % (rhythm_score, stress_pos, ref_prom, len(marked_m), len(miss_s), len(extra_s)))

    # ---- 画图（复用 compare_rhythm 的绘图函数；不画就会留下上一次的旧图）----
    log_step("画图")
    try:
        cr.plot_blocks(items, title="节奏对比（长音频·按词块序号，仅已读句）")
    except Exception as e:
        log("（画节奏图失败，不影响数据：%s）" % e, show=True)
    try:
        cr.plot_pitch(cr.pitch_to_semitone(f0_r), cr.pitch_to_semitone(f0_m))
    except Exception as e:
        log("（画音高图失败，不影响数据：%s）" % e, show=True)

    out("")
    out("=" * 62)
    out("   长音频评测结果")
    out("=" * 62)
    out("整段音素相似度：%.1f / 100" % sim)
    if baseline:
        out("母语者基准：%.1f / 100（原音频自己跟参考音素比 = 工具误差底线）"
            % baseline["相似度"])
        out("  → 你的真实差距是 %.1f 分，不是 %.1f 分"
            % (max(0.0, baseline["相似度"] - sim), max(0.0, 100 - sim)))
    out("已模仿 %d / %d 句（漏读 %d 句，碎句忽略 %d 句）"
        % (len(scored), len(sent_results), len(missed), len(ignored)))
    if is_partial:
        out("⚠ 半截模仿：只读到 %.0f%%，后 %d 句未读（不计分）" % (coverage_ratio * 100, tail_missed))
    out("发音分 %.1f ｜ 音变分 %.1f ｜ 节奏分 %.1f ｜ 重音位置 %s（仅按已模仿句计算）"
        % (pron, change, rhythm_score,
           ("%.1f" % stress_pos) if stress_pos is not None else "—"))
    out("")
    for r in sent_results:
        tag = " [漏句-未读]" if r["漏句"] else (" [碎句-忽略]" if r["忽略"] else "")
        out("  句%-2d 参考%3d音/模仿%3d音 匹配%3d 替换%2d 缺失%2d 多余%2d  %s%s"
            % (r["句号"], r["参考音素数"], r["模仿音素数"], r["匹配"], r["替换"],
               r["缺失"], r["多余"], r["文本"][:36], tag))

    out("")
    out("=" * 62)
    out("   重音：先看两侧每块的判定依据，再看纠错结论")
    out("=" * 62)
    _nr = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_r_all)}
    _nm = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_m_all)}
    if detail_r or detail_m:
        for line in cr.render_stress_detail(detail_r, "原音频", miss=miss_s):
            out(line)
        out("")
        for line in cr.render_stress_detail(detail_m, "我的模仿", extra=extra_s):
            out(line)
        out("")
        out("纠错结论：")
        for line in cr.render_stress_verdict(_nr, _nm, marked_r, marked_m):
            out(line)
        out("")
        out("重点词清单（照这个单子练，不用自己从上面表格里找）：")
        for line in cr.render_focus_words(detail_r, detail_m, _nr, _nm):
            out(line)
        for line in cr.render_extra_words(detail_r, detail_m, _nr, _nm):
            out(line)
    else:
        out("   （无已读句，无重音判定）")

    # ---- 写 JSON（兼容单句版 schema，overall_score.py 可直接用）----
    log_step("写结果")
    align_stat = {"原生音数": total_ph + total["多余"],
                  "模仿音数": sum(r["模仿音素数"] for r in scored),
                  "匹配音数": total["匹配"], "替换音数": total["替换"],
                  "缺失音数": total["缺失"], "多余音数": total["多余"]}
    diff_lines = []
    for r in scored:
        for d in r["差异"]:
            # 类型标记带分类（10-28）：口音型 / 未读出 / 多读出 / 替换
            #   综合报告按这个标记分组呈现 —— 原来混在一起，用户不知道
            #   "口音型该练发音"和"没读出该练清晰度"是两回事。
            _cat = d.get("类别") or ("未读出" if d["类型"] == "缺失"
                                    else "多读出" if d["类型"] == "多余" else "替换")
            _mark = {"口音型": "口音", "未读出": "没读出",
                     "多读出": "多读出", "替换": "替换"}.get(_cat, "替换")
            if _cat == "未读出":
                line = "  [%s] 句%d %s%s：没读出来" % (_mark, r["句号"], d.get("词") or "",
                                                    ("应读 %s" % d["应读"]) if d["应读"] else "")
            else:
                line = "  [%s] 句%d %s%s：%s" % (_mark, r["句号"], d.get("词") or "",
                                             ("应读 %s" % d["应读"]) if d["应读"] else "",
                                             ("实读 %s" % d["实读"]) if d["实读"] else "")
            if d.get("要领"):
                line += "   " + d["要领"]
            diff_lines.append(line)
    word_list = [{"word": wb["word"], "角色": wb.get("角色"), "句号": wb.get("句号")}
                 for wb in word_bounds]
    # 问题严重的词（去重，保序）——overall_score 优先用它框原句
    problem_words = []
    seen = set()
    for r in scored:
        for wp in r.get("问题词") or []:
            if wp["词"] not in seen:
                seen.add(wp["词"])
                problem_words.append(wp["词"])

    # 问题分类（10-28 加）：这几类要采取的行动完全不同，混在一起报，
    # 用户不知道该练发音还是该练清晰度。
    #   口音型（L2 音变）→ 练发音：θ→f、ð→d、ɹ→l 这类，母语者几乎不犯
    #   未读出           → 读太轻/漏读/被吞了，练清晰度
    #   多读出           → 连读黏出来的，不扣分，注意别抢话
    #   替换             → 其他读错
    # ⚠️ 必须在写 payload **之前**算 —— payload 和人读报告都要用。
    #    之前只在报告那段算，main() 里的 payload 用不到 → UnboundLocalError
    #    （真实跑一次才发现；自检测不到，因为自检不跑 main）。
    _cats = classify_diffs(scored)

    payload = {
        "原音频": ref, "我的模仿": mine,
        "相似度": round(sim, 1), "对齐统计": align_stat,
        "差异": diff_lines,
        "词表": word_list,
        "问题词": problem_words,
        "词块": {"原音频": blocks_r_all, "我的": blocks_m_all},
        "半截模仿": {"半截": is_partial,
                     "覆盖比例": round(coverage_ratio, 4),
                     "已读句数": len(scored), "总句数": len(sent_results),
                     "尾部漏读句数": tail_missed,
                     "漏句": [{"句号": r["句号"], "文本": r["文本"]} for r in missed],
                     "说明": "句子级：区间内无模仿音素对齐上的句判为漏句，不计分；"
                             "少于 %d 音的碎句忽略（转写瑕疵）" % MIN_SENT_PH},
        "长音频": {"模式": "按内容对齐·逐句评测",
                   "句子": sent_results,
                   "发音分": round(pron, 1), "音变分": round(change, 1),
                   "节奏分": round(rhythm_score, 1),
                   "重音位置分": stress_pos,
                   "全局音素覆盖率": round(cov, 4),
                   # 问题分类计数（10-28）：三类问题要采取的行动不同，
                   # 混在一起报，用户不知道该练发音还是该练清晰度。
                   "问题分类": {"口音型": len(_cats.get("口音型", [])),
                                "未读出": len(_cats.get("未读出", [])),
                                "多读出": len(_cats.get("多读出", [])),
                                "其他替换": len(_cats.get("替换", []))},
                   # 母语者基准：同口径下「工具自己会错多少」。
                   # 有了它，报告里的分数才有参照 —— 否则用户只能跟满分比，
                   # 以为差 40 分，实际母语者读同一段也差这个数。
                   "母语者基准": baseline},
        "参考音素可用": True,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- 写 节奏对比.json（schema 同单句版，overall_score.py 直接可用）----
    net_r = ((blocks_r_all[-1]["end_ms"] - blocks_r_all[0]["start_ms"]) / 1000.0
             if blocks_r_all else 0.0)
    net_m = ((blocks_m_all[-1]["end_ms"] - blocks_m_all[0]["start_ms"]) / 1000.0
             if blocks_m_all else 0.0)
    # 停顿：只在句内统计（句间停顿两侧边界不一致，跨句比较无意义）
    pauses_r_all = cr.find_boundary_pauses(blocks_r_all)[0] if blocks_r_all else []
    pauses_m_all = cr.find_boundary_pauses(blocks_m_all)[0] if blocks_m_all else []
    nr = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_r_all)}
    nm = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_m_all)}
    rhythm_payload = {
        "节奏": {
            "数据源": "compare_long.py 逐句对称词块（长音频模式）",
            "整句净时长": {"原音频": round(net_r, 3), "我的": round(net_m, 3),
                           "相对差": round((net_m - net_r) / net_r, 4) if net_r > 0 else 0},
            "词块数": {"原音频": len(blocks_r_all), "我的": len(blocks_m_all)},
            "序号对齐风险": (len(blocks_r_all) != len(blocks_m_all)),
            "句子边界停顿": {
                "原音频处数": len(pauses_r_all), "我的处数": len(pauses_m_all),
                "原音频": pauses_r_all, "我的": pauses_m_all,
                "换人处_预期停顿": {"原音频": [], "我的": []},
                "说明": "长音频模式：只统计句内停顿（句间停顿两侧边界不一致，不跨句比）",
            },
            "词块时长": items,
        },
        "重音_信号": {
            "方法": "邻域法（三路投票：音高半音/响度÷中位/时长÷中位；局部最大且落差≥K·σ）",
            "分组": "角色",
            "半径R": cr.NEIGH_R, "阈值K": cr.NEIGH_K,
            "原音频突出块": [{"序号": i, "块": nr.get(i, "?")} for i in marked_r],
            "我的突出块": [{"序号": i, "块": nm.get(i, "?")} for i in marked_m],
            "漏（原音频有）": [{"序号": i, "块": nr.get(i, "?")} for i in miss_s],
            "多（我的）": [{"序号": i, "块": nm.get(i, "?")} for i in extra_s],
            "命中": [{"序号": i, "块": nr.get(i, "?")} for i in hit_s],
            "逐块明细": {"原音频": detail_r, "我的": detail_m},
            "说明": "长音频模式：已读部分当一个整体做邻域统计（同单句口径）",
        },
        "词表": {"原音频": [{"词": b["词"], "start_ms": b["start_ms"], "end_ms": b["end_ms"],
                             "连读": b.get("连读", False), "角色": b.get("角色")}
                            for b in blocks_r_all],
                 "我的": [{"词": b["词"], "start_ms": b["start_ms"], "end_ms": b["end_ms"],
                           "连读": b.get("连读", False), "角色": b.get("角色")}
                          for b in blocks_m_all]},
        "节奏分": round(rhythm_score, 1),
        "语调分": None,
    }
    with open(RHYTHM_JSON, "w", encoding="utf-8") as f:
        json.dump(rhythm_payload, f, ensure_ascii=False, indent=2)

    # ---- 人读报告 ----
    L = []
    L.append("=" * 60)
    L.append("   音素对比报告（长音频·按内容对齐·逐句评测）")
    L.append("=" * 60)
    L.append("")
    L.append("%s  vs  %s" % (short_name(REF_FILE), short_name(MY_FILE)))
    L.append("整段音素相似度：%.1f / 100" % sim)
    L.append("已模仿 %d / %d 句（漏读 %d，碎句忽略 %d）"
             % (len(scored), len(sent_results), len(missed), len(ignored)))
    if is_partial:
        L.append("⚠ 半截模仿：只读到 %.0f%%（后 %d 句未读，不计分）"
                 % (coverage_ratio * 100, tail_missed))
    L.append("")
    L.append("-" * 60)
    L.append("一、问题分类总览（先看哪类最多，再看明细）")
    L.append("-" * 60)
    # _cats 在上面（写 payload 之前）已算好，这里直接用
    L.append("   口音型错误 %d 处（学习者最典型的音变，练发音）"
             % len(_cats.get("口音型", [])))
    L.append("   未读出的音 %d 处（读太轻/漏读，练清晰度）"
             % len(_cats.get("未读出", [])))
    L.append("   多读出的音 %d 处（连读黏出来的，不扣分）"
             % len(_cats.get("多读出", [])))
    L.append("   其他替换   %d 处" % len(_cats.get("替换", [])))
    if baseline:
        L.append("")
        L.append("   母语者基准：同一口径下，原音频自己跟参考音素比是 %.1f 分"
                 % baseline["相似度"])
        L.append("   → 你的 %.1f 分里，约 %.1f 分是工具固有误差，"
                 "真实差距 %.1f 分"
                 % (sim, max(0.0, 100 - baseline["相似度"]),
                    max(0.0, baseline["相似度"] - sim)))
    L.append("")
    L.append("-" * 60)
    L.append("二、逐句明细")
    L.append("-" * 60)
    for r in sent_results:
        L.append("  句%-2d %s" % (r["句号"], r["文本"]))
        if r["漏句"]:
            L.append("      [漏句] 这句整句未读，不计分")
        elif r["忽略"]:
            L.append("      [碎句] 参考音素少于 %d 个（转写瑕疵），忽略" % MIN_SENT_PH)
        else:
            L.append("      参考 %d 音 / 模仿 %d 音 ｜ 匹配 %d 替换 %d 缺失 %d 多余 %d"
                     % (r["参考音素数"], r["模仿音素数"], r["匹配"], r["替换"],
                        r["缺失"], r["多余"]))
            for d in r["差异"][:6]:
                label = d.get("词") or ""
                tip = ("   " + d["要领"]) if d.get("要领") else ""
                cat = d.get("类别") or ""
                if d["类型"] == "替换":
                    mark = "口音" if cat == "口音型" else "替换"
                    L.append("      [%s] %s应读 %s → 实读 %s%s"
                             % (mark, label, d["应读"], d["实读"], tip))
                elif d["类型"] == "缺失":
                    # 明确说「没读出来」——原来写"应读 X"会让人误以为要读 X
                    L.append("      [没读出] %s这个音没出来（%s）%s"
                             % (label, d["应读"], tip))
                else:
                    L.append("      [多读出] %s多了 %s" % (label, d["实读"]))
    # 口音型专段：把该练的音变集中列出来，一眼能抄下来练
    if _cats.get("口音型"):
        L.append("")
        L.append("-" * 60)
        L.append("三、口音型错误专段（这些是中国学习者最典型的音变，优先练）")
        L.append("-" * 60)
        for sid, d in _cats["口音型"][:30]:
            L.append("   句%-2d %s应读 %s → 实读 %s"
                     % (sid, d.get("词") or "", d["应读"], d["实读"]))
            if d.get("要领"):
                L.append("        %s" % d["要领"])
        if len(_cats["口音型"]) > 30:
            L.append("   （还有 %d 处，见上方逐句明细）"
                     % (len(_cats["口音型"]) - 30))
    if missed:
        L.append("")
        L.append("-" * 60)
        L.append("四、漏句清单（未读到的句子，不扣分）")
        L.append("-" * 60)
        for r in missed:
            L.append("   句%d：%s" % (r["句号"], r["文本"]))
    L.append("")
    L.append("-" * 60)
    L.append("五、重音：先看两侧每块的判定依据，再看纠错结论")
    L.append("-" * 60)
    L.append("")
    _nr = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_r_all)}
    _nm = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_m_all)}
    L.append("判定口径：三路（音高半音 / 响度÷中位 / 时长÷中位）各自按角色组归一，")
    L.append("        每块与【同角色前后邻块】中位数比，落差 ≥ %.1fσ 且为局部极值 → 该路投突出；"
             % cr.NEIGH_K)
    L.append("        至少 2 路投出才算这块突出（够 2 路即取半数向上取整）。")
    L.append("        ⚠ 音高路【双向】（升调/降调都算 prominence，因为英语重音可由降调实现）；")
    L.append("          响度/时长路【单向】（只有「更响/更长」算突出，轻/短是弱化不是重读）。")
    L.append("")
    if detail_r:
        L.extend(cr.render_stress_detail(detail_r, "原音频", miss=miss_s))
        L.append("")
    if detail_m:
        L.extend(cr.render_stress_detail(detail_m, "我的模仿", extra=extra_s))
        L.append("")
        L.append("纠错结论：")
        L.extend(cr.render_stress_verdict(_nr, _nm, marked_r, marked_m))
        L.append("")
        L.append("重点词清单（照这个单子练，不用自己从上面表格里找）：")
        L.extend(cr.render_focus_words(detail_r, detail_m, _nr, _nm))
        L.extend(cr.render_extra_words(detail_r, detail_m, _nr, _nm))
    else:
        L.append("   （无已读句，无重音判定）")
    L.append("")
    L.append("-" * 60)
    L.append("四、汇总（仅按已模仿句）")
    L.append("-" * 60)
    L.append("   发音分 %.1f / 100 ｜ 音变分 %.1f / 100" % (pron, change))
    L.append("   节奏分 %.1f / 100 ｜ 重音位置 %s / 100"
             % (rhythm_score, ("%.1f" % stress_pos) if stress_pos is not None else "—"))
    L.append("   净时长：原音频 %.2fs ｜ 你 %.2fs ｜ 停顿 原 %d 处 / 你 %d 处"
             % (net_r, net_m, len(pauses_r_all), len(pauses_m_all)))
    L.append("   重音：原音频突出 %d 块 ｜ 你 %d 块 ｜ 命中 %d ｜ 漏 %d ｜ 多 %d"
             % (len(marked_r), len(marked_m), len(hit_s), len(miss_s), len(extra_s)))
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    out("")
    out("报告（人看）：音素对比_报告.txt")
    out("数据（机器）：音素对比.json")
    log_done("长音频评测")


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
