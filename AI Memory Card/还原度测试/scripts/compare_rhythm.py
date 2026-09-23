# -*- coding: utf-8 -*-
r"""还原度对比 · 第一块：节奏 / 重音

数据来源分两条线，各管各的：

  节奏（讯飞词级时间戳）
    - 词块配对：按【词序号】一一对应，绝不拿绝对秒数相减（两段录音时间轴不匹配）
    - 词块时长：讯飞给初值，用【能量实证】复核；差得离谱时以实证为准
    - 停顿    ：只在【句子边界】判（词与词之间正常就是连着走的，不算停顿）
    - 整句净时长：首词 start -> 末词 end

  重音（音频信号，各侧自比）
    - 每块算 响度峰值 / 音高均值，各自与【本侧】全句均值比（不跨侧比绝对值，
      因为两段录音的电平本来就不一样）
    - 超过 本侧均值 × STRESS_X 即判为突出；响度或音高任一过线就算
    - 两侧的"突出块序号集合"对比 -> 漏 / 多 / 命中
    - 只报差异，不下"你错了"的结论（对错取决于说话意图，机器判不了）

用法：
  1. 把两个音频放进 audio\ 文件夹，命名成：原音频.wav 和 模仿.wav
  2. 双击 scripts\run_rhythm.bat（或根目录的 一键跑还原度.bat）
  3. 屏幕出报告，结果写到 output\ 文件夹

目录约定（本脚本在 scripts\ 里，用 ROOT 指回上一层）：
  audio\   放录音（你手动放，只读）
  output\  放本次结果（自动生成，可随时清空）

依赖：soundfile / numpy / matplotlib / requests

注：语调（F0 音高走向）本次暂不出分，等判分标准重做后再接回来。
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
from urllib.parse import urlencode

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 目录约定 ----
# 本脚本位于 <项目>\scripts\ 下，ROOT = 项目根（上一层）
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(ROOT, "audio")
OUT_DIR = os.path.join(ROOT, "output")
for _d in (SRC_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

REF_FILE = os.path.join(SRC_DIR, "原音频.wav")
MY_FILE = os.path.join(SRC_DIR, "模仿.wav")
OUT_PNG = os.path.join(OUT_DIR, "节奏对比.png")
OUT_JSON = os.path.join(OUT_DIR, "节奏对比.json")   # 给机器/综合评分器用
PHONEME_JSON = os.path.join(OUT_DIR, "音素对比.json")  # 读连读区（本脚本不写它）

XF_CONFIG = os.path.join(ROOT, "讯飞配置.json")
XF_HOST = "https://raasr.xfyun.cn/v2/api"

SR = 16000
FRAME_MS = 25
HOP_MS = 10
VOICED_RATIO = 0.12      # 能量低于 全句 max*该比例 算静音（判"实证有声时长"用）
FIX_TOLERANCE_S = 0.15   # 讯飞时长与实证有声差超过这个 -> 以实证为准

# ---- 节奏参数 ----
BOUNDARY_GAP_S = 0.12    # 词间空隙 >= 这个 且 落在句子边界 -> 算停顿
REL_FAST = 0.30          # 块时长相对差 >= +30% -> 要读快一些
REL_SLOW = -0.30         # 块时长相对差 <= -30% -> 要读慢一些

# ---- 重音参数 ----
STRESS_X = 1.15          # 超过 本侧均值 × X 即判突出（各侧自比）
DETREND = True           # 先去线性趋势（治"句首响句尾弱"的音量下倾），再判突出


# ---- 日志（统一模块，带时间戳；详细模式见 logkit.py 顶部说明）----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logkit import log, out, log_setup, log_done, log_step, log_error, VERBOSE


# ---------------- 读音频 + 变单声道 + 重采样 ----------------
def load_mono(path):
    if not os.path.exists(path):
        return None
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(audio) * SR / sr))
        x_in = np.linspace(0, 1, len(audio), endpoint=False)
        x_out = np.linspace(0, 1, n_out, endpoint=False)
        audio = np.interp(x_out, x_in, audio)
    return audio.astype(np.float32)


# ---------------- 能量包络 ----------------
def frame_rms(y):
    fl = int(SR * FRAME_MS / 1000)
    hl = int(SR * HOP_MS / 1000)
    if len(y) < fl:
        return np.zeros(1)
    n = 1 + (len(y) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    return np.sqrt((y[idx] ** 2).mean(axis=1))


# ---------------- 音高 F0（自相关，粗略） ----------------
def pitch_contour(y, fmin=70, fmax=400):
    fl = int(SR * 0.04)
    hl = int(SR * HOP_MS / 1000)
    lag_min = int(SR / fmax)
    lag_max = int(SR / fmin)
    f0 = []
    for i in range(0, max(1, len(y) - fl), hl):
        fr = y[i:i + fl]
        if len(fr) < fl:
            break
        fr = fr - fr.mean()
        corr = np.correlate(fr, fr, mode="full")[len(fr) - 1:]
        if corr[0] <= 0 or lag_max >= len(corr):
            f0.append(0.0)
            continue
        seg = corr[lag_min:lag_max]
        if len(seg) == 0:
            f0.append(0.0)
            continue
        peak = int(np.argmax(seg)) + lag_min
        if corr[peak] / corr[0] < 0.3:
            f0.append(0.0)
        else:
            f0.append(SR / peak)
    return np.array(f0)


# ================= 讯飞转写（拿词级时间戳） =================
def xf_make_signa(appid, secret_key, ts):
    """讯飞 LFASR v2 签名：base64(HMAC-SHA1(secret_key, md5(appid+ts)))"""
    md5_hex = hashlib.md5((appid + ts).encode("utf-8")).hexdigest()
    raw = hmac.new(secret_key.encode("utf-8"), md5_hex.encode("utf-8"),
                   hashlib.sha1).digest()
    return base64.b64encode(raw).decode("utf-8")


def xf_prepare_audio(src_path):
    """转成 16kHz 单声道 wav（讯飞常规要求），返回 (临时文件路径, 时长秒)。"""
    import soundfile as sf
    audio, sr = sf.read(src_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(audio) * SR / sr))
        x_in = np.linspace(0, 1, len(audio), endpoint=False)
        x_out = np.linspace(0, 1, n_out, endpoint=False)
        audio = np.interp(x_out, x_in, audio).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", dir=OUT_DIR, delete=False)
    tmp.close()
    sf.write(tmp.name, audio, SR, subtype="PCM_16")
    return tmp.name, len(audio) / SR


def xf_collect_words(node, bg, out_words):
    """递归收集带 cw/wb/we 的词节点；绝对时间 = bg + 帧*10ms。"""
    if isinstance(node, dict):
        if "cw" in node and ("wb" in node or "we" in node):
            wb = int(node.get("wb") or 0)
            we = int(node.get("we") or 0)
            for c in node.get("cw") or []:
                w = (c.get("w") or "").strip()
                if w and c.get("wp") != "p":
                    out_words.append({"word": w,
                                      "start_ms": bg + wb * 10,
                                      "end_ms": bg + we * 10})
        for v in node.values():
            xf_collect_words(v, bg, out_words)
    elif isinstance(node, list):
        for v in node:
            xf_collect_words(v, bg, out_words)


def xf_parse_order_result(text):
    data = json.loads(text)
    words = []
    for seg in data.get("lattice", []):
        one = json.loads(seg["json_1best"])
        st = one.get("st", {})
        bg = int(st.get("bg") or one.get("bg") or 0)
        xf_collect_words(st, bg, words)
    return words


def transcribe_words(audio_path):
    """对音频跑讯飞转写，返回 [{"word","start_ms","end_ms"}]；失败抛异常。"""
    import requests
    if not os.path.exists(XF_CONFIG):
        raise RuntimeError("缺少配置文件 " + XF_CONFIG)
    with open(XF_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    appid = (cfg.get("appid") or "").strip()
    secret_key = (cfg.get("secret_key") or "").strip()
    if not appid or not secret_key:
        raise RuntimeError("讯飞配置.json 缺 appid / secret_key")

    tmp_audio, duration_s = xf_prepare_audio(audio_path)
    try:
        ts = str(int(time.time()))
        signa = xf_make_signa(appid, secret_key, ts)
        params = {
            "appId": appid, "signa": signa, "ts": ts,
            "fileSize": os.path.getsize(tmp_audio),
            "fileName": os.path.basename(tmp_audio),
            "duration": str(int(round(duration_s))),
        }
        with open(tmp_audio, "rb") as f:
            data = f.read()
        resp = requests.post(XF_HOST + "/upload?" + urlencode(params),
                             headers={"Content-type": "application/json"},
                             data=data, timeout=60).json()
        if resp.get("code") != "000000":
            raise RuntimeError("上传失败：" + json.dumps(resp, ensure_ascii=False))
        order_id = resp["content"]["orderId"]
        for _ in range(60):
            time.sleep(3)
            r = requests.post(XF_HOST + "/getResult?" + urlencode({
                "appId": appid, "signa": signa, "ts": ts,
                "orderId": order_id, "resultType": "transfer"}), timeout=60).json()
            if r.get("code") != "000000":
                raise RuntimeError("查询失败：" + json.dumps(r, ensure_ascii=False))
            info = r["content"].get("orderInfo", {})
            status = info.get("status")
            if status == 4:
                return xf_parse_order_result(r["content"].get("orderResult"))
            if status == -1:
                raise RuntimeError("订单失败：" + json.dumps(info, ensure_ascii=False))
        raise RuntimeError("转写超时")
    finally:
        try:
            os.remove(tmp_audio)
        except Exception:
            pass


# ================= 词块构建（连读区合并） =================
def load_merge_zones():
    """从 音素对比.json 取连读区，返回 [(词1, 词2), ...]。
    连读在语言上切不开，整块看；这里复用音素脚本已识别好的结果。"""
    if not os.path.exists(PHONEME_JSON):
        return []
    try:
        with open(PHONEME_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    zones = []
    for z in (data.get("词切分", {}).get("连读区") or []):
        ws = z.get("词") or []
        if len(ws) >= 2:
            zones.append(tuple(ws))
    return zones


def build_blocks(words, zones):
    """把词序列按连读区合并成块，返回 [{"词":[...], "start_ms", "end_ms", "连读"}]。
    块的位置由词在序列中的次序决定 —— 全程只用序号，不用绝对秒数。"""
    used = [False] * len(words)
    blocks = []
    i = 0
    while i < len(words):
        if used[i]:
            i += 1
            continue
        matched = None
        for zone in zones:
            if i + len(zone) <= len(words):
                if tuple(w["word"] for w in words[i:i + len(zone)]) == zone:
                    matched = zone
                    break
        if matched:
            grp = words[i:i + len(matched)]
            blocks.append({"词": [w["word"] for w in grp],
                           "start_ms": grp[0]["start_ms"],
                           "end_ms": grp[-1]["end_ms"],
                           "连读": True})
            for k in range(i, i + len(matched)):
                used[k] = True
            i += len(matched)
        else:
            blocks.append({"词": [words[i]["word"]],
                           "start_ms": words[i]["start_ms"],
                           "end_ms": words[i]["end_ms"],
                           "连读": False})
            used[i] = True
            i += 1
    return blocks


def block_span(block):
    return (block["end_ms"] - block["start_ms"]) / 1000.0


# ================= 停顿（只判句子边界） =================
def find_boundary_pauses(words):
    """词间空隙 >= 阈值处记为停顿候选。
    注：正常口语词与词之间本就是连着走的，这里只作为"句子边界"的候选信号，
        报告里统称"停顿"，不逐词报"第几个词后停了几秒"。"""
    result = []
    for i in range(len(words) - 1):
        gap = (words[i + 1]["start_ms"] - words[i]["end_ms"]) / 1000.0
        if gap >= BOUNDARY_GAP_S:
            result.append({"after_word": words[i]["word"],
                           "after_index": i + 1,
                           "gap": round(gap, 3)})
    return result


# ================= 时长复核（能量实证） =================
def measure_voiced(env, block):
    """块区间内"实际有声"的时长（能量 > 全句 max*比例 的帧数 × hop）。"""
    hop_s = HOP_MS / 1000.0
    if len(env) == 0 or env.max() <= 0:
        return 0.0
    thr = VOICED_RATIO * env.max()
    i0 = max(0, int(block["start_ms"] / 1000.0 / hop_s))
    i1 = min(len(env), max(i0 + 1, int(block["end_ms"] / 1000.0 / hop_s) + 1))
    return float(np.sum(env[i0:i1] > thr)) * hop_s


def resolve_duration(block, env):
    """讯飞时长 vs 能量实证：差得离谱时以实证为准（讯飞词边界是猜的，会虚高）。"""
    span = block_span(block)
    voiced = measure_voiced(env, block)
    if voiced > 0 and abs(span - voiced) > FIX_TOLERANCE_S:
        return voiced, "实证", span, voiced
    return span, "讯飞", span, voiced


# ================= 重音（信号，各侧自比） =================
def block_signal_features(env, f0, blocks):
    """每块算 响度峰值 / 音高均值（音高检测不到浊音时为 None）。"""
    hop_s = HOP_MS / 1000.0
    feats = []
    for b in blocks:
        i0 = max(0, int(b["start_ms"] / 1000.0 / hop_s))
        i1 = min(len(env), max(i0 + 1, int(b["end_ms"] / 1000.0 / hop_s) + 1))
        seg_env = env[i0:i1]
        loud = float(seg_env.max()) if len(seg_env) else 0.0

        nf0 = min(len(f0), len(env))
        seg_f0 = f0[i0:min(i1, nf0)]
        voiced = seg_f0[seg_f0 > 0] if len(seg_f0) else np.array([])
        pitch = float(voiced.mean()) if len(voiced) >= 2 else None

        feats.append({"响度": loud, "音高": pitch})
    return feats


def detrend_linear(values):
    """减去线性趋势线，消除"句首响句尾弱"的音量下倾，只留局部起伏。
    全 None / 全 0 时原样返回。"""
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return list(values)
    x = np.arange(len(arr))[valid]
    y = arr[valid]
    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * np.arange(len(arr)) + intercept
    out = [None if np.isnan(v) else float(v - t)
           for v, t in zip(arr, trend)]
    return out


def mark_stress(feats, x, detrend=True):
    """各侧自比：某块 响度/音高 超过【本侧】均值 × x 即算突出；任一过线即为突出块。
    返回 (突出块序号列表, 逐块明细)。"""
    louds = [f["响度"] for f in feats]
    pitches_raw = [f["音高"] for f in feats]

    louds_use = detrend_linear(louds) if detrend else louds
    pitches_use = detrend_linear(pitches_raw) if detrend else pitches_raw

    valid_loud = [v for v in louds_use if v is not None]
    valid_pitch = [v for v in pitches_use if v is not None]
    # 去趋势后均值为 0 附近，改用"有效值均值 + 标准差"作基准，否则阈值无意义
    mean_loud = float(np.mean(valid_loud)) if valid_loud else 0.0
    std_loud = float(np.std(valid_loud)) if len(valid_loud) > 1 else 0.0
    mean_pitch = float(np.mean(valid_pitch)) if valid_pitch else 0.0
    std_pitch = float(np.std(valid_pitch)) if len(valid_pitch) > 1 else 0.0

    marked, detail = [], []
    for i, f in enumerate(feats):
        flags = []
        lv = louds_use[i]
        pv = pitches_use[i]
        if lv is not None and std_loud > 0 and lv > mean_loud + (x - 1.0) * std_loud * 3:
            flags.append("响")
        if pv is not None and std_pitch > 0 and pv > mean_pitch + (x - 1.0) * std_pitch * 3:
            flags.append("高")
        if flags:
            marked.append(i + 1)
        detail.append({"序号": i + 1,
                       "词": "+".join(f.get("词", [])) if "词" in f else "",
                       "命中": flags})
    return marked, detail


# ---------------- DTW 距离（保留，供后续语调使用） ----------------
def dtw_distance(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    b = (b - b.min()) / (b.max() - b.min() + 1e-9)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / max(n, m)


# ---------------- 节奏分 ----------------
def score_blocks(pairs):
    """按块算相对差，给每块一个判定；再汇总成节奏分。
    节奏分 = 100 - 每块相对差（截断到 1.0）的平均值 × 100，最低 0 分。"""
    items = []
    for p in pairs:
        dr, dm = p["原音频时长"], p["我的时长"]
        if dr <= 0:
            continue
        rel = (dm - dr) / dr
        if rel >= REL_FAST:
            verdict = "要读快一些"
        elif rel <= REL_SLOW:
            verdict = "要读慢一些"
        else:
            verdict = "差不多"
        items.append({"序号": p["序号"], "块": p["块"], "连读": p["连读"],
                      "原音频时长": round(dr, 3), "我的时长": round(dm, 3),
                      "相对差": round(rel, 4), "判定": verdict})
    if not items:
        return items, 0.0
    penalty = np.mean([min(abs(it["相对差"]), 1.0) for it in items])
    return items, max(0.0, (1 - penalty) * 100)


# ---------------- 画图 ----------------
def plot_blocks(items, title="节奏对比（按词块序号）"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = ["%d.%s" % (it["序号"], it["块"]) for it in items]
    x = np.arange(len(items))
    dr = [it["原音频时长"] for it in items]
    dm = [it["我的时长"] for it in items]
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, dr, width, label="原音频", color="#c0392b")
    ax.bar(x + width / 2, dm, width, label="我的模仿", color="#2471a3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("块时长 (秒)")
    ax.set_title(title + "   ·   横轴是词块序号，不是绝对时间")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    # 标出差异大的块
    for i, it in enumerate(items):
        if abs(it["相对差"]) >= REL_FAST:
            ax.annotate(it["判定"], (i, max(dr[i], dm[i])),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, color="#7d3c98")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    out("图已保存：" + OUT_PNG)


def write_json(net_r, net_m, pauses_r, pauses_m, items, rhythm_score,
               stress_r, stress_m, blocks_r, blocks_m, words_r, words_m):
    """写 节奏对比.json —— 给机器/综合评分器读。
    注：字段「节奏分」保持不变（overall_score.py 按这个名字读）；
       「语调分」本次不产出（语调整理中），overall_score.py 会自动跳过缺项。"""
    nr = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_r)}
    nm = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_m)}
    miss = sorted(set(stress_r) - set(stress_m))
    extra = sorted(set(stress_m) - set(stress_r))
    hit = sorted(set(stress_r) & set(stress_m))

    data = {
        "节奏_讯飞": {
            "整句净时长": {"原音频": round(net_r, 3), "我的": round(net_m, 3),
                           "相对差": round((net_m - net_r) / net_r, 4) if net_r > 0 else 0},
            "词块数": {"原音频": len(blocks_r), "我的": len(blocks_m)},
            "序号对齐风险": (len(blocks_r) != len(blocks_m)),
            "句子边界停顿": {
                "原音频处数": len(pauses_r), "我的处数": len(pauses_m),
                "原音频": pauses_r, "我的": pauses_m,
            },
            "词块时长": items,
        },
        "重音_信号": {
            "阈值X": STRESS_X,
            "去趋势": DETREND,
            "原音频突出块": [{"序号": i, "块": nr.get(i, "?")} for i in stress_r],
            "我的突出块": [{"序号": i, "块": nm.get(i, "?")} for i in stress_m],
            "漏（原音频有）": [{"序号": i, "块": nr.get(i, "?")} for i in miss],
            "多（我的）": [{"序号": i, "块": nm.get(i, "?")} for i in extra],
            "命中": [{"序号": i, "块": nr.get(i, "?")} for i in hit],
            "说明": "跟原音频比结构，不下对错结论（对错取决于说话意图）",
        },
        "词表": {"原音频": words_r, "我的": words_m},
        "节奏分": round(rhythm_score, 1),
        "语调分": None,          # 语调整理中，暂不出分
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    log_setup("节奏/重音")
    out("=" * 62)
    out("   还原度对比 · 节奏 / 重音")
    out("=" * 62)

    ref = load_mono(REF_FILE)
    mine = load_mono(MY_FILE)
    if ref is None or mine is None:
        out("")
        out("X 没找到音频文件。请把两个 wav 放进本文件夹：")
        out("    " + REF_FILE)
        out("    " + MY_FILE)
        log_done("（音频缺失，提前退出）")
        return

    hop_s = HOP_MS / 1000.0
    log_step("读入音频")
    log("   原音频 %.2fs ｜ 我的模仿 %.2fs" % (len(ref) / SR, len(mine) / SR),
        verbose=True)

    log_step("算能量包络 / 音高曲线")
    env_r, env_m = frame_rms(ref), frame_rms(mine)
    f0_r, f0_m = pitch_contour(ref), pitch_contour(mine)

    # ---- 讯飞转写 ----
    log_step("讯飞转写（拿词级时间戳）")
    try:
        words_r = transcribe_words(REF_FILE)
        words_m = transcribe_words(MY_FILE)
        log("   原音频 %d 词 ｜ 我的 %d 词" % (len(words_r), len(words_m)),
            verbose=True)
    except Exception as e:
        out("")
        out("X 讯飞转写失败，本次节奏对比无法进行：")
        out("    " + str(e))
        out("  （检查 讯飞配置.json / 网络 / 额度）")
        # 降级：仍出一份"只有信号侧"的结果，但节奏分留空
        log_error(str(e))
        try:
            write_json(0, 0, [], [], [], 0.0, [], [], [], [], words_r if 'words_r' in dir() else [], [])
        except Exception:
            pass
        log_done("（讯飞不可用，提前退出）")
        return

    zones = load_merge_zones()
    if zones:
        log("   连读区：" + "、".join("+".join(z) for z in zones), verbose=True)

    blocks_r = build_blocks(words_r, zones)
    blocks_m = build_blocks(words_m, zones)

    # ---- 词块配对（按序号）----
    log_step("词块配对（按序号）")
    pairs = []
    n = max(len(blocks_r), len(blocks_m))
    for i in range(n):
        br = blocks_r[i] if i < len(blocks_r) else None
        bm = blocks_m[i] if i < len(blocks_m) else None
        if br is None or bm is None:
            continue
        dr, src_r, span_r, _ = resolve_duration(br, env_r)
        dm, src_m, span_m, _ = resolve_duration(bm, env_m)
        pairs.append({"序号": i + 1,
                      "块": "+".join(br["词"]),
                      "连读": br["连读"],
                      "原音频时长": dr, "我的时长": dm,
                      "原音频时长来源": src_r, "我的时长来源": src_m})

    # ---- 句子边界停顿 ----
    pauses_r = find_boundary_pauses(words_r)
    pauses_m = find_boundary_pauses(words_m)

    # ---- 重音 ----
    log_step("重音判定（各侧自比）")
    feats_r = block_signal_features(env_r, f0_r, blocks_r)
    feats_m = block_signal_features(env_m, f0_m, blocks_m)
    for i, b in enumerate(blocks_r):
        feats_r[i]["词"] = b["词"]
    for i, b in enumerate(blocks_m):
        feats_m[i]["词"] = b["词"]
    marked_r, _ = mark_stress(feats_r, STRESS_X, DETREND)
    marked_m, _ = mark_stress(feats_m, STRESS_X, DETREND)

    # ---- 节奏分 ----
    items, rhythm_score = score_blocks(pairs)

    # ---- 终端简报 ----
    net_r = (words_r[-1]["end_ms"] - words_r[0]["start_ms"]) / 1000.0
    net_m = (words_m[-1]["end_ms"] - words_m[0]["start_ms"]) / 1000.0

    out("")
    out("整句净时长：原音频 %.2fs   |   我的 %.2fs   |   相对差 %+.1f%%"
        % (net_r, net_m, (net_m - net_r) / net_r * 100 if net_r > 0 else 0))
    out("句子边界停顿：原音频 %d 处   |   我的 %d 处"
        % (len(pauses_r), len(pauses_m)))
    for label, ps in (("原音频", pauses_r), ("我的", pauses_m)):
        for p in ps:
            out("   [%s] \"%s\" 后停了 %.2fs" % (label, p["after_word"], p["gap"]))
    out("")
    if len(blocks_r) != len(blocks_m):
        out("⚠ 两侧词块数不一致（%d vs %d），序号可能错位，结果仅供参考"
            % (len(blocks_r), len(blocks_m)))
    out("词块时长对比（横轴=序号，不卡点）：")
    for it in items:
        tag = " [连读]" if it["连读"] else ""
        out("   #%-2d %-16s 原音频 %.2fs | 我 %.2fs | %+.1f%%  %s%s"
            % (it["序号"], it["块"], it["原音频时长"], it["我的时长"],
               it["相对差"] * 100, it["判定"], tag))
    out("")
    nr = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_r)}
    nm = {i + 1: "+".join(b["词"]) for i, b in enumerate(blocks_m)}
    out("重音落点（跟原音频比结构，不下对错结论）：")
    out("   原音频：" + ("、".join("#%d %s" % (i, nr[i]) for i in marked_r)
                       or "（无）"))
    out("   我的：  " + ("、".join("#%d %s" % (i, nm[i]) for i in marked_m)
                       or "（无）"))
    miss = sorted(set(marked_r) - set(marked_m))
    extra = sorted(set(marked_m) - set(marked_r))
    hit = sorted(set(marked_r) & set(marked_m))
    out("   → 漏（原音频有、我没有）：" + ("、".join("#%d %s" % (i, nr[i])
                                              for i in miss) or "无"))
    out("   → 多（我有、原音频没有）：" + ("、".join("#%d %s" % (i, nm[i])
                                              for i in extra) or "无"))
    out("   → 命中：" + ("、".join("#%d %s" % (i, nr[i]) for i in hit) or "无"))
    out("")
    out("节奏分：%.1f / 100" % rhythm_score)
    out("语调分：暂不出（判分标准重做中）")
    out("")

    # ---- 画图 ----
    try:
        plot_blocks(items)
    except Exception as e:
        log("（画图失败，不影响上面数据：%s）" % e, show=True)

    # ---- 写 json ----
    try:
        write_json(net_r, net_m, pauses_r, pauses_m, items, rhythm_score,
                   marked_r, marked_m, blocks_r, blocks_m, words_r, words_m)
        out("数据（机器）：节奏对比.json")
    except Exception as e:
        log("（写 json 失败：%s）" % e, show=True)

    log_done("节奏 / 重音")


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
