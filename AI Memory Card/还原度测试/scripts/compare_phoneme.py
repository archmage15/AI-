# -*- coding: utf-8 -*-
r"""还原度对比 · 第二块：音素

把「原音频」和「你的模仿」分别喂给音素识别模型，得到两条音素序列，
对齐后告诉你：哪些音没还原、哪里多了/少了音（连读、弱化、吞音就看这里）。

模型：facebook/wav2vec2-lv-60-espeak-cv-ft
  - 首次运行会自动从 HuggingFace 下载（约 1.2GB），之后走本地缓存，不再下
  - 输出 eSpeak 风格音素（接近 IPA）

用法：
  1. 把两个音频放进 audio\ 文件夹，命名成：原音频.wav 和 模仿.wav
  2. 双击 scripts\run_phoneme.bat（或根目录的 一键跑还原度.bat）
  3. 屏幕出对比结果，结果写到 output\ 文件夹

目录约定（本脚本在 scripts\ 里，用 ROOT 指回上一层）：
  audio\   放录音（你手动放，只读）
  output\  放本次结果（自动生成，可随时清空）
  models\  放音素模型（只读，缺了自动下载）

可选（给音素标"单词"）：
  把 讯飞配置.json（appid / secret_key）放在 ROOT，脚本会额外调用讯飞转写，
  把原音频转成"词 + 时间"，从而在每条纠错后面标出对应的单词。
  没有该文件 / 断网 / 无额度时自动跳过，只给音素级结果。

依赖：torch / transformers / soundfile / numpy / requests
      （解码直接用模型自带 vocab.json，不需要额外装 phonemizer）
"""
import base64
import difflib
import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
import unicodedata
from urllib.parse import urlencode

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 国内访问 HuggingFace：官方 huggingface.co 常常连不上，默认走国内镜像。
# 想用官方站，运行前设环境变量 HF_ENDPOINT=https://huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 关掉 xet 传输协议：它和镜像站不兼容，会让下载卡住。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ---- 目录约定 ----
# 本脚本位于 <项目>\scripts\ 下，ROOT = 项目根（上一层）
#   audio\  你放录音     output\  脚本写结果     models\  模型权重     根目录  讯飞配置.json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(ROOT, "audio")
OUT_DIR = os.path.join(ROOT, "output")
for _d in (SRC_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

REF_FILE = os.path.join(SRC_DIR, "原音频.wav")
MY_FILE = os.path.join(SRC_DIR, "模仿.wav")
OUT_JSON = os.path.join(OUT_DIR, "音素对比.json")
# 给人看的报告（json 留给机器/AI，这份是纯文本、人读的）
OUT_REPORT = os.path.join(OUT_DIR, "音素对比_报告.txt")

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
# 模型存放目录：默认放在 ROOT\models\ 里（首次运行自动下载）。
# 想改到别处，设环境变量 PHONEME_MODEL_DIR 指向目标目录即可。
LOCAL_DIR = os.environ.get("PHONEME_MODEL_DIR") or os.path.join(ROOT, "models", "wav2vec2-espeak")
SR = 16000

# 讯飞转写（LFASR）配置：放在 ROOT 下，用于给音素打"单词"标签。
# 没有这个文件 / 断网 / 无额度时，脚本自动降级成"只给音素级结果"。
XF_CONFIG = os.path.join(ROOT, "讯飞配置.json")
XF_HOST = "https://raasr.xfyun.cn/v2/api"
# wav2vec2 特征提取下采样步长 320 采样点 @16kHz -> 每帧 20ms
FRAME_MS = 20

# ---- 连读区判定 ----
# 讯飞在连读处给的"词边界"本身不可靠：两个词黏成一团时，它会把整团划给前一个词，
# 后一个词只剩几十毫秒的碎屑（实测 "are" 只有 30ms）。
# 连读在语言上本来就是一整团、切不开的东西，所以这里不硬切，而是标成"连读区"。
MIN_WORD_MS = 80        # 短于此长度的词，视为"碎片词"，可能与邻居构成连读
MERGE_GAP_MS = 60       # 碎片词与邻居间隔小于此值才合并（FRAME_MS * 3）

# 「参考音素表」：由 gen_ref_phonemes.py 离线预生成（espeak-ng 是 GPL，只在本机跑一次）
# 运行时只读这个纯数据文件 —— 绝不 import / 链接 espeak-ng
REF_PHONEME_JSON = os.path.join(SRC_DIR, "参考音素.json")

# 判"这个词读错了"的阈值（compute_word_errors 用）
WORD_ERR_RATIO = 0.40   # 错音占该词音素比例 ≥ 此值
WORD_ERR_MIN = 2        # 或错音个数 ≥ 此值


# ---- 日志（统一模块，带时间戳；详细模式见 logkit.py 顶部说明）----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logkit import log, out, log_setup, log_done, log_step, log_error, VERBOSE


def load_ref_phonemes():
    """读离线预生成的参考音素表。返回 dict 或 None（读不到 → 下游降级）。"""
    if not os.path.exists(REF_PHONEME_JSON):
        return None
    try:
        with open(REF_PHONEME_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log("⚠ 参考音素.json 读失败：%s" % e, show=True)
        return None
    if not data.get("音素") or not data.get("词边界"):
        log("⚠ 参考音素.json 内容不完整（缺 音素 / 词边界）", show=True)
        return None
    return data


def load_mono(path):
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(audio) * SR / sr))
        x_in = np.linspace(0, 1, len(audio), endpoint=False)
        x_out = np.linspace(0, 1, n_out, endpoint=False)
        audio = np.interp(x_out, x_in, audio)
    return audio.astype(np.float32)


def load_vocab(model_path):
    """读模型自带符号表，得到 id -> 音素 的映射"""
    with open(os.path.join(model_path, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    return {int(v): k for k, v in vocab.items()}


def ctc_decode(ids, id2tok, blank=0, specials=(1, 2, 3)):
    """CTC 解码：合并重复 -> 去空白/特殊符 -> 映射成音素。
    同时返回每个音素所在的帧号（用于按时间归属到单词）。"""
    out = []
    frames = []
    prev = None
    for pos, i in enumerate(ids):
        i = int(i)
        if i == prev:
            continue
        prev = i
        if i == blank or i in specials:
            continue
        tok = id2tok.get(i, "")
        if tok:
            out.append(tok)
            frames.append(pos)
    return out, frames


def load_feature_extractor(model_path):
    try:
        from transformers import AutoFeatureExtractor
        return AutoFeatureExtractor.from_pretrained(model_path)
    except Exception:
        from transformers import Wav2Vec2FeatureExtractor
        return Wav2Vec2FeatureExtractor.from_pretrained(model_path)


def model_ready(path):
    """判断模型目录是否真的完整。

    只看文件在不在不够 —— HF 缓存异常时会出现 0 字节的坏文件，
    加载时会报 "is not a valid JSON file"。所以这里顺带查大小。
    """
    need = ["config.json", "preprocessor_config.json", "vocab.json"]
    for name in need:
        p = os.path.join(path, name)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            return False
    weight = os.path.join(path, "pytorch_model.bin")
    if os.path.exists(weight):
        return os.path.getsize(weight) > 0
    # 有些仓库只提供 safetensors 格式
    return any(f.endswith(".safetensors") and os.path.getsize(os.path.join(path, f)) > 0
               for f in os.listdir(path))


def bad_files(path):
    """列出目录里 0 字节的坏文件（HF 缓存符号链接落盘失败会留下这些）。"""
    if not os.path.isdir(path):
        return []
    out = []
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isfile(p) and os.path.getsize(p) == 0:
            out.append(name)
    return out


def repair_model():
    """删掉坏文件 + 清掉缓存索引里对应记录，让下次下载真的重下。

    背景：HF 缓存会记"这个文件我下过了"。若上次下成 0 字节，
    重跑时它会跳过不重下，坏文件就一直留着 —— 必须连缓存记录一起删。
    """
    broken = bad_files(LOCAL_DIR)
    if not broken:
        return []
    log("")
    log("发现 %d 个 0 字节的坏文件，正在自动修复：" % len(broken))
    cache_dir = os.path.join(LOCAL_DIR, ".cache", "huggingface", "download")
    for name in broken:
        os.remove(os.path.join(LOCAL_DIR, name))
        log("   已删除坏文件：" + name)
        meta = os.path.join(cache_dir, name + ".metadata")
        if os.path.exists(meta):
            os.remove(meta)
            log("   已清除其缓存记录，确保会重新下载")
    return broken


def ensure_model():
    """模型不存在或损坏则修复/下载（走 local_dir 模式）。返回本地模型目录。"""
    repair_model()                      # 先清坏文件，避免"已存在故跳过"的误判
    if model_ready(LOCAL_DIR):
        return LOCAL_DIR
    log("")
    log("首次运行：下载音素模型（约 1.2GB，走 hf-mirror 镜像，请耐心等）...")
    log("存放位置：" + LOCAL_DIR)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)
    repair_model()                      # 下载后若仍有坏文件，再修一轮
    if not model_ready(LOCAL_DIR):
        broken = bad_files(LOCAL_DIR)
        raise RuntimeError(
            "模型文件不完整，自动修复后仍未通过检查。\n"
            "  仍然为 0 字节的文件：" + (", ".join(broken) if broken else "（无，可能是缺文件）") + "\n"
            "  解决办法：删掉整个目录后重跑本脚本，让它重新下载：\n"
            "    " + LOCAL_DIR)
    log("模型下载完成。")
    return LOCAL_DIR


# ---------------- 讯飞转写（拿"词 + 时间"，给音素打单词标签）----------------

def xf_make_signa(appid, secret_key, ts):
    """讯飞 LFASR v2 签名：base64(HMAC-SHA1(secret_key, md5(appid+ts)))"""
    md5_hex = hashlib.md5((appid + ts).encode("utf-8")).hexdigest()
    raw = hmac.new(secret_key.encode("utf-8"), md5_hex.encode("utf-8"), hashlib.sha1).digest()
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
    return tmp.name, len(audio) / float(SR)


def xf_collect_words(node, bg, out):
    """递归收集带 cw/wb/we 的词节点；绝对时间 = bg + 帧*10ms。"""
    if isinstance(node, dict):
        if "cw" in node and ("wb" in node or "we" in node):
            wb = int(node.get("wb") or 0)
            we = int(node.get("we") or 0)
            for c in node.get("cw") or []:
                w = (c.get("w") or "").strip()
                if w and c.get("wp") != "p":
                    out.append({"word": w, "start_ms": bg + wb * 10, "end_ms": bg + we * 10})
        for v in node.values():
            xf_collect_words(v, bg, out)
    elif isinstance(node, list):
        for v in node:
            xf_collect_words(v, bg, out)


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


def assign_word_indices(times_ms, words):
    """给每个音素时间点找所属单词下标（落在词区间内；否则取最近词的边界）。"""
    if not words:
        return [None] * len(times_ms)
    out = []
    for t in times_ms:
        hit = None
        for wi, w in enumerate(words):
            if w["start_ms"] <= t <= w["end_ms"]:
                hit = wi
                break
        if hit is None:
            hit = min(range(len(words)),
                      key=lambda wi: min(abs(t - words[wi]["start_ms"]),
                                         abs(t - words[wi]["end_ms"])))
        out.append(hit)
    return out


def word_indices(idxs, words):
    return sorted({i for i in idxs if i is not None})


# ---------------- 连读区识别（用讯飞词边界校正音素归词）----------------
#
# 为什么要有这一段：
#   本地模型是 CTC 盲拆，没有"词"的概念。原来把每个音素按时间点塞进讯飞的词区间里，
#   塞不进去的就甩给"最近的词"。在连读处（如 Where are），讯飞自己都把整团划给了
#   Where，are 只剩 30ms 碎屑 —— 结果音素被整团错归给 Where，报告只能含糊写
#   "Where → are 之间"。
#
#   这里的做法：先看讯飞给的词区间靠不靠谱。够宽的正常归词；太窄的判定为"碎片词"，
#   与相邻词合成"连读区"，明确告诉学生"这里是一个连读，按整体看"——因为连读本来
#   就切不开，硬切反而错。

def build_word_spans(words, frame_times, total_ms):
    """把讯飞的词表变成一段段稳定的区间，并统计每段里落了几个音素。

    做三件修补，防止词表边界与音素时间对不上：
      1) 首尾补边：第 0 个词之前、最后一个词之后若还有音素，也要有区间可归。
      2) 单调修正：强制 start 严格递增，避免区间互相挤夹。
      3) 容差命中：FRAME_MS 之内都算落在该词里（帧边界会抖）。
    """
    if not words:
        return []

    spans = []
    for i, w in enumerate(words):
        s = int(w.get("start_ms") or 0)
        e = int(w.get("end_ms") or 0)
        if e < s:
            e = s
        spans.append({"idx": i, "word": w["word"], "start_ms": s, "end_ms": e, "n_ph": 0})

    # 1) 首尾补边：让整条时间轴都被覆盖，音素不会有"无主"的空白
    if spans:
        spans[0]["start_ms"] = 0
        spans[-1]["end_ms"] = max(int(total_ms), spans[-1]["end_ms"])

    # 2) 单调修正：start 必须严格递增
    for i in range(1, len(spans)):
        if spans[i]["start_ms"] <= spans[i - 1]["start_ms"]:
            spans[i]["start_ms"] = spans[i - 1]["start_ms"] + 1
        if spans[i]["start_ms"] >= spans[i]["end_ms"]:
            spans[i]["end_ms"] = spans[i]["start_ms"] + 1

    # 3) 容差命中的方式统计每个区间的音素数（用区间中点判，避免边界抖动）
    for t in frame_times:
        t = int(t)
        for sp in spans:
            lo = sp["start_ms"] - FRAME_MS
            hi = sp["end_ms"] + FRAME_MS
            if lo <= t <= hi:
                sp["n_ph"] += 1
                break

    for sp in spans:
        span_ms = sp["end_ms"] - sp["start_ms"]
        sp["span_ms"] = span_ms
        # 碎片词 = 时长过短（只按时长判，不看音素数）。
        # 有些长词可能一个音都没分到（如感叹词 oh 在原音频里没被单独拆出音素），
        # 那是识别粒度问题，不是连读，不该并进连读区。
        sp["short"] = span_ms < MIN_WORD_MS

    return spans


def build_text_spans(word_bounds, owner, frame_times):
    """用「参考文本给的词边界 + CTC 帧给的时间」拼出 spans。

    ⚠️ 与旧 build_word_spans **同构**（同样的键），所以 pair_fragments 逐字不改就能接：
        {idx, word, start_ms, end_ms, span_ms, n_ph, short}

    参数
      word_bounds : 参考音素.json 的 词边界 [{word, a, b, 角色}]
      owner       : map_words_to_audio 的输出 —— 每个「实际音素」归属的词下标
      frame_times : 每个实际音素的时间(ms)，与 owner 等长

    沿用了旧实现的两个修补（首尾补边、单调修正）—— 换数据源不等于可以少护栏。
    """
    if not word_bounds:
        return []

    spans = []
    for wi, wb in enumerate(word_bounds):
        ts = [frame_times[j] for j in range(len(owner)) if owner[j] == wi]
        spans.append({"idx": wi, "word": wb["word"],
                      "start_ms": int(min(ts)) if ts else None,
                      # frame_times 存的是音素的**起始**帧，末尾要补一帧（20ms）
                      "end_ms": (int(max(ts)) + FRAME_MS) if ts else None,
                      "n_ph": len(ts)})

    # ⚠️ 不做"首尾补边"（旧 build_word_spans 有，这里实测**不该有**）：
    # 旧实现的时间来自讯飞词表，补边只是填边角空白；
    # 新实现的时间**直接来自音素帧** —— 把首词 start 硬拉成 0 会让它的时长虚高
    # （实测：so 真实 740→1030ms，补边后变成 0→1000ms）。那会污染节奏维度。

    # 1) 一个音素都没分到的词（如原音频里的感叹词 oh）：用邻居边界补一个区间。
    #    绝不能给 0 宽 —— 那会被 pair_fragments 当成"碎片词"并进连读区，纯属误报。
    for i, sp in enumerate(spans):
        if sp["start_ms"] is not None and sp["end_ms"] is not None:
            continue
        prev_end = None
        for k in range(i - 1, -1, -1):
            if spans[k]["end_ms"] is not None:
                prev_end = spans[k]["end_ms"]
                break
        next_start = None
        for k in range(i + 1, len(spans)):
            if spans[k]["start_ms"] is not None:
                next_start = spans[k]["start_ms"]
                break
        if prev_end is None and next_start is None:
            sp["start_ms"] = sp["end_ms"] = 0
        elif prev_end is None:
            sp["start_ms"], sp["end_ms"] = next_start - 1, next_start
        elif next_start is None:
            sp["start_ms"], sp["end_ms"] = prev_end, prev_end + 1
        else:
            mid = (prev_end + next_start) // 2
            sp["start_ms"], sp["end_ms"] = mid, max(mid + 1, next_start)

    # 2) 单调修正：start 必须严格递增（帧边界会抖，可能压出 start >= end）
    for i in range(1, len(spans)):
        if spans[i]["start_ms"] <= spans[i - 1]["start_ms"]:
            spans[i]["start_ms"] = spans[i - 1]["start_ms"] + 1
        if spans[i]["start_ms"] >= spans[i]["end_ms"]:
            spans[i]["end_ms"] = spans[i]["start_ms"] + 1

    for sp in spans:
        sp["span_ms"] = sp["end_ms"] - sp["start_ms"]
        sp["short"] = sp["span_ms"] < MIN_WORD_MS

    return spans


def pair_fragments(spans):
    """把"碎片词"和它的邻居并成"连读区"。

    三条限制，都是为了防止"越并越大"：
      1) 只并"时长过短"的碎片词（不看 n_ph —— 有些长词可能一个音都没分到，
         那是别的问题，不是连读，不该并）。
      2) 一个词一旦进了某个连读区，就不再作为下一轮的合并起点（防连锁传染）。
      3) 连读区最多 2 个词 —— 真连读是"两个词黏在一起"，三个词整体糊成一团
         不是典型连读，并进去只会让报告更糊。

    返回 (主区间列表, 连读区列表, 区间下标 -> 连读区下标 映射)
    """
    if not spans:
        return [], [], {}

    zone_of = {}          # 主区间下标 -> 连读区下标
    zones = []
    consumed = set()      # 已被并进连读区、不再单独出现的主区间下标

    for i, sp in enumerate(spans):
        if not sp["short"] or i in consumed:
            continue
        prev_sp = spans[i - 1] if i > 0 else None
        next_sp = spans[i + 1] if i + 1 < len(spans) else None

        # 邻居是否"紧挨着"（判断连读的关键：够近才可能黏在一起）
        prev_near = (prev_sp is not None and (i - 1) not in consumed
                     and (sp["start_ms"] - prev_sp["end_ms"]) <= MERGE_GAP_MS)
        next_near = (next_sp is not None and (i + 1) not in consumed
                     and (next_sp["start_ms"] - sp["end_ms"]) <= MERGE_GAP_MS)

        if prev_near:
            # 优先朝前并（连读多是"前词拖住了后词"）
            members = [i - 1, i]
        elif next_near:
            members = [i, i + 1]
        else:
            continue          # 孤立短词：不并，保持独立

        if any(m in consumed for m in members):
            continue

        zi = len(zones)
        for m in members:
            consumed.add(m)
            zone_of[m] = zi
        zones.append({
            "zone_idx": zi,
            "member_idx": members,
            "words": [spans[m]["word"] for m in members],
            "start_ms": spans[members[0]]["start_ms"],
            "end_ms": spans[members[-1]]["end_ms"],
            "n_ph": sum(spans[m]["n_ph"] for m in members),
            "why": " ".join(spans[m]["word"] for m in members),
        })

    main = [sp for i, sp in enumerate(spans) if i not in consumed]
    return main, zones, zone_of


def assign_word_owner(times_ms, spans, zones, main_map):
    """音素时间点 -> 归属编码。

    返回值 list，每项：
      "zone:K"  → 落在第 K 个连读区（切不开，按整体看）
      "word:W"  → 落在某个正常词（W 是词名）
      None      → 没归属（降级用）

    判定顺序很重要：**先正常词、后连读区**。
    如果倒过来（连读区优先），连读区边界上的音素会把相邻正常词的音也吸进来
    —— 实测发生过：so 的音 oʊ 被算进了 Where are 连读区。
    """
    out = []
    for t in times_ms:
        t = int(t)
        hit = None
        # 1) 先看是不是落在某个正常词的精确区间里（含一个帧的容差）
        for sp in spans:
            if (sp["start_ms"] - FRAME_MS) <= t <= (sp["end_ms"] + FRAME_MS):
                hit = "word:%s" % sp["word"]
                break
        # 2) 正常词都没命中，才考虑连读区（不加容差，避免边界溢出）
        if hit is None:
            for z in zones:
                if z["start_ms"] <= t <= z["end_ms"]:
                    hit = "zone:%d" % z["zone_idx"]
                    break
        if hit is None:
            for sp in spans:
                if (sp["start_ms"] - FRAME_MS) <= t <= (sp["end_ms"] + FRAME_MS):
                    hit = "word:%s" % sp["word"]
                    break
        if hit is None:
            # 兜底：取最近的主区间（与旧逻辑同义，保证任何音素都有归属）
            best, best_d = None, None
            for sp in spans:
                d = min(abs(t - sp["start_ms"]), abs(t - sp["end_ms"]))
                if best_d is None or d < best_d:
                    best, best_d = sp, d
            if best is not None:
                hit = "word:%s" % best["word"]
        out.append(hit)
    return out


def map_words_to_audio(text_ph, audio_ph, word_bounds):
    """按「参考音素序列」与「实际音素序列」的对齐关系，给每个实际音素定词归属。

    返回：list，长度 = len(audio_ph)，每项是 word_bounds 的下标（int）。

    ⚠️ 这是替掉 assign_word_owner 的关键改动。旧实现拿音素**时间点**去撞词区间，
    跨词边界时会把音归错词 —— 实测：`still at the` 里 at 的 /t/ 在 /ð/ 前省掉了，
    却被算到 the 头上，报告写成「[缺音] the：少了 t」（正确答案是 at）。
    改成按**音素序号**对齐 —— 正是旧文档 §7.3 那句
    「比较单位必须是音素序号 / 句子内相对位置，绝不能拿绝对时间戳直接相减」。
    """
    if not audio_ph:
        return []
    if not text_ph or not word_bounds:
        return [0] * len(audio_ph)

    pos2word = {}
    for wi, wb in enumerate(word_bounds):
        for i in range(wb["a"], wb["b"]):
            pos2word[i] = wi

    owner = [None] * len(audio_ph)
    sm = difflib.SequenceMatcher(a=text_ph, b=audio_ph)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(min(i2 - i1, j2 - j1)):
                owner[j1 + k] = pos2word.get(i1 + k)
        elif op == "replace":
            n, m = i2 - i1, j2 - j1
            for k in range(m):
                i = i1 + (k * n // m) if m else i1
                owner[j1 + k] = pos2word.get(min(i, i2 - 1))
        # delete：文本有、实际没有 → 没有实际音素要归属
        # insert：实际多出来的音 → 交给下面的"就近继承"

    # 未定的（多读出来的音）：继承前一个词；开头就多读的，取后面第一个
    last = None
    for j in range(len(owner)):
        if owner[j] is None:
            owner[j] = last
        else:
            last = owner[j]
    nxt = None
    for j in range(len(owner) - 1, -1, -1):
        if owner[j] is None:
            owner[j] = nxt
        else:
            nxt = owner[j]
    return [0 if o is None else o for o in owner]


def collapse_owner(codes):
    """把归属编码压成 (连读区名列表, 单词名列表)，保持首次出现顺序。"""
    zones_seen, words_seen = [], []
    for c in codes:
        if not c:
            continue
        kind, _, name = c.partition(":")
        if kind == "zone":
            z = int(name)
            if z not in zones_seen:
                zones_seen.append(z)
        elif kind == "word":
            if name not in words_seen:
                words_seen.append(name)
    return zones_seen, words_seen


def compute_word_errors(word_bounds, ref_ph, ref_owner, my_ph, my_owner):
    """逐词比较「原音频实际音素」与「模仿实际音素」，标出读错的词。

    ★ 基准（09-23 定）：**以原音频实际音素为准**，参考文本只用来定词边界。
      理由：判据是「还原度」（像不像原音频），不是「达标度」（像不像字典标准音）。
      连读吞音这类 —— 原音频也吞、学生也吞 → **不算错**。
      反例：若拿参考文本 `want to` 当基准，学生把 want to 正确连读成 wanna
      会被误报成"缺音"，那就把对的判成错的了。

    返回：[{词序, 词, 角色, 类型, 期望音素, 实读音素, 错误音数, 错误比例}]
    """
    kind_cn = {"replace": "替换", "delete": "缺失", "insert": "多余"}
    out = []
    for wi, wb in enumerate(word_bounds):
        ref_seg = [ref_ph[j] for j in range(len(ref_ph)) if ref_owner[j] == wi]
        my_seg = [my_ph[j] for j in range(len(my_ph)) if my_owner[j] == wi]
        if not ref_seg and not my_seg:
            continue
        sm = difflib.SequenceMatcher(a=ref_seg, b=my_seg)
        wrong, kinds = 0, []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            wrong += max(i2 - i1, j2 - j1)
            kinds.append(kind_cn.get(op, op))
        if not wrong:
            continue
        ratio = wrong / float(max(len(ref_seg), len(my_seg), 1))
        if wrong >= WORD_ERR_MIN or ratio >= WORD_ERR_RATIO:
            out.append({
                "词序": wi + 1,
                "词": wb["word"],
                "角色": wb.get("角色"),
                "类型": "+".join(sorted(set(kinds))),
                "期望音素": " ".join(ref_seg),
                "实读音素": " ".join(my_seg),
                "错误音数": wrong,
                "错误比例": round(ratio, 3),
            })
    return out


def format_words(word_ids, words, zone_names=None):
    """生成词标签，附在每条差异后面。

    注意（这是一个隐式契约，不能破）：
      overall_score.py 会反向解析本函数产出的字符串 —— 它把【】里的内容按双引号
      切开、取奇数位当"问题词"。所以：
        - 双引号必须成对
        - 被双引号包住的必须是"单个词"，不能是 "Where are" 这种拼接串
          （拼成一串的话，那边拿 "Where are" 去跟单个词 "Where" 比，一个都匹配不上）
      连读区的两个词因此各自单独加引号：`"Where" "are"`，抠词得 ["Where","are"]。
    """
    if zone_names:
        # 连读区：连读本来切不开，直接告诉学生"这里是一个连读"。
        # 每个词单独加引号（而不是拼成一串放进一个引号里）—— 这样上游按
        # split('"')[1::2] 抠词时，能分别拿到 "Where"、"are"，原句里两个词都能框出。
        all_words = []
        for nm in zone_names:
            all_words.extend(nm.split(" "))
        names = '" "'.join(all_words)
        return "【连读区 \"%s\"】⚠ 这里是连读，按整体的读法看，不必拆开" % names
    if not words or not word_ids:
        return ""
    if len(word_ids) == 1:
        return "【单词 \"%s\"】" % words[word_ids[0]]["word"]
    return "【单词 \"%s\" → \"%s\" 之间】" % (words[word_ids[0]]["word"],
                                            words[word_ids[-1]]["word"])


def dwidth(s):
    """按东亚宽度估算显示宽度：宽/全角算 2，其余算 1（配合等宽字体）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad_to(s, width):
    """右侧补空格到指定显示宽度。"""
    return s + " " * max(0, width - dwidth(s))


def center_to(s, width):
    """在指定显示宽度里居中。"""
    total = max(0, width - dwidth(s))
    left = total // 2
    return " " * left + s + " " * (total - left)


def build_aligned(ref, mine, sm):
    """按对齐关系把两条音素序列展开成 (原音频音, 你的音) 列表；缺/多处为空串。"""
    pairs = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        r, m = ref[i1:i2], mine[j1:j2]
        for k in range(max(len(r), len(m))):
            pairs.append((r[k] if k < len(r) else "", m[k] if k < len(m) else ""))
    return pairs


def write_report(ref, mine, sm, words, ratio, diff_items, problem_words):
    """写一份给人读的纯文本报告（json 留给机器/AI）。"""
    L = []
    L.append("=" * 60)
    L.append("   音素对比报告（发音 + 音变）")
    L.append("=" * 60)
    L.append("")
    L.append("原音频.wav  vs  模仿.wav")
    L.append("整体音素相似度：%.1f / 100" % ratio)
    L.append("")
    # 一、逐音对照：横排，上行=原音频，下行=我的，★=不一样
    L.append("-" * 60)
    L.append("一、逐音对照   （上=原音频 应读，下=我的 实读，★=不一样）")
    L.append("-" * 60)
    L.append("")
    pairs = [(r if r else "—", m if m else "—") for r, m in build_aligned(ref, mine, sm)]
    if pairs:
        cell = max(3, max(dwidth(p) for pr in pairs for p in pr)) + 2
        la, lm = "原音频:", "我的:"
        pw = max(dwidth(la), dwidth(lm)) + 1
        step = 12                      # 每行最多放多少个音，放不下就换块
        for s0 in range(0, len(pairs), step):
            chunk = pairs[s0:s0 + step]
            L.append((pad_to(la, pw) + "".join(center_to(r, cell) for r, m in chunk)).rstrip())
            L.append((pad_to(lm, pw) + "".join(center_to(m, cell) for r, m in chunk)).rstrip())
            L.append((" " * pw + "".join(center_to("★", cell) if r != m else " " * cell
                                         for r, m in chunk)).rstrip())
            L.append("")
    # 二、问题清单
    L.append("-" * 60)
    L.append("二、问题清单")
    L.append("-" * 60)
    if diff_items:
        for it in diff_items:
            if it.get("zone_names"):
                who = "、".join(it["zone_names"])          # 连读区：直接报词组
            elif it["words"]:
                who = "、".join(it["words"])
            else:
                who = "（未定位到词）"
            if it["kind"] == "替换":
                L.append("    [替换] %s：应读 %s  →  实读 %s" % (who, it["ref"], it["my"]))
            elif it["kind"] == "缺失":
                L.append("    [缺音] %s：少了 %s" % (who, it["ref"]))
            else:
                L.append("    [多音] %s：多了 %s" % (who, it["my"]))
        L.append("")
        L.append("    ※ [缺音]/[多音] 多出现在连读、弱化、吞音处——这就是“音变”")
        if any(it.get("zone") for it in diff_items):
            L.append("    ※ 标了「连读区」的地方，是两个词黏在一起读（如 Where are），")
            L.append("      连读本来就是一整团、切不开，按整体的读法练就好，不必拆单个音。")
    else:
        L.append("    （音素序列完全一致，很棒）")
    # 原句（问题词用【】框出）
    if words:
        bad = set(w for w, _ in problem_words)
        sent = " ".join(("【%s】" % w["word"]) if w["word"] in bad else w["word"] for w in words)
        L.append("")
        L.append("-" * 60)
        L.append("原句（有问题的词已用【】框出）")
        L.append("-" * 60)
        L.append("    " + sent)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    # 先把耗时大户 import 掉，再开日志 —— 否则 log_setup 的时间戳会比真正开始早十几秒
    import torch
    from transformers import AutoModelForCTC
    log_setup("音素对比")

    out("=" * 62)
    out("   还原度对比 · 音素")
    out("=" * 62)

    if not (os.path.exists(REF_FILE) and os.path.exists(MY_FILE)):
        out("")
        out("X 没找到音频文件。请把两个 wav 放进本文件夹：")
        out("    " + REF_FILE)
        out("    " + MY_FILE)
        log_done("（音频缺失，提前退出）")
        return

    log_step("加载音素识别模型")
    model_path = ensure_model()
    fe = load_feature_extractor(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)
    model.eval()
    id2tok = load_vocab(model_path)
    log("模型就绪（%s）" % os.path.basename(model_path), verbose=True)

    results = {}
    frames_map = {}
    for tag, path in (("原音频", REF_FILE), ("我的模仿", MY_FILE)):
        log_step("识别：%s" % tag)
        audio = load_mono(path)
        log("   音频时长 %.2f 秒（%d 采样点）" % (len(audio) / float(SR), len(audio)),
            verbose=True)
        inputs = fe(audio, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        log("   模型输出 %d 帧（每帧 %dms）" % (logits.shape[1], FRAME_MS), verbose=True)
        ids = torch.argmax(logits, dim=-1)[0].tolist()
        tokens, frames = ctc_decode(ids, id2tok)
        results[tag] = tokens
        frames_map[tag] = frames
        log("   识别出的音素共 %d 个" % len(tokens))

    ref = results["原音频"]
    mine = results["我的模仿"]
    log("原音频 %d 音 vs 我的 %d 音" % (len(ref), len(mine)), verbose=True)

    # 读「参考音素表」（由 gen_ref_phonemes.py 离线预生成）—— 不再调讯飞。
    # 词边界来自参考文本；音素归属按「序号对齐」定（见 map_words_to_audio）
    log_step("读取参考音素表")
    ref_data = load_ref_phonemes()
    text_ph = ref_data["音素"] if ref_data else []
    word_bounds = ref_data["词边界"] if ref_data else []
    if ref_data:
        log("   参考音素 %d 个，词边界 %d 个（角色 %s）"
            % (len(text_ph), len(word_bounds),
               "".join(sorted({wb.get("角色") or "?" for wb in word_bounds}))))
    else:
        log("   ⚠ 参考音素.json 不可用 → 本次只出音素级结果，不标词", show=True)

    # 音素时间轴：帧号 × FRAME_MS
    ref_frame_times = [f * FRAME_MS for f in frames_map["原音频"]]
    my_frame_times = [f * FRAME_MS for f in frames_map["我的模仿"]]

    # 音素归属：按「参考音素序列 ↔ 实际音素序列」的**对齐关系**定，
    # 不再按时间点撞词区间（那正是"跨词归错词"的根因）
    ref_owner = map_words_to_audio(text_ph, ref, word_bounds)
    my_owner = map_words_to_audio(text_ph, mine, word_bounds)

    # 两侧词块：节奏脚本的唯一计时输入，也是连读区检测的入口
    log_step("连读区识别")
    spans = build_text_spans(word_bounds, ref_owner, ref_frame_times)
    spans_my = build_text_spans(word_bounds, my_owner, my_frame_times)
    main_spans, zones, zone_of = pair_fragments(spans)          # ★ 逐字不改
    _main_my, zones_my, _zone_my = pair_fragments(spans_my)
    log("   词区间 %d 段；碎片词 %d 个；连读区 %d 处"
        % (len(spans), sum(1 for s in spans if s["short"]), len(zones)), verbose=True)
    for z in zones:
        log("   连读区：%s（%d-%dms）" % (" ".join(z["words"]), z["start_ms"], z["end_ms"]))

    # 词下标 -> 连读区下标（差异分析里判断"这个词在连读区里吗"）
    zone_by_word = {}
    for z in zones:
        for m in z["member_idx"]:
            zone_by_word[m] = z["zone_idx"]
    my_zone_words = set()
    for z in zones_my:
        for m in z["member_idx"]:
            my_zone_words.add(m)

    log_step("对齐与差异分析")
    sm = difflib.SequenceMatcher(a=ref, b=mine)
    ratio = sm.ratio() * 100

    # 对齐统计：把音素序列的对齐关系数成"几个替换 / 几个缺失 / 几个多余"。
    # 综合评分器据此直接算「发音分（替换）」和「音变分（缺/多）」，
    # 不必再重跑 wav2vec2 —— 数据在这一步就已算好。
    align_stat = {"原生音数": len(ref), "模仿音数": len(mine),
                  "匹配音数": 0, "替换音数": 0, "缺失音数": 0, "多余音数": 0}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            align_stat["匹配音数"] += (i2 - i1)
        elif op == "replace":
            align_stat["替换音数"] += max(i2 - i1, j2 - j1)
        elif op == "delete":
            align_stat["缺失音数"] += (i2 - i1)
        else:  # insert
            align_stat["多余音数"] += (j2 - j1)

    def resolve_by_words(ph_idx_list):
        """把一组「原音频音素下标」解析成 (连读区名列表, 词下标列表, 词名列表)。

        规则：只要落到的词里有在连读区里的，就整体按连读区呈现 ——
        连读本来就切不开，硬拆反而误导（与旧 resolve 同旨）。
        """
        wis = sorted({ref_owner[j] for j in ph_idx_list if 0 <= j < len(ref_owner)})
        if not wis:
            return [], [], []
        zone_ids = sorted({zone_by_word[wi] for wi in wis if wi in zone_by_word})
        if zone_ids:
            return [" ".join(zones[z]["words"]) for z in zone_ids], [], []
        return [], wis, [word_bounds[wi]["word"] for wi in wis]

    diff_lines = []
    problem_words = []      # (词, 类型) 汇总
    diff_items = []         # 给人类报告用的结构化差异
    zone_hits = {}          # 连读区名 -> {"词": [...], "n": 次数}，写进 json 供后期复用
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            line = "  [替换] 应该读 %s  →  你读成了 %s" % (" ".join(ref[i1:i2]), " ".join(mine[j1:j2]))
            kind = "替换"
            zn, idxs, names = resolve_by_words(list(range(i1, i2)))
        elif op == "delete":
            line = "  [缺失] 你没读 %s（原音频里有）" % " ".join(ref[i1:i2])
            kind = "缺失"
            zn, idxs, names = resolve_by_words(list(range(i1, i2)))
        else:  # insert
            # 多读出来的音在原音频里没有对应位置 → 就近挂到它前面那个音所属的词
            line = "  [多余] 你多读了 %s（原音频里没有）" % " ".join(mine[j1:j2])
            kind = "多余"
            zn, idxs, names = resolve_by_words([max(0, i1 - 1)])
        label = format_words(idxs, word_bounds, zone_names=zn if zn else None)
        if label:
            line = line + "   " + label
            for i in idxs:
                problem_words.append((word_bounds[i]["word"], kind))
            for nm in zn:
                zone_hits.setdefault(nm, {"词": nm.split(" "), "n": 0})
                zone_hits[nm]["n"] += 1
                # 连读区涉及的两个词都算"有问题"，原句里一并框出（并集框）
                for wname in nm.split(" "):
                    problem_words.append((wname, kind))
        diff_items.append({"kind": kind,
                           "ref": " ".join(ref[i1:i2]),
                           "my": " ".join(mine[j1:j2]),
                           "words": names,
                           "zone": bool(zn),
                           "zone_names": zn})
        diff_lines.append(line)

    out("")
    out("=" * 62)
    out("   音素对比结果")
    out("=" * 62)
    out("整体音素相似度：%.1f / 100" % ratio)
    if diff_lines:
        out("差异 %d 处（明细见报告）" % len(diff_lines))
    else:
        out("音素序列完全一致。")
    log("对齐统计：匹配 %d ｜ 替换 %d ｜ 缺失 %d ｜ 多余 %d"
        % (align_stat["匹配音数"], align_stat["替换音数"],
           align_stat["缺失音数"], align_stat["多余音数"]), verbose=True)

    # 写一份"给人看"的纯文本报告（json 留给机器/AI）
    log_step("写报告与数据")
    try:
        write_report(ref, mine, sm, word_bounds, ratio, diff_items, problem_words)
        out("")
        out("报告（人看）：音素对比_报告.txt")
        out("数据（机器）：音素对比.json")
    except Exception as e:
        log("（写报告失败：%s）" % e, show=True)

    # 词切分：连读区 / 碎片词 / 每个词分到几个音素。
    word_split = {}
    if word_bounds:
        word_split = {
            "最小词长ms": MIN_WORD_MS,
            "碎片词": [{"word": sp["word"], "start_ms": sp["start_ms"],
                        "end_ms": sp["end_ms"], "span_ms": sp["span_ms"],
                        "n_ph": sp["n_ph"],
                        "原因": "时长不足 %dms" % MIN_WORD_MS}
                       for sp in spans if sp["short"]],
            "连读区": [{"词": z["words"], "start_ms": z["start_ms"], "end_ms": z["end_ms"],
                        "n_ph": z["n_ph"],
                        "归词音素数": [[spans[m]["word"], spans[m]["n_ph"]] for m in z["member_idx"]],
                        "说明": "%s 处连读，按整体看" % " ".join(z["words"])}
                       for z in zones],
            "命中差异的连读区": list(zone_hits.values()),
            "词音素分布": [{"word": sp["word"], "n_ph": sp["n_ph"],
                            "span_ms": sp["span_ms"]} for sp in spans],
        }

    # 单词错误（标注层，不进综合分。基准 = 原音频实际音素，见 compute_word_errors）
    word_errors = compute_word_errors(word_bounds, ref, ref_owner, mine, my_owner) if word_bounds else []

    # 词块：节奏脚本的唯一计时输入（两侧都要）
    word_blocks = {
        "原音频": [{"词": sp["word"], "start_ms": sp["start_ms"], "end_ms": sp["end_ms"],
                    "连读": sp["idx"] in zone_by_word,
                    "角色": word_bounds[sp["idx"]].get("角色") if sp["idx"] < len(word_bounds) else None}
                   for sp in spans],
        "我的": [{"词": sp["word"], "start_ms": sp["start_ms"], "end_ms": sp["end_ms"],
                  "连读": sp["idx"] in my_zone_words,
                  "角色": word_bounds[sp["idx"]].get("角色") if sp["idx"] < len(word_bounds) else None}
                 for sp in spans_my],
    }

    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"原音频": ref, "我的模仿": mine,
                       "相似度": ratio, "对齐统计": align_stat,
                       "差异": diff_lines,
                       # 词表去掉 start_ms/end_ms —— 时间上移到「词块」
                       "词表": [{"word": wb["word"], "角色": wb.get("角色")} for wb in word_bounds],
                       "词切分": word_split,
                       "词块": word_blocks,
                       "单词错误": word_errors or None,
                       "参考音素可用": bool(ref_data)}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    log_done("音素对比")


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
