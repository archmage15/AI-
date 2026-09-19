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


def log(msg=""):
    print(str(msg), flush=True)


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


def format_words(word_ids, words):
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
            who = "、".join(it["words"]) if it["words"] else "（未定位到词）"
            if it["kind"] == "替换":
                L.append("    [替换] %s：应读 %s  →  实读 %s" % (who, it["ref"], it["my"]))
            elif it["kind"] == "缺失":
                L.append("    [缺音] %s：少了 %s" % (who, it["ref"]))
            else:
                L.append("    [多音] %s：多了 %s" % (who, it["my"]))
        L.append("")
        L.append("    ※ [缺音]/[多音] 多出现在连读、弱化、吞音处——这就是“音变”")
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
    import torch
    from transformers import AutoModelForCTC

    log("=" * 62)
    log("   还原度对比 · 音素")
    log("=" * 62)

    if not (os.path.exists(REF_FILE) and os.path.exists(MY_FILE)):
        log("")
        log("X 没找到音频文件。请把两个 wav 放进本文件夹：")
        log("    " + REF_FILE)
        log("    " + MY_FILE)
        return

    log("")
    log("加载音素识别模型 ...")
    model_path = ensure_model()
    fe = load_feature_extractor(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)
    model.eval()
    id2tok = load_vocab(model_path)
    log("模型就绪。")

    results = {}
    frames_map = {}
    for tag, path in (("原音频", REF_FILE), ("我的模仿", MY_FILE)):
        log("正在识别：%s ..." % tag)
        audio = load_mono(path)
        inputs = fe(audio, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)[0].tolist()
        tokens, frames = ctc_decode(ids, id2tok)
        results[tag] = tokens
        frames_map[tag] = frames
        log("   识别出的音素共 %d 个" % len(tokens))

    ref = results["原音频"]
    mine = results["我的模仿"]

    # 可选：对原音频跑讯飞转写，拿到"词 + 时间"，给音素打单词标签
    words = []
    log("")
    log("正在获取单词时间轴（讯飞转写）...")
    try:
        words = transcribe_words(REF_FILE)
        log("   拿到 %d 个词：%s" % (len(words), " ".join(w["word"] for w in words)))
    except Exception as e:
        words = []
        log("   （跳过，只给音素级结果）：%s" % e)

    ref_word_idx = assign_word_indices([f * FRAME_MS for f in frames_map["原音频"]], words)

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

    diff_lines = []
    problem_words = []      # (词, 类型) 汇总
    diff_items = []         # 给人类报告用的结构化差异
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            line = "  [替换] 应该读 %s  →  你读成了 %s" % (" ".join(ref[i1:i2]), " ".join(mine[j1:j2]))
            kind, ws = "替换", word_indices(ref_word_idx[i1:i2], words)
        elif op == "delete":
            line = "  [缺失] 你没读 %s（原音频里有）" % " ".join(ref[i1:i2])
            kind, ws = "缺失", word_indices(ref_word_idx[i1:i2], words)
        else:  # insert
            # 你的录音和原音频时间轴不一定对齐，"多读"的音不能靠时间归属；
            # 改用它出现在 diff 里的位置 —— 即相邻的两个原生词之间（不依赖时间）。
            line = "  [多余] 你多读了 %s（原音频里没有）" % " ".join(mine[j1:j2])
            kind, ws = "多余", word_indices(ref_word_idx[max(0, i1 - 1):i1 + 1], words)
        label = format_words(ws, words)
        if label:
            line = line + "   " + label
            for wi in ws:
                problem_words.append((words[wi]["word"], kind))
        diff_items.append({"kind": kind,
                           "ref": " ".join(ref[i1:i2]),
                           "my": " ".join(mine[j1:j2]),
                           "words": [words[wi]["word"] for wi in ws]})
        diff_lines.append(line)

    log("")
    log("=" * 62)
    log("   音素对比结果")
    log("=" * 62)
    log("整体音素相似度：%.1f / 100" % ratio)
    if diff_lines:
        log("差异 %d 处（明细见报告）" % len(diff_lines))
    else:
        log("音素序列完全一致。")

    # 写一份"给人看"的纯文本报告（json 留给机器/AI）
    try:
        write_report(ref, mine, sm, words, ratio, diff_items, problem_words)
        log("")
        log("报告（人看）：音素对比_报告.txt")
        log("数据（机器）：音素对比.json")
    except Exception as e:
        log("（写报告失败：%s）" % e)

    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"原音频": ref, "我的模仿": mine,
                       "相似度": ratio, "对齐统计": align_stat,
                       "差异": diff_lines,
                       "词表": words}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("[致命错误]")
        log(traceback.format_exc())
    try:
        input("按回车退出 ...")
    except Exception:
        pass
