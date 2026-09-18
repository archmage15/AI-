# -*- coding: utf-8 -*-
"""还原度对比 · 第二块：音素

把「原音频」和「你的模仿」分别喂给音素识别模型，得到两条音素序列，
对齐后告诉你：哪些音没还原、哪里多了/少了音（连读、弱化、吞音就看这里）。

模型：facebook/wav2vec2-lv-60-espeak-cv-ft
  - 首次运行会自动从 HuggingFace 下载（约 1.2GB），之后走本地缓存，不再下
  - 输出 eSpeak 风格音素（接近 IPA）

用法：
  1. 把两个音频放进本文件夹，命名成：原音频.wav 和 模仿.wav
  2. 双击 运行音素对比.bat
  3. 屏幕出对比结果，并生成 音素对比.json

依赖：torch / transformers / soundfile / numpy
      （解码直接用模型自带 vocab.json，不需要额外装 phonemizer）
"""
import difflib
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 国内访问 HuggingFace：官方 huggingface.co 在本机连不上，自动走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

HERE = os.path.dirname(os.path.abspath(__file__))
REF_FILE = os.path.join(HERE, "原音频.wav")
MY_FILE = os.path.join(HERE, "模仿.wav")
OUT_JSON = os.path.join(HERE, "音素对比.json")

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
# 模型存放目录（用 local_dir 直接复制，避开默认缓存的符号链接问题）
LOCAL_DIR = r"C:\Users\24048\.workbuddy\models\wav2vec2-espeak"
SR = 16000


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
    """CTC 解码：合并重复 -> 去空白/特殊符 -> 映射成音素"""
    out = []
    prev = None
    for i in ids:
        i = int(i)
        if i == prev:
            continue
        prev = i
        if i == blank or i in specials:
            continue
        tok = id2tok.get(i, "")
        if tok:
            out.append(tok)
    return out


def load_feature_extractor(model_path):
    try:
        from transformers import AutoFeatureExtractor
        return AutoFeatureExtractor.from_pretrained(model_path)
    except Exception:
        from transformers import Wav2Vec2FeatureExtractor
        return Wav2Vec2FeatureExtractor.from_pretrained(model_path)


def ensure_model():
    """模型不存在则先下载（走 local_dir 模式）。返回本地模型目录。"""
    if os.path.exists(os.path.join(LOCAL_DIR, "config.json")):
        return LOCAL_DIR
    log("")
    log("首次运行：下载音素模型（约 1.2GB，走 hf-mirror 镜像，请耐心等）...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)
    log("模型下载完成。")
    return LOCAL_DIR


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
    for tag, path in (("原音频", REF_FILE), ("我的模仿", MY_FILE)):
        log("正在识别：%s ..." % tag)
        audio = load_mono(path)
        inputs = fe(audio, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)[0].tolist()
        tokens = ctc_decode(ids, id2tok)
        results[tag] = tokens
        log("   识别出的音素共 %d 个" % len(tokens))

    ref = results["原音频"]
    mine = results["我的模仿"]

    log("")
    log("=" * 62)
    log("   音素序列对比（原音频 -> 你的模仿）")
    log("=" * 62)
    log("")
    log("原音频：" + " ".join(ref))
    log("我的  ：" + " ".join(mine))
    log("")

    sm = difflib.SequenceMatcher(a=ref, b=mine)
    ratio = sm.ratio() * 100
    log("整体音素相似度：%.1f" % ratio)
    log("")
    log("逐段差异（只看不一致的地方）：")
    log("-" * 62)
    diff_lines = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            line = "  [替换] 应该读 %s  →  你读成了 %s" % (" ".join(ref[i1:i2]), " ".join(mine[j1:j2]))
        elif op == "delete":
            line = "  [缺失] 你没读 %s（原音频里有）" % " ".join(ref[i1:i2])
        else:  # insert
            line = "  [多余] 你多读了 %s（原音频里没有）" % " ".join(mine[j1:j2])
        log(line)
        diff_lines.append(line)
    if not diff_lines:
        log("  （音素序列完全一致）")

    log("")
    log("怎么看：")
    log("  - [缺失]/[多余] 常出现在连读、弱化、吞音处 —— 这就是'音变还原度'")
    log("  - [替换] 就是某个音发错了（如 θ 读成 s）")
    log("  - 注意：模型对'学习者发音'的判断未必 100% 准，先看趋势")

    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"原音频": ref, "我的模仿": mine,
                       "相似度": ratio, "差异": diff_lines}, f, ensure_ascii=False, indent=2)
        log("")
        log("（结果已存到 音素对比.json）")
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
