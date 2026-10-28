# -*- coding: utf-8 -*-
r"""原音频分析 · 一次性跑完原音频的所有分析，产出缓存文件

为什么单独这个脚本（10-28 加）：
  原音频是固定范本，它的转写文本、参考音素表、CTC 识别音素、节奏/重音特征
  全都不会变。但 compare_long 每次都要对原音频重新跑一遍这些（尤其 1.2GB
  模型的 CTC 推理 + RMS/音高计算），白花时间。本脚本把原音频的完整分析
  结果一次性算好、存成缓存，compare_long 之后直接读缓存，只分析模仿音频。

产出：audio\原音频\原音频缓存.json，包含：
  转写文本、参考音素表（音素/词边界/句子边界）、原音频 CTC 音素序列 + 帧、
  原音频 RMS、原音频音高。

用法：python scripts\analyze_reference.py
      （被 audio\原音频\分析原音频.bat 调用）
"""
import json
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(ROOT, "audio")
REF_DIR = os.path.join(SRC_DIR, "原音频")          # 原音频放这里
REF_PHONEME_JSON = os.path.join(SRC_DIR, "参考音素.json")
TRANSCRIBE_TXT = os.path.join(SRC_DIR, "转写文本.txt")
CACHE_JSON = os.path.join(REF_DIR, "原音频缓存.json")
XUNFEI_CONFIG = os.path.join(ROOT, "讯飞配置.json")

sys.path.insert(0, HERE)
from logkit import log, out, log_setup, log_done, log_step, log_error
from audio_paths import find_audio, short_name
import compare_phoneme as cp
import compare_rhythm as cr


def find_reference():
    """定位原音频：优先 audio\原音频\原音频.*，回退 audio\原音频.*。"""
    p = find_audio("原音频", REF_DIR)
    if p:
        return p
    return find_audio("原音频", SRC_DIR)


def run_script(name, args=()):
    cmd = [sys.executable, os.path.join(HERE, name)] + list(args)
    r = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL)
    return r.returncode


def clean_transcript(text):
    """转写文本轻量清理（与 run_all.clean_transcript 同规则）：
    修 ASR 粘连（andWhat → and What）、压多余空白。"""
    import re
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)     # 粘连拆开
    text = re.sub(r"\s+", " ", text).strip()             # 压空白
    return text


def main():
    log_setup("原音频分析")
    out("=" * 62)
    out("   原音频分析 · 一次性跑完并存缓存")
    out("=" * 62)

    ref_path = find_reference()
    if not ref_path:
        out("X 没找到原音频，请放到 audio\\原音频\\ 目录：")
        out("      audio\\原音频\\原音频.mp3（或 wav / flac / ogg）")
        log_done("（缺原音频，提前退出）")
        return
    out("原音频：%s" % short_name(ref_path))

    # ---- 1. 讯飞转写（拿台词文本当真值）----
    log_step("讯飞转写原音频")
    if not os.path.exists(XUNFEI_CONFIG):
        out("X 没有 讯飞配置.json，无法转写。请放好再跑。")
        log_done("（缺讯飞配置，提前退出）")
        return
    if run_script("xunfei_transcribe.py", [ref_path]) != 0:
        out("X 没识别到文本（转写失败或音频无人声）。")
        log_done("（转写失败，提前退出）")
        return
    text = None
    if os.path.exists(TRANSCRIBE_TXT):
        with open(TRANSCRIBE_TXT, encoding="utf-8") as f:
            text = f.read().strip()
    if not text:
        out("X 没识别到文本（转写结果为空）。")
        log_done("（转写为空，提前退出）")
        return
    text = clean_transcript(text)
    log("   台词 %d 字符" % len(text))

    # ---- 2. 生成参考音素表 ----
    log_step("生成参考音素表（音素/词边界/句子边界）")
    if run_script("gen_ref_phonemes.py", [text]) != 0:
        out("X 生成参考音素表失败（espeak-ng 是否安装？）")
        log_done("（生成失败，提前退出）")
        return
    with open(REF_PHONEME_JSON, encoding="utf-8") as f:
        ref_data = json.load(f)
    log("   参考 %d 音素 / %d 词 / %d 句"
        % (len(ref_data.get("音素") or []), len(ref_data.get("词边界") or []),
           len(ref_data.get("句子边界") or [])))

    # ---- 3. CTC 识别原音频 ----
    log_step("CTC 识别原音频（跑音素模型）")
    import torch
    model_path = cp.ensure_model()
    fe = cp.load_feature_extractor(model_path)
    from transformers import AutoModelForCTC
    model = AutoModelForCTC.from_pretrained(model_path)
    model.eval()
    id2tok = cp.load_vocab(model_path)
    audio = cp.load_mono(ref_path)
    inputs = fe(audio, sampling_rate=cp.SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    ids = torch.argmax(logits, dim=-1)[0].tolist()
    ref_toks, ref_frames = cp.ctc_decode(ids, id2tok)
    log("   原音频 %d 音素" % len(ref_toks))

    # ---- 4. 节奏/重音特征（RMS + 音高）----
    log_step("算节奏/重音特征（RMS + 音高）")
    env = cr.frame_rms(cp.load_mono(ref_path))
    f0 = cr.pitch_contour(cp.load_mono(ref_path))
    log("   RMS %d 帧 / 音高 %d 帧" % (len(env), len(f0)))

    # ---- 5. 打包缓存 ----
    log_step("写缓存")
    cache = {
        "转写文本": text,
        "参考音素": ref_data,
        "原音频_CTC音素": ref_toks,
        "原音频_帧": ref_frames,
        "原音频_RMS": [float(x) for x in env],
        "原音频_音高": [float(x) for x in f0],
        "原音频路径": ref_path,
        "生成时间": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(REF_DIR, exist_ok=True)
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    out("")
    out("原音频分析完成，缓存已存：")
    out("    " + CACHE_JSON)
    out("以后一键跑还原度会直接读这份缓存，不再重跑原音频分析。")
    log_done("原音频分析")


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
