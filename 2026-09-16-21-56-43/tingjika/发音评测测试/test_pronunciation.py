# -*- coding: utf-8 -*-
"""发音评测 · 最小验证脚本

目的：只用一句英文，确认你的 Azure 账号能不能做「发音评测」。
     能的话，会打印出逐音素的分数（这就是后面标红"哪个音不准"的依据）。

用法：双击 运行测试.bat
      程序会显示一句标准英文，你跟着读一遍，它会打分。

看清三件事：
  1. 整句得分 PronScore
  2. 每个词的准确度 AccuracyScore
  3. 每个音素（phoneme）的准确度 —— 低分的音就是"读不准的音"
"""
import json
import os
import sys
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "测试日志.txt")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def log(msg=""):
    text = str(msg)
    try:
        print(text, flush=True)
    except Exception:
        pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


import azure.cognitiveservices.speech as speechsdk

# ========== 1. 密钥 ==========
# 三种填法，任选一种（优先级从高到低）：
#   1) 设环境变量 AZURE_SPEECH_KEY / AZURE_SPEECH_REGION
#   2) 在「发音评测测试」文件夹里建 keys.txt，写两行：
#          key=你的密钥
#          region=eastasia
#   3) 直接把下面两行的引号里填上（不推荐，容易误传给别人）
SPEECH_KEY = "在这里填你的KEY"
SPEECH_REGION = "eastasia"


def _load_keys():
    """从环境变量或 keys.txt 覆盖上面的默认值。"""
    global SPEECH_KEY, SPEECH_REGION
    env_key = os.environ.get("AZURE_SPEECH_KEY")
    env_reg = os.environ.get("AZURE_SPEECH_REGION")
    if env_key:
        SPEECH_KEY = env_key
    if env_reg:
        SPEECH_REGION = env_reg
    kf = os.path.join(HERE, "keys.txt")
    if os.path.exists(kf):
        with open(kf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip().lower(), v.strip()
                if k == "key" and v:
                    SPEECH_KEY = v
                elif k == "region" and v:
                    SPEECH_REGION = v


# ========== 标准句（参考文本）：这就是"标准答案" ==========
REFERENCE_TEXT = "I want to improve my English pronunciation."

# 低于这个分数算"不准"（0-100）
THRESHOLD = 60


def main():
    log("=" * 56)
    log("   发音评测 · 最小验证    " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("=" * 56)

    _load_keys()
    if "在这里填" in SPEECH_KEY or not SPEECH_KEY:
        log("")
        log("X 还没有填 Azure 密钥。三选一：")
        log("   1) 设环境变量 AZURE_SPEECH_KEY / AZURE_SPEECH_REGION")
        log("   2) 在本文件夹建 keys.txt，内容两行：")
        log("          key=你的密钥")
        log("          region=eastasia")
        log("   3) 打开本脚本，改 SPEECH_KEY 那一行")
        log("")
        return

    log("区域: " + SPEECH_REGION)

    log("")
    log("[1] 构造语音配置（英文）...")
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"   # 关键：英文
    log("    OK")

    log("[2] 打开麦克风 ...")
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    log("[3] 创建识别器 + 挂上发音评测配置 ...")
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                            audio_config=audio_config)

    pron_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=REFERENCE_TEXT,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True)          # 顺带检测"读错词/漏读"
    pron_config.phoneme_alphabet = "IPA"     # 音标用国际音标，好认
    pron_config.apply_to(recognizer)
    log("    OK")

    log("")
    log("-" * 56)
    log("请跟读这句话（读完自动结束）：")
    log("")
    log("    " + REFERENCE_TEXT)
    log("")
    log("-" * 56)

    result = recognizer.recognize_once()   # 说一句、评一句

    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        log("")
        log("X 没识别到语音，原因: " + str(result.reason))
        if result.reason == speechsdk.ResultReason.NoMatch:
            log("  => 没听到声音，或太安静/太快。靠近麦克风再试。")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cd = result.cancellation_details
            log("  取消原因: " + str(cd.reason))
            log("  错误详情: " + str(cd.error_details))
            log("  => 如果出现 401/403，多半是密钥或区域不符；")
            log("     如果提到 Pronunciation，说明该资源没开发音评测。")
        return

    # ---- 拿到 JSON 结果 ----
    json_str = result.properties.get(
        speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
    data = json.loads(json_str)
    best = data["NBest"][0]
    pron = best["PronunciationAssessment"]

    log("")
    log("=" * 56)
    log("   评测结果")
    log("=" * 56)
    log("识别到的文本: " + best.get("Display", result.text))
    log("")
    log("整句得分:")
    log("    准确度 AccuracyScore   : %.1f" % pron.get("AccuracyScore", 0))
    log("    流利度 FluencyScore    : %.1f" % pron.get("FluencyScore", 0))
    log("    完整度 CompletenessScore: %.1f" % pron.get("CompletenessScore", 0))
    log("    总分 PronScore         : %.1f" % pron.get("PronScore", 0))

    log("")
    log("-" * 56)
    log("逐词 / 逐音素明细（< %d 分标 [?] 表示不准）:" % THRESHOLD)
    log("-" * 56)

    if "Words" not in best:
        log("  ⚠ 返回里没有 Words 字段 —— 说明这个资源没返回音素级结果。")
        log("    完整 JSON 已存到 测试结果.json，可发给我看。")
    else:
        for w in best["Words"]:
            w_score = w["PronunciationAssessment"].get("AccuracyScore", 0)
            err = w["PronunciationAssessment"].get("ErrorType", "")
            flag = " [x]" if w_score < THRESHOLD or err == "Mispronunciation" else ""
            log("")
            log("  词: %-16s  %.1f 分%s %s" % (w.get("Word"), w_score, flag,
                                             ("(" + err + ")") if err else ""))
            for p in w.get("Phonemes", []):
                p_score = p["PronunciationAssessment"].get("AccuracyScore", 0)
                pf = " <<< 不准" if p_score < THRESHOLD else ""
                log("      /%s/  %.1f 分%s" % (p.get("Phoneme"), p_score, pf))

    # 存完整 JSON，方便排查
    try:
        with open(os.path.join(HERE, "测试结果.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log("")
        log("（完整结果已存到 测试结果.json）")
    except Exception:
        pass

    log("")
    log("=" * 56)
    log("结论：如果你上面看到了音素分数，说明发音评测可用 ✅")
    log("=" * 56)


if __name__ == "__main__":
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass
    try:
        main()
    except Exception:
        log("[致命错误]")
        log(traceback.format_exc())
