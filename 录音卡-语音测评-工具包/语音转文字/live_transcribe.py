# -*- coding: utf-8 -*-
"""听记卡 · 实时语音转文字（诊断版 v2）

- 边说边出字：[识别中] 会变，[定稿] 落整句
- 每句定稿自动追加到 听记笔记.txt
- 全过程同时写入 runlog.txt，出错也记进去（方便远程排查）

双击 run.bat 运行。
"""
import os
import sys
import time
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE_FILE = os.path.join(HERE, "听记笔记.txt")
LOG_FILE = os.path.join(HERE, "runlog.txt")

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
SPEECH_KEY = "4qad26JxJiWwH7PRKnOq5mSxxNqCOVOTefoa00JqOWeiPYRl1e01JQQJ99CIAC3pKaRXJ3w3AAAYACOGsZpM"
# ========== 2. 区域 ==========
SPEECH_REGION = "eastasia"


def main():
    log("=" * 46)
    log("   听记卡 · 实时语音转文字  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("=" * 46)
    log("Python 路径: " + sys.executable)
    log("区域: " + SPEECH_REGION)

    log("")
    log("[1] 构造语音配置 ...")
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "zh-CN"
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "5000")
    log("    OK")

    log("[2] 打开麦克风 ...")
    try:
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        log("    OK")
    except Exception as e:
        log("    X 打不开麦克风: %s" % e)
        log(traceback.format_exc())
        return

    log("[3] 创建识别器 ...")
    try:
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                                audio_config=audio_config)
        log("    OK")
    except Exception as e:
        log("    X 创建识别器失败: %s" % e)
        log(traceback.format_exc())
        return

    finished = False

    def on_session_started(evt):
        log("[系统] 已连接到 Azure，开始听...")

    def on_recognizing(evt):
        if evt.result.text:
            log("[识别中] " + evt.result.text)

    def on_recognized(evt):
        text = evt.result.text
        if not text:
            return
        log("[定稿  ] " + text)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(NOTE_FILE, "a", encoding="utf-8") as f:
                f.write("[{}] {}\n".format(ts, text))
            log("        （已写入 听记笔记.txt）")
        except Exception as e:
            log("        写笔记失败: %s" % e)

    def on_canceled(evt):
        log("[错误] 识别被取消，原因: %s" % evt.reason)
        if evt.reason == speechsdk.CancellationReason.Error:
            log("      错误码: %s" % evt.error_code)
            log("      详情: %s" % evt.error_details)
        nonlocal finished
        finished = True

    def on_session_stopped(evt):
        log("[系统] 会话结束。")
        nonlocal finished
        finished = True

    recognizer.session_started.connect(on_session_started)
    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_session_stopped)

    log("[4] 启动连续识别。对着麦克风说话，Ctrl+C 停止。")
    log("-" * 46)
    recognizer.start_continuous_recognition()

    try:
        while not finished:
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("")
        log("[系统] 收到 Ctrl+C。")
    finally:
        try:
            recognizer.stop_continuous_recognition()
        except Exception:
            pass
        log("已停止。")
        log("笔记文件: " + NOTE_FILE)
        log("运行日志: " + LOG_FILE)


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
