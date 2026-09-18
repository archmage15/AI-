# -*- coding: utf-8 -*-
"""麦克风通道测试：不要求人说话，只验证能否打开麦克风并建立识别会话。"""
import os
import sys
import time
import traceback

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("lt", os.path.join(HERE, "live_transcribe.py"))
lt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lt)

import azure.cognitiveservices.speech as speechsdk

print("=" * 52)
print("   麦克风通道测试（静默 6 秒）")
print("=" * 52)
print("Python:", sys.executable)

try:
    cfg = speechsdk.SpeechConfig(subscription=lt.SPEECH_KEY, region=lt.SPEECH_REGION)
    cfg.speech_recognition_language = "zh-CN"
    audio = speechsdk.audio.AudioConfig(use_default_microphone=True)
    print("[1] AudioConfig(默认麦克风): OK")
    rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)
    print("[2] SpeechRecognizer: OK")
except Exception:
    print("[X] 初始化失败：")
    print(traceback.format_exc())
    sys.exit(1)

st = {"started": False, "canceled": None, "rec": 0, "ing": 0}


def on_start(evt):
    st["started"] = True
    print("    [session_started] 会话已建立")


def on_cancel(evt):
    st["canceled"] = "reason=%s code=%s details=%s" % (
        evt.reason, getattr(evt, "error_code", None), getattr(evt, "error_details", None))
    print("    [canceled]", st["canceled"])


def on_rec(evt):
    st["rec"] += 1


def on_ing(evt):
    st["ing"] += 1


rec.session_started.connect(on_start)
rec.canceled.connect(on_cancel)
rec.recognized.connect(on_rec)
rec.recognizing.connect(on_ing)

print("[3] 启动识别，静默采集 6 秒 ...")
rec.start_continuous_recognition()
t0 = time.time()
while time.time() - t0 < 6:
    time.sleep(0.3)
rec.stop_continuous_recognition()

print("-" * 52)
print("session_started :", st["started"])
print("canceled        :", st["canceled"])
print("中间结果次数    :", st["ing"])
print("定稿次数        :", st["rec"])
if st["started"] and not st["canceled"]:
    print("=> 麦克风通道可用，程序侧没问题")
else:
    print("=> 麦克风通道有问题")
print("=" * 52)
