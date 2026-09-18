# -*- coding: utf-8 -*-
"""讯飞语音评测(流式版) · 最小验证脚本

目的：用一句英文，确认讯飞 ISE 能不能做「发音 + 音变 + 节奏」评测。

用法：
  1. 填好下面三把钥匙（APP_ID / API_KEY / API_SECRET）
  2. 双击 运行测试.bat
  3. 屏幕显示标准句 → 按回车开始 → 对着麦克风跟读 → 按回车结束
  4. 看结果：整句分 + 逐词/逐音素明细；原始 XML 会存到 评测结果.xml

注意：
  - 音频要求 16kHz / 16bit / 单声道（脚本已自动处理）
  - 单次会话不能超过 5 分钟
  - 英文文本建议 >= 5 个单词，才有"标准度"分
  - 若返回 is_rejected="true"，表示引擎判定"乱说"，分数可忽略
"""
import base64
import datetime
import hashlib
import hmac
import json
import os
import ssl
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import numpy as np
import sounddevice as sd
import websocket

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_XML = os.path.join(HERE, "评测结果.xml")
RAW_LOG = os.path.join(HERE, "测试日志.txt")

# ========== 1. 三把钥匙（去讯飞控制台复制粘贴）==========
APP_ID = "在这里填APPID"
API_KEY = "在这里填APIKey"
API_SECRET = "在这里填APISecret"

# ========== 2. 评测参数 ==========
REF_TEXT = "I want to improve my English pronunciation."   # 标准句（参考文本）
GROUP = "adult"        # 成人：adult；儿童：pupil
MAX_SECONDS = 15       # 录音最长秒数（防止忘记按回车）
SAMPLE_RATE = 16000

HOST = "ise-api.xfyun.cn"
URL = "wss://ise-api.xfyun.cn/v2/open-ise"

STATE = {"audio": b"", "payload": None}


def log(msg=""):
    text = str(msg)
    try:
        print(text, flush=True)
    except Exception:
        pass
    try:
        with open(RAW_LOG, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


# ---------- 鉴权 URL ----------
def build_url():
    date = format_date_time(mktime(datetime.datetime.now().timetuple()))
    origin = "host: %s\ndate: %s\nGET /v2/open-ise HTTP/1.1" % (HOST, date)
    sig = hmac.new(API_SECRET.encode(), origin.encode(), hashlib.sha256).digest()
    sig = base64.b64encode(sig).decode()
    auth_origin = ('api_key="%s", algorithm="hmac-sha256", headers="host date request-line", '
                   'signature="%s"' % (API_KEY, sig))
    auth = base64.b64encode(auth_origin.encode()).decode()
    return URL + "?" + urlencode({"authorization": auth, "date": date, "host": HOST})


# ---------- 录音 ----------
def record():
    log("")
    log("请跟读这句话：")
    log("    " + REF_TEXT)
    log("")
    try:
        input("按回车开始录音 ...")
    except EOFError:
        pass
    for i in (3, 2, 1):
        log("    %d ..." % i)
        time.sleep(1)

    frames = []
    stop_flag = {"stop": False}

    def wait_enter():
        try:
            input()
        except Exception:
            pass
        stop_flag["stop"] = True

    threading.Thread(target=wait_enter, daemon=True).start()

    log(">>> 开始录音，读完后按回车结束 <<<")

    def callback(indata, n, t, status):
        frames.append(indata.copy())

    t0 = time.time()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        while not stop_flag["stop"] and (time.time() - t0) < MAX_SECONDS:
            time.sleep(0.05)

    if frames:
        audio = np.concatenate(frames, axis=0)
    else:
        audio = np.zeros((0, 1), dtype="int16")
    log("录音结束，时长 %.2f 秒" % (len(audio) / SAMPLE_RATE))
    return audio.tobytes()


# ---------- WebSocket 回调 ----------
def on_open(ws):
    log("")
    log("已连接讯飞，发送参数帧 ...")
    frame = {
        "common": {"app_id": APP_ID},
        "business": {
            "sub": "ise",
            "ent": "en_vip",                 # 英文
            "category": "read_sentence",     # 句子朗读
            "cmd": "ssb",
            "text": "\uFEFF" + REF_TEXT,     # 带 BOM 头
            "tte": "utf-8",
            "ttp_skip": True,
            "aue": "raw",                    # 原始 PCM
            "auf": "audio/L16;rate=16000",
            "group": GROUP,
        },
        "data": {"status": 0, "data": ""},
    }
    ws.send(json.dumps(frame))

    audio = STATE["audio"]
    step = 1280          # 每片 1280 字节，约 40ms
    pos = 0
    first = True
    while pos < len(audio):
        chunk = audio[pos:pos + step]
        pos += step
        aus = 1 if first else 2
        first = False
        ws.send(json.dumps({
            "business": {"cmd": "auw", "aus": aus, "aue": "raw"},
            "data": {"status": 1, "data": base64.b64encode(chunk).decode()},
        }))
        time.sleep(0.04)

    ws.send(json.dumps({
        "business": {"cmd": "auw", "aus": 4, "aue": "raw"},
        "data": {"status": 2, "data": ""},
    }))
    log("音频已发完，等待结果 ...")


def on_message(ws, message):
    try:
        msg = json.loads(message)
    except Exception:
        log("返回不是 JSON：" + str(message)[:300])
        return
    data = msg.get("data") or {}
    if data.get("status") == 2:
        raw = data.get("data", "")
        try:
            payload = base64.b64decode(raw).decode("utf-8", "ignore")
        except Exception:
            payload = raw
        STATE["payload"] = payload
        ws.close()


def on_error(ws, error):
    log("[错误] " + str(error))


def on_close(ws, code, msg):
    pass


# ---------- 结果展示 ----------
def show_result(payload):
    if not payload:
        log("")
        log("X 没有拿到评测结果。请把 测试日志.txt 发给小巴看看。")
        return

    try:
        with open(RESULT_XML, "w", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        pass

    log("")
    log("=" * 62)
    log("   讯飞原始结果")
    log("=" * 62)
    log(payload)
    log("")
    log("（原始结果已存到 评测结果.xml）")

    try:
        root = ET.fromstring(payload)
    except Exception as e:
        log("")
        log("（结果不是标准 XML，暂不自动解析：%s）" % e)
        return

    log("")
    log("=" * 62)
    log("   自动解析（完整体检单）")
    log("=" * 62)

    def dump(el, depth=0):
        if el.attrib:
            log("  " * depth + "<%s> %s" % (el.tag, json.dumps(el.attrib, ensure_ascii=False)))
        else:
            log("  " * depth + "<%s>" % el.tag)
        for c in el:
            dump(c, depth + 1)

    dump(root)

    log("")
    log("=" * 62)
    log("看完这份结果，把 测试日志.txt 发给小巴，一起判断：")
    log("  - 有没有返回音素级明细")
    log("  - 有没有连读 / 停顿 / 重读 / 升降调 的检错字段")
    log("=" * 62)


def main():
    log("=" * 62)
    log("   讯飞语音评测(流式版) · 最小验证    "
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("=" * 62)

    if "在这里填" in APP_ID or "在这里填" in API_KEY or "在这里填" in API_SECRET:
        log("")
        log("X 请先打开 xunfei_ise_test.py，把顶部三把钥匙填好：")
        log("    APP_ID / API_KEY / API_SECRET")
        log("  （去讯飞控制台 → 语音评测(流式版) → 服务接口认证信息）")
        return

    STATE["audio"] = record()

    log("")
    log("连接讯飞 ...")
    ws = websocket.WebSocketApp(
        build_url(),
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    show_result(STATE["payload"])


if __name__ == "__main__":
    try:
        with open(RAW_LOG, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass
    try:
        main()
    except Exception:
        log("[致命错误]")
        log(traceback.format_exc())
    try:
        input("按回车退出 ...")
    except Exception:
        pass
