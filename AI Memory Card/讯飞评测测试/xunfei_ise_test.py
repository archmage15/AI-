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
# 注意：APIKey 和 APISecret 都是 32 位字符串，肉眼分不出，极易填反。
#       填反的症状是握手报 401 "apikey not found"。
#       拿不准就跑同目录的 swap_test.bat，它会告诉你要不要对调。
APP_ID = "e9d09a3a"
API_KEY = "02789c5937198efc7004ec8ada615ccf"
API_SECRET = "ZDM0OTFiZmViYTM3ZDM0YjI3NmYzMjJj"

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

    # 原始 XML 只存文件，不打印到屏幕
    try:
        with open(RESULT_XML, "w", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        pass

    try:
        root = ET.fromstring(payload)
    except Exception as e:
        log("")
        log("X 返回的结果读不出来，可能不是标准格式。")
        log("  原始内容已存到 评测结果.xml，把它发给小巴看看。")
        log("  （技术原因：%s）" % e)
        return

    chapter = root.find(".//read_chapter")
    if chapter is None:
        log("")
        log("X 返回里没有评测内容。原始结果已存到 评测结果.xml。")
        return

    def f(key, default=0.0):
        try:
            return float(chapter.get(key, default))
        except Exception:
            return default

    # 讯飞是 5 分制，换成 100 分制更直观
    def p100(v):
        return v * 20.0

    log("")
    log("=" * 62)
    log("   评测结果")
    log("=" * 62)
    log("")

    # 讯飞判定"乱说"时分数不可信，先提醒
    if chapter.get("is_rejected") == "true":
        log("  ⚠ 讯飞认为这次录音不像是照着原文读的，分数仅供参考。")
        log("")

    total = f("total_score")
    log("  综合得分      %5.1f / 100" % p100(total))
    log("  发音准确度    %5.1f / 100" % p100(f("accuracy_score")))
    log("  流利度        %5.1f / 100" % p100(f("fluency_score")))
    log("  完整度        %5.1f / 100" % p100(f("integrity_score")))
    log("")

    # ---- 逐词明细，从低分到高分 ----
    words = []
    for w in chapter.findall(".//word"):
        content = (w.get("content") or "").strip()
        if not content or content == "sil":
            continue
        try:
            sc = float(w.get("total_score", 0))
        except Exception:
            sc = 0.0
        words.append((content, sc, w))

    if not words:
        log("  （没有拆出单词，可能是整句判定失败）")
        log("")
        log("  原始数据已存到 评测结果.xml")
        return

    words.sort(key=lambda t: t[1])

    bad = [t for t in words if t[1] < 4.0]
    if bad:
        log("-" * 62)
        log("  有问题的词（按严重程度排）")
        log("-" * 62)
        log("")
        for content, sc, w in bad:
            issue = describe_word_issue(w)
            log("  %-16s %5.1f 分%s" % (content, p100(sc), issue))
            # 音节级：哪个音节不行
            for syl in w.findall("syll"):
                try:
                    ss = float(syl.get("syll_score", 0))
                except Exception:
                    ss = 0.0
                if ss < 4.0:
                    log("      └ 音节 [%s]  %5.1f 分" % (syl.get("content"), p100(ss)))
                    # 音素级：哪个音偏得最多
                    phones = []
                    for ph in syl.findall("phone"):
                        try:
                            gw = float(ph.get("gwpp", "0"))
                        except Exception:
                            gw = 0.0
                        phones.append((ph.get("content"), gw, ph.get("dp_message")))
                    wrong = [p for p in phones if p[1] < -1.0]
                    for name, gw, dpm in sorted(wrong, key=lambda x: x[1]):
                        log("          · 音 [%s]  偏离明显" % name)
            log("")
    else:
        log("-" * 62)
        log("  每个词都在 80 分以上，这遍读得不错。")
        log("-" * 62)
        log("")

    # ---- 全句概览：所有词的分数 ----
    log("-" * 62)
    log("  全句单词得分（从低到高）")
    log("-" * 62)
    line = "  "
    for content, sc, w in words:
        piece = "%s %.0f  " % (content, p100(sc))
        if len(line) + len(piece) > 60:
            log(line.rstrip())
            line = "  "
        line += piece
    if line.strip():
        log(line.rstrip())
    log("")

    log("-" * 62)
    log("  怎么用")
    log("-" * 62)
    log("  · 先挑分数最低的那个词反复练，练到 80 分以上")
    log("  · 标注【偏离明显】的音，是具体发音要改的地方")
    log("  · 标记【没读】的词，说明吞音或语速太快漏掉了")
    log("")
    log("  原始数据已存到 评测结果.xml（给技术人员看的）")


def describe_word_issue(w):
    """根据讯飞字段，用大白话描述这个词的问题。"""
    dp = w.get("dp_message", "0")
    # dp_message != 0 表示有"增删"（漏读/多读）
    if dp not in ("0", "", None):
        return "  ← 没读 / 漏读"
    # 音节错误标记
    has_serr = any(s.get("serr_msg", "0") not in ("0", "", None)
                   for s in w.findall("syll"))
    if has_serr:
        return "  ← 读错了"
    return "  ← 偏低"


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
