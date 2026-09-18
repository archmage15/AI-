# -*- coding: utf-8 -*-
"""听记卡 · 链路诊断（不需要麦克风）

依次检查四件事：
  1. 域名能不能解析
  2. 443 端口能不能连
  3. 密钥能不能换到令牌（REST）
  4. Speech SDK 能不能建立起识别会话（WebSocket，这个才是真正用的通道）

用法：python diag.py        （双击/手动跑，结束会停住）
      python diag.py --auto （自动跑完退出）
"""
import importlib.util
import os
import socket
import sys
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

AUTO = "--auto" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))


def hold():
    if AUTO:
        return
    try:
        input("按回车退出...")
    except EOFError:
        pass


# 从主脚本读密钥，避免重复填写。
# 主脚本支持环境变量 / keys.txt，这里先让它加载一遍再取值。
spec = importlib.util.spec_from_file_location("lt", os.path.join(HERE, "live_transcribe.py"))
lt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lt)
lt._load_keys()
KEY = lt.SPEECH_KEY
REGION = lt.SPEECH_REGION
HOST = REGION + ".api.cognitive.microsoft.com"

print("=" * 50)
print("   听记卡 · 链路诊断")
print("=" * 50)
print("区域:", REGION)
print("主机:", HOST)
print("密钥长度:", len(KEY), "字符")
print("-" * 50)

print("[1] 解析域名 ...")
try:
    ip = socket.gethostbyname(HOST)
    print("    OK ->", ip)
except Exception as e:
    print("    X 失败:", e)
    print("    => 域名解析不了，网络层就断了")
    hold()
    sys.exit(1)

print("[2] 连接 443 端口 ...")
t0 = time.time()
try:
    s = socket.create_connection((HOST, 443), timeout=10)
    s.close()
    print("    OK，耗时 %.2f 秒" % (time.time() - t0))
except Exception as e:
    print("    X 失败:", type(e).__name__, e)
    print("    => 端口连不上")
    hold()
    sys.exit(1)

print("[3] 用密钥换令牌（REST 通道）...")
url = "https://" + HOST + "/sts/v1.0/issueToken"
req = urllib.request.Request(url, data=b"", method="POST")
req.add_header("Ocp-Apim-Subscription-Key", KEY)
t0 = time.time()
rest_ok = False
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        token = r.read()
        code = r.status
    print("    HTTP %s，耗时 %.2f 秒，令牌 %d 字节" % (code, time.time() - t0, len(token)))
    print("    => 密钥有效，REST 通畅")
    rest_ok = True
except urllib.error.HTTPError as e:
    print("    HTTP %s %s，耗时 %.2f 秒" % (e.code, e.reason, time.time() - t0))
    try:
        print("    详情:", e.read().decode("utf-8", "ignore")[:200])
    except Exception:
        pass
    if e.code == 401:
        print("    => 密钥无效，或密钥不属于", REGION)
    else:
        print("    => 服务端拒绝")
except Exception as e:
    print("    X 失败:", type(e).__name__, e, "耗时 %.2f 秒" % (time.time() - t0))
    print("    => REST 请求超时/发不出去")

print("[4] 建立 Speech SDK 识别会话（WebSocket 通道，识别真正走这条路）...")
sdk_ok = False
try:
    import azure.cognitiveservices.speech as speechsdk

    cfg = speechsdk.SpeechConfig(subscription=KEY, region=REGION)
    cfg.speech_recognition_language = "zh-CN"
    fmt = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    push = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
    audio = speechsdk.audio.AudioConfig(stream=push)
    rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)

    st = {"started": False, "err": False, "errinfo": ""}

    def on_start(evt):
        st["started"] = True

    def on_cancel(evt):
        st["err"] = True
        st["errinfo"] = "reason=%s code=%s details=%s" % (
            evt.reason, getattr(evt, "error_code", None), getattr(evt, "error_details", None))

    rec.session_started.connect(on_start)
    rec.canceled.connect(on_cancel)

    t0 = time.time()
    rec.start_continuous_recognition()
    while time.time() - t0 < 12:
        push.write(b"\x00" * 3200)
        time.sleep(0.1)
        if st["started"] or st["err"]:
            break

    if st["started"]:
        print("    会话已建立（session_started 触发），耗时 %.2f 秒" % (time.time() - t0))
        print("    => SDK 通道正常")
        sdk_ok = True
    elif st["err"]:
        print("    X 会话被取消:", st["errinfo"])
    else:
        print("    X 12 秒内没建立会话（连不上识别服务）")

    rec.stop_continuous_recognition()
    try:
        push.close()
    except Exception:
        pass
except Exception as e:
    print("    X 异常:", type(e).__name__, e)

print("-" * 50)
print("结论：")
if rest_ok and sdk_ok:
    print("  网络和密钥都没问题，问题在麦克风采集端。")
elif rest_ok and not sdk_ok:
    print("  密钥对、REST 通，但 SDK 的 WebSocket 通道建不起来 —— 多半是网络对 wss 做了拦截。")
elif not rest_ok:
    print("  连密钥都换不到令牌，先解决网络或密钥问题。")
print("=" * 50)
hold()
