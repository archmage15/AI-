# -*- coding: utf-8 -*-
"""讯飞钥匙自检：只做鉴权，不录音、不评测。

作用：拿着三把钥匙去连一次讯飞，看能不能通过握手。
     通过了 -> 钥匙没问题，问题在别处
     401/403 -> 明确告诉你哪一把不对
"""
import base64
import datetime
import hashlib
import hmac
import importlib.util
import json
import os
import ssl
import sys
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import websocket

# 直接复用 xunfei_ise_test.py 里填好的三把钥匙，不用填两遍
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "xft", os.path.join(HERE, "xunfei_ise_test.py"))
_xft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xft)
APP_ID = _xft.APP_ID
API_KEY = _xft.API_KEY
API_SECRET = _xft.API_SECRET

HOST = "ise-api.xfyun.cn"
PATH = "/v2/open-ise"
URL = "wss://ise-api.xfyun.cn/v2/open-ise"


def build_url():
    date = format_date_time(mktime(datetime.datetime.now().timetuple()))
    origin = "host: %s\ndate: %s\nGET %s HTTP/1.1" % (HOST, date, PATH)
    sig = hmac.new(API_SECRET.encode(), origin.encode(), hashlib.sha256).digest()
    sig = base64.b64encode(sig).decode()
    auth_origin = ('api_key="%s", algorithm="hmac-sha256", '
                   'headers="host date request-line", signature="%s"'
                   % (API_KEY, sig))
    auth = base64.b64encode(auth_origin.encode()).decode()
    return URL + "?" + urlencode({"authorization": auth, "date": date, "host": HOST})


print("=" * 60)
print("   讯飞钥匙自检（只握手，不录音）")
print("=" * 60)
print()

# ---- 先做格式体检 ----
print("[格式体检]")
ok_fmt = True
for name, val in (("APP_ID", APP_ID), ("API_KEY", API_KEY), ("API_SECRET", API_SECRET)):
    if "在这里填" in val:
        print("  %-11s ✗ 还是占位符，没填" % name)
        ok_fmt = False
        continue
    problems = []
    if val != val.strip():
        problems.append("首尾有空格/换行")
    if " " in val:
        problems.append("中间含空格")
    if not val.isascii():
        problems.append("含非 ASCII 字符（可能是中文标点）")
    tag = "✓ 长度 %d" % len(val) if not problems else "✗ " + "、".join(problems)
    print("  %-11s %s" % (name, tag))
    if problems:
        ok_fmt = False

if not ok_fmt:
    print()
    print("=> 先修好上面的格式问题再跑。常见原因是从控制台复制时")
    print("   带了空格或换行，或者引号是中文引号。")
    print()
    input("按回车退出 ...")
    raise SystemExit(1)

print()
print("  APP_ID 长度 8 才是正常的（十六进制）。")
print("  API_KEY / API_SECRET 都是 32 位。两个很容易互换，")
print("  如果下面握手报 apikey not found，先把这两把对调试试。")
print()

# ---- 握手 ----
print("[连接讯飞鉴权]")
try:
    ws = websocket.create_connection(
        build_url(), timeout=15, sslopt={"cert_reqs": ssl.CERT_NONE})
    print("  ✓ 握手成功！三把钥匙都是对的。")
    print("    => 问题不在钥匙，而在别处（看运行日志的详细报错）。")
    ws.close()
except Exception as e:
    msg = str(e)
    print("  ✗ 握手失败")
    print()
    print("  原始返回：")
    for line in msg.splitlines():
        print("     ", line)
    print()
    print("  【怎么读这个错】")
    if "apikey not found" in msg:
        print("    -> 讯飞不认识你的 APIKey。可能：")
        print("       ① APIKey 和 APISecret 填反了（最常见，两个都是32位）")
        print("       ② 这个应用没有开通「语音评测（流式版）」服务")
        print("       ③ 三把钥匙不是来自同一个应用")
    elif "401" in msg or "403" in msg:
        print("    -> 鉴权被拒。检查三把钥匙是否配套、服务是否开通。")
    elif "date" in msg.lower() or "signature" in msg.lower():
        print("    -> 签名/时间问题。校准系统时间试试。")
    else:
        print("    -> 把上面原始返回发给小巴看。")
    print()
    print("  【排查动作】")
    print("    1. 登录 https://console.xfyun.cn/ → 我的应用")
    print("    2. 点进你要用的那个应用，确认服务列表里有")
    print("       「语音评测（流式版）」")
    print("    3. 在「服务接口认证信息」里，把 APPID / APIKey / APISecret")
    print("       三个一起重新复制一遍，逐个核对，别串行")

print()
input("按回车退出 ...")
