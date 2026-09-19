# -*- coding: utf-8 -*-
"""临时把 API_KEY / API_SECRET 对调，测一次握手。

只在本脚本内存里对调，不改你 xunfei_ise_test.py 的原文件。
用途：判断报错是不是"两个32位字段填反了"造成的。
"""
import base64
import datetime
import hashlib
import hmac
import importlib.util
import os
import ssl
import sys
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import websocket

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "xft", os.path.join(HERE, "xunfei_ise_test.py"))
_xft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xft)

APP_ID = _xft.APP_ID
KEY = _xft.API_KEY
SECRET = _xft.API_SECRET

HOST = "ise-api.xfyun.cn"
PATH = "/v2/open-ise"


def try_handshake(api_key, api_secret, label):
    date = format_date_time(mktime(datetime.datetime.now().timetuple()))
    origin = "host: %s\ndate: %s\nGET %s HTTP/1.1" % (HOST, date, PATH)
    sig = hmac.new(api_secret.encode(), origin.encode(), hashlib.sha256).digest()
    sig = base64.b64encode(sig).decode()
    auth_origin = ('api_key="%s", algorithm="hmac-sha256", '
                   'headers="host date request-line", signature="%s"'
                   % (api_key, sig))
    auth = base64.b64encode(auth_origin.encode()).decode()
    url = "wss://%s%s?%s" % (HOST, PATH,
                             urlencode({"authorization": auth, "date": date, "host": HOST}))
    try:
        ws = websocket.create_connection(url, timeout=15,
                                         sslopt={"cert_reqs": ssl.CERT_NONE})
        ws.close()
        return True, "OK"
    except Exception as e:
        return False, str(e)


print("=" * 62)
print("   讯飞钥匙 · 对调测试")
print("=" * 62)
print()

print("[A] 按你现在的填法（KEY=APIKey, SECRET=APISecret）")
ok_a, err_a = try_handshake(KEY, SECRET, "A")
print("    结果:", "✓ 通过" if ok_a else "✗ 失败")
if not ok_a:
    print("    报错:", err_a[:150])
print()

print("[B] 对调（KEY=APISecret, SECRET=APIKey）")
ok_b, err_b = try_handshake(SECRET, KEY, "B")
print("    结果:", "✓ 通过" if ok_b else "✗ 失败")
if not ok_b:
    print("    报错:", err_b[:150])
print()

print("=" * 62)
print("   结论")
print("=" * 62)
if ok_a and not ok_b:
    print("  你现在填的是对的。问题不在钥匙对调。")
    print("  => 大概率是：应用没开通「语音评测（流式版）」，")
    print("     或该应用被停用/欠费。去控制台确认服务列表。")
elif ok_b and not ok_a:
    print("  ★ 找到问题了：APIKey 和 APISecret 填反了！")
    print("  => 打开 xunfei_ise_test.py，把这两行的值互换：")
    print("        API_KEY    = \"...\"")
    print("        API_SECRET = \"...\"")
elif ok_a and ok_b:
    print("  两种填法都能通过？这很反常。")
    print("  => 说明讯飞对该账号鉴权宽松，或 key 是通用的。")
    print("     那 401 的来源要另查，把日志发我。")
else:
    print("  两种填法都失败。")
    print("  => 钥匙本身有问题，或者应用没开通服务。")
    print("     去控制台：我的应用 → 确认服务列表含「语音评测（流式版）」")
    print("     → 重新复制三把钥匙（APPID 8位 / APIKey 32位 / APISecret 32位）")
    print()
    print("  如果控制台里根本没有「语音评测」这个服务，")
    print("  说明还没开通，需要先添加服务（可能需企业实名）。")

print()
input("按回车退出 ...")
