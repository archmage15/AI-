# -*- coding: utf-8 -*-
r"""还原度测试 · 讯飞录音文件转写(LFASR)探测脚本

目的：验证「音频文件 -> 词 + 每个词的起止时间」这条链路能跑通。
这是给「纠正音标时显示对应单词」做前置验证的独立小脚本，不影响别的文件。

用法：
  1. 把 讯飞配置.json（含 appid / secret_key）放在 ROOT（上一层的还原度测试\）
  2. 把要转写的音频放 audio\，命名 原音频.wav（或运行时用参数指定路径）
  3. 运行本脚本，屏幕打印「词 + 起止时间」，并存一份 转写结果.json 到 output\

注意：本脚本不在「一键跑还原度.bat」的三步链里，是单独跑的工具。

鉴权（讯飞 LFASR v2）：
  signa = base64( HMAC-SHA1( key=secret_key, msg=md5(appid + ts) ) )
  上传  https://raasr.xfyun.cn/v2/api/upload   -> 拿 orderId
  轮询  https://raasr.xfyun.cn/v2/api/getResult -> status=4 取结果

依赖：requests / soundfile / numpy（工具包已装）
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import tempfile
import time
from urllib.parse import urlencode

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 目录约定 ----
# 本脚本位于 <项目>\scripts\ 下，ROOT = 项目根（上一层）
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(ROOT, "audio")
OUT_DIR = os.path.join(ROOT, "output")
for _d in (SRC_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

CONFIG_FILE = os.path.join(ROOT, "讯飞配置.json")
DEFAULT_AUDIO = os.path.join(SRC_DIR, "原音频.wav")
OUT_JSON = os.path.join(OUT_DIR, "转写结果.json")

LFASR_HOST = "https://raasr.xfyun.cn/v2/api"


def log(msg=""):
    print(str(msg), flush=True)


def load_config():
    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = json.load(f)
    appid = (cfg.get("appid") or "").strip()
    secret_key = (cfg.get("secret_key") or "").strip()
    if not appid or not secret_key:
        raise RuntimeError("讯飞配置.json 里缺 appid 或 secret_key")
    return appid, secret_key


def make_signa(appid, secret_key, ts):
    """讯飞 LFASR v2 签名：base64(HMAC-SHA1(secret_key, md5(appid+ts)))"""
    md5_hex = hashlib.md5((appid + ts).encode("utf-8")).hexdigest()
    raw = hmac.new(secret_key.encode("utf-8"),
                   md5_hex.encode("utf-8"),
                   hashlib.sha1).digest()
    return base64.b64encode(raw).decode("utf-8")


def prepare_audio(src_path, dst_dir):
    """讯飞常规要求 16kHz；这里统一转成 16kHz 单声道 wav，避免采样率不兼容。
    返回 (临时文件路径, 时长秒)。"""
    import soundfile as sf
    import numpy as np

    audio, sr = sf.read(src_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        n_out = int(round(len(audio) * 16000 / sr))
        x_in = np.linspace(0, 1, len(audio), endpoint=False)
        x_out = np.linspace(0, 1, n_out, endpoint=False)
        audio = np.interp(x_out, x_in, audio).astype(np.float32)
        sr = 16000
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", dir=dst_dir, delete=False)
    tmp.close()
    sf.write(tmp.name, audio, 16000, subtype="PCM_16")
    return tmp.name, len(audio) / 16000.0


def upload(requests, appid, secret_key, audio_path, ts, duration_s):
    signa = make_signa(appid, secret_key, ts)
    params = {
        "appId": appid,
        "signa": signa,
        "ts": ts,
        "fileSize": os.path.getsize(audio_path),
        "fileName": os.path.basename(audio_path),
        "duration": str(int(round(duration_s))),
    }
    url = LFASR_HOST + "/upload?" + urlencode(params)
    with open(audio_path, "rb") as f:
        data = f.read()
    resp = requests.post(url, headers={"Content-type": "application/json"}, data=data, timeout=60)
    return resp.json(), signa


def get_result(requests, appid, signa, ts, order_id):
    params = {
        "appId": appid,
        "signa": signa,
        "ts": ts,
        "orderId": order_id,
        "resultType": "transfer",
    }
    url = LFASR_HOST + "/getResult?" + urlencode(params)
    resp = requests.post(url, timeout=60)
    return resp.json()


def collect_words(node, bg, out):
    """递归收集带 cw/wb/we 的词节点，绝对时间 = bg + 帧*10ms。"""
    if isinstance(node, dict):
        if "cw" in node and ("wb" in node or "we" in node):
            wb = int(node.get("wb") or 0)
            we = int(node.get("we") or 0)
            for c in node.get("cw") or []:
                w = (c.get("w") or "").strip()
                if w and c.get("wp") != "p":
                    out.append({"word": w, "start_ms": bg + wb * 10, "end_ms": bg + we * 10})
        for v in node.values():
            collect_words(v, bg, out)
    elif isinstance(node, list):
        for v in node:
            collect_words(v, bg, out)


def parse_order_result(order_result_text):
    data = json.loads(order_result_text)
    words = []
    for seg in data.get("lattice", []):
        one = json.loads(seg["json_1best"])
        st = one.get("st", {})
        bg = int(st.get("bg") or one.get("bg") or 0)
        collect_words(st, bg, words)
    return words


def main():
    import requests  # 放函数内，import 失败时报错更清楚

    log("=" * 62)
    log("   还原度测试 · 讯飞转写(LFASR)探测")
    log("=" * 62)

    audio_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO
    if not os.path.exists(audio_path):
        log("X 找不到音频：" + audio_path)
        return 1
    if not os.path.exists(CONFIG_FILE):
        log("X 找不到配置：" + CONFIG_FILE)
        return 1

    appid, secret_key = load_config()
    log("音频：" + audio_path)

    tmp_audio, duration_s = prepare_audio(audio_path, OUT_DIR)
    log("已转 16kHz 单声道，时长 %.2f 秒，准备上传 ..." % duration_s)

    try:
        ts = str(int(time.time()))
        resp, signa = upload(requests, appid, secret_key, tmp_audio, ts, duration_s)
        log("上传返回：" + json.dumps(resp, ensure_ascii=False))
        if resp.get("code") != "000000":
            log("X 上传失败（见上方返回）。常见：服务未开通 / 时长不符 / 白名单。")
            return 1
        order_id = resp["content"]["orderId"]
        log("orderId = " + order_id)

        log("等待转写 ...")
        order_result_text = None
        for _ in range(60):
            time.sleep(3)
            r = get_result(requests, appid, signa, ts, order_id)
            if r.get("code") != "000000":
                log("X 查询失败：" + json.dumps(r, ensure_ascii=False))
                return 1
            content = r["content"]
            status = content.get("orderInfo", {}).get("status")
            log("  状态 status=%s" % status)
            if status == 4:
                order_result_text = content.get("orderResult")
                break
            if status == -1:
                log("X 订单失败：" + json.dumps(content.get("orderInfo", {}), ensure_ascii=False))
                return 1
        if not order_result_text:
            log("X 超时，未拿到结果。")
            return 1

        words = parse_order_result(order_result_text)
        log("")
        log("=" * 62)
        log("   转写结果（词 + 起止时间）")
        log("=" * 62)
        if not words:
            log("（没解析到词，检查音频是否有语音）")
        for w in words:
            log("  %7.2fs - %7.2fs   %s" % (w["start_ms"] / 1000.0,
                                            w["end_ms"] / 1000.0,
                                            w["word"]))
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False, indent=2)
        log("")
        log("（结果已存到 转写结果.json，共 %d 个词）" % len(words))
        return 0
    finally:
        try:
            os.remove(tmp_audio)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        log("[致命错误]")
        log(traceback.format_exc())
        sys.exit(1)
