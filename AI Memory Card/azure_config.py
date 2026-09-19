# -*- coding: utf-8 -*-
"""读取 Azure 语音服务配置。

密钥只存在本机的 azure配置.json 里（已 gitignore，永不进版本库）。
代码里只留"去哪读"，不留值。

用法：
    from azure_config import load_azure_config
    key, region = load_azure_config()

配置文件格式（放在本文件同目录）：
    {
      "speech_key": "你的密钥",
      "speech_region": "eastasia"
    }
"""
import json
import os

# 配置文件查找顺序：同目录 → 上一级
_SEARCH = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "azure配置.json"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "azure配置.json"),
]


def load_azure_config():
    """返回 (speech_key, speech_region)。找不到配置时抛出清晰的错误。"""
    for path in _SEARCH:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            key = cfg.get("speech_key", "").strip()
            region = cfg.get("speech_region", "").strip() or "eastasia"
            if not key:
                raise RuntimeError(
                    "配置文件里 speech_key 为空：" + path)
            return key, region

    raise RuntimeError(
        "找不到 azure配置.json。\n"
        "请在下面任一位置新建该文件：\n"
        "    " + "\n    ".join(_SEARCH) + "\n"
        "内容格式：\n"
        '    {\n      "speech_key": "你的密钥",\n      "speech_region": "eastasia"\n    }\n'
        "（该文件已被 .gitignore 忽略，不会上传）"
    )
