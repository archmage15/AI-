# -*- coding: utf-8 -*-
r"""音频文件定位（格式无关层）

项目原本只认 `audio\原音频.wav` / `audio\模仿.wav`。现在改为：
**文件名叫「原音频 / 模仿」，扩展名随便** —— mp3 / wav / flac 都能直接跑，
不用手工转格式。

用法（所有脚本统一）：
    from audio_paths import find_audio, missing_hint
    REF = find_audio("原音频")      # 找不到返回 None
    MY  = find_audio("模仿")

为什么直接读 mp3 而不是先转 wav：
    底层 libsndfile 1.2.2 已内置 mp3 解码器（随 soundfile 0.12+ 一起装），
    读进来的 float32 采样值和 wav 是同一套 API，下游的
    load_mono / frame_rms / pitch_contour / wav2vec2 全都不用改。
    ⚠️ mp3 是有损压缩（尤其低码率会砍掉高频与瞬态），帧级节奏/音高分析
    对波形细节敏感。**对分要求高时建议用 wav，或 mp3 码率 >= 192kbps。**
    放在同一目录下若同时有同名 wav 和 mp3，按下面的优先级取一个（不叠加混用）。

支持的扩展名：mp3 / wav / flac / ogg
不支持 m4a / aac（底层 libsndfile 不带 AAC 解码器），要 m4a 请先自行转成 mp3 或 wav。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(ROOT, "audio")

# 优先级从高到低：mp3 优先（用户日常放的就是 mp3），wav 兜底
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg")

SUPPORTED_HINT = "支持：" + "、".join(AUDIO_EXTS) + "（不支持 m4a / aac）"


def find_audio(stem, src_dir=None):
    """在 audio 目录里找 `<stem>.<ext>`，返回第一个命中的绝对路径；都没有返回 None。

    stem 用不带扩展名的名字，如 "原音频" / "模仿"。
    """
    base = src_dir or SRC_DIR
    for ext in AUDIO_EXTS:
        p = os.path.join(base, stem + ext)
        if os.path.exists(p):
            return p
    return None


def find_reference(src_dir=None):
    """定位原音频：优先 audio\\原音频\\原音频.*（缓存方案后范本移进子目录），
    回退 audio\\原音频.*（旧布局）。找不到返回 None。"""
    base = src_dir or SRC_DIR
    ref_dir = os.path.join(base, "原音频")
    p = find_audio("原音频", ref_dir)
    if p:
        return p
    return find_audio("原音频", base)


def find_audio_pair(src_dir=None):
    """一次性定位 原音频 / 模仿。返回 (ref_path|None, mine_path|None)。"""
    return find_reference(src_dir), find_audio("模仿", src_dir)


def short_name(path):
    """报告里显示用的短文件名（不含目录）。找不到文件时原样返回。"""
    return os.path.basename(path) if path else "（缺失）"


def missing_hint(paths, src_dir=None):
    """缺文件时的提示文本（列出试过的全部路径 + 支持的格式）。"""
    base = src_dir or SRC_DIR
    lines = []
    for stem, p in paths:
        if p is None:
            lines.append("  缺：" + stem + " → 请把音频放进 " + base)
            lines.append("      文件名（不含扩展名）必须是 " + stem)
            lines.append("      " + " / ".join(stem + e for e in AUDIO_EXTS))
    lines.append(SUPPORTED_HINT)
    return "\n".join(lines)