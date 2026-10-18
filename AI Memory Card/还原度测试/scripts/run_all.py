# -*- coding: utf-8 -*-
r"""一键跑还原度 · 自动路由（单句模式 / 长音频模式）

被 一键跑还原度.bat 调用，替代原来固定跑三个脚本的写法。

路由逻辑：
  原音频时长 <= LONG_AUDIO_S（20s）→ 单句模式：compare_phoneme → compare_rhythm
  原音频时长 >  LONG_AUDIO_S        → 长音频模式：
      1. 需要参考音素表（带句子边界）。没有/过期 → 先拿台词文本：
         - audio\转写文本.txt 存在且比原音频新 → 直接用
         - 否则有 讯飞配置.json → 跑 xunfei_transcribe.py 自动转写
         - 都没有 → 明确报错并告诉用户怎么办（不静默失败）
      2. 转写文本做轻量清理（andWhat → and What 这类粘连）后生成参考音素表
      3. compare_long.py（四维度逐句评测）
  最后都跑 overall_score.py

用法：python scripts\run_all.py
"""
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(ROOT, "audio")

REF_WAV = os.path.join(SRC_DIR, "原音频.wav")
MY_WAV = os.path.join(SRC_DIR, "模仿.wav")
REF_PHONEME_JSON = os.path.join(SRC_DIR, "参考音素.json")
TRANSCRIBE_TXT = os.path.join(SRC_DIR, "转写文本.txt")
XUNFEI_CONFIG = os.path.join(ROOT, "讯飞配置.json")

LONG_AUDIO_S = 20.0        # 原音频超过此时长（秒）走长音频模式

sys.path.insert(0, HERE)
from logkit import log, out, log_setup, log_done, log_step, log_error


def audio_duration(path):
    """音频时长（秒）；读不到返回 None。"""
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return None


def newer_than(path, ref_path):
    """path 是否比 ref_path 新（用于判断参考音素表/转写文本是否过期）。"""
    try:
        return os.path.getmtime(path) > os.path.getmtime(ref_path)
    except Exception:
        return False


def has_sentence_bounds(path):
    """参考音素.json 是否带句子边界（长音频模式必需）。"""
    try:
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("句子边界"))
    except Exception:
        return False


def clean_transcript(text):
    """转写文本轻量清理：修 ASR 粘连（andWhat→and What）、压多余空白。
    只做安全的字符级修补，不改内容。"""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)     # 小写后紧跟大写 → 插空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_script(name, args=()):
    """调另一个脚本（stdin 关掉，避免结尾 input() 卡住）。返回退出码。"""
    cmd = [sys.executable, os.path.join(HERE, name)] + list(args)
    log("   运行：%s %s" % (name, " ".join(args[:1]) + ("..." if args else "")), verbose=True)
    try:
        r = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL)
        return r.returncode
    except Exception as e:
        log("（运行 %s 失败：%s）" % (name, e), show=True)
        return 1


def ensure_reference(long_mode):
    """长音频模式：确保有带句子边界的参考音素表。返回 True=就绪。"""
    if (os.path.exists(REF_PHONEME_JSON) and has_sentence_bounds(REF_PHONEME_JSON)
            and newer_than(REF_PHONEME_JSON, REF_WAV)):
        log("   参考音素.json 已存在且比原音频新，跳过生成", verbose=True)
        return True

    # 需要台词文本：优先用现成转写文本，其次讯飞自动转写
    text = None
    if os.path.exists(TRANSCRIBE_TXT) and newer_than(TRANSCRIBE_TXT, REF_WAV):
        with open(TRANSCRIBE_TXT, encoding="utf-8") as f:
            text = f.read().strip()
        log("   用现成的 audio\\转写文本.txt（比原音频新）")
    elif os.path.exists(XUNFEI_CONFIG):
        log("   没有现成转写文本 → 调讯飞自动转写")
        if run_script("xunfei_transcribe.py") != 0:
            out("")
            out("X 讯飞转写失败（可能没网 / 钥匙无效 / 服务未开通）。")
            out("  两个解决办法：")
            out("   1. 检查网络与 讯飞配置.json，修好后重跑；")
            out("   2. 或手动把这段音频的台词文本写进 audio\\转写文本.txt 再重跑。")
            return False
        if os.path.exists(TRANSCRIBE_TXT):
            with open(TRANSCRIBE_TXT, encoding="utf-8") as f:
                text = f.read().strip()
    if not text:
        out("")
        out("X 长音频模式需要「台词文本」才能生成参考音素表，但没找到。")
        out("  两个解决办法（二选一）：")
        out("   1. 自动转写：在项目根目录放 讯飞配置.json"
            "（{\"app_id\":\"..\",\"secret_key\":\"..\"}），重跑即可；")
        out("   2. 手动提供：把这段音频的英文台词文本存成 audio\\转写文本.txt，重跑。")
        return False

    text = clean_transcript(text)
    log("   台词 %d 字符，生成参考音素表（含句子边界）..." % len(text), verbose=True)
    if run_script("gen_ref_phonemes.py", [text]) != 0:
        out("X 生成参考音素表失败（espeak-ng 是否安装？见 gen_ref_phonemes.py 说明）")
        return False
    return True


def main():
    log_setup("一键路由")
    out("=" * 62)
    out("   还原度测试 · 一键跑（自动路由）")
    out("=" * 62)

    if not (os.path.exists(REF_WAV) and os.path.exists(MY_WAV)):
        out("X 没找到音频文件，请把两个 wav 放进 audio\\：")
        out("    " + REF_WAV)
        out("    " + MY_WAV)
        log_done("（缺音频，提前退出）")
        return

    dur = audio_duration(REF_WAV) or 0.0
    long_mode = dur > LONG_AUDIO_S
    out("原音频时长 %.1f 秒 → %s" % (dur, "长音频模式（逐句评测）" if long_mode else "单句模式"))

    if long_mode:
        log_step("长音频模式：准备参考音素表")
        if not ensure_reference(long_mode):
            log_done("（缺台词文本，提前退出）")
            return
        log_step("[1/2] 长音频逐句评测（发音/音变/节奏/重音）")
        if run_script("compare_long.py") != 0:
            out("X 长音频评测失败，详见上方日志")
            log_done("（评测失败）")
            return
    else:
        log_step("[1/3] 音素对比（发音 + 音变）")
        # 单句模式没有台词文本，不能自动重生成参考音素表 —— 但能检测「过期」：
        # 换了新音频却没重生成 → 词标签/连读区/节奏词块全按旧句子边界算，
        # 音素分照对但词级结果全错，且不报错（最危险的静默错误）。必须警告。
        if (not os.path.exists(REF_PHONEME_JSON)
                or not newer_than(REF_PHONEME_JSON, REF_WAV)):
            out("⚠ 参考音素.json 比原音频旧（或不存在）—— 单句模式不会自动重新生成。")
            out("   如果这段是**新句子**，请先跑：")
            out("       python scripts\\gen_ref_phonemes.py \"新句子的文本\"")
            out("   否则词标签 / 节奏词块会按旧句子的边界算，结果不可信。")
            out("   （如果还是原来那句，忽略此提示。）")
        if run_script("compare_phoneme.py") != 0:
            out("X 音素对比失败，详见上方日志")
            log_done("（评测失败）")
            return
        log_step("[2/3] 节奏 / 重音")
        if run_script("compare_rhythm.py") != 0:
            out("X 节奏对比失败，详见上方日志")
            log_done("（评测失败）")
            return

    log_step("[%s] 综合评分 + 报告" % ("2/2" if long_mode else "3/3"))
    if run_script("overall_score.py") != 0:
        out("X 综合评分失败，详见上方日志")
        log_done("（评分失败）")
        return

    out("")
    out("=" * 62)
    out("全部完成。结果在 output\\：")
    out("    还原度综合报告.txt   ← 先看这份（人看）")
    out("    音素对比_报告.txt    ← 逐句/逐音明细")
    if long_mode:
        out("    （长音频模式：逐句明细 + 漏句清单见 音素对比_报告.txt）")
    out("=" * 62)
    log_done("一键跑还原度")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log_error(traceback.format_exc())
    try:
        input("按回车退出 ...")
    except Exception:
        pass
