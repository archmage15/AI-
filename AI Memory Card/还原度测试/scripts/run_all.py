# -*- coding: utf-8 -*-
r"""一键跑还原度 · 统一走长文本逐句评测（单句 = 长文本切 1 句）

被 长文本模式一键跑还原度.bat / 单句模式一键跑还原度.bat 调用。

模式（10-28 并轨：删掉独立的单句链路，单句并入长文本）：
  所有音频统一走：
      1. 每次先调讯飞转写取台词文本当真值（不依赖手动放 转写文本.txt）
         - 转写空/失败 → 报「没识别到文本」直接退出
      2. 转写文本做轻量清理（andWhat → and What 这类粘连）后生成参考音素表
      3. compare_long.py 按标点切句、逐句评测（一句文本切出 1 句，自然只跑 1 次）
  最后跑 overall_score.py

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

REF_PHONEME_JSON = os.path.join(SRC_DIR, "参考音素.json")
TRANSCRIBE_TXT = os.path.join(SRC_DIR, "转写文本.txt")
XUNFEI_CONFIG = os.path.join(ROOT, "讯飞配置.json")
REF_DIR = os.path.join(SRC_DIR, "原音频")
REF_CACHE_JSON = os.path.join(REF_DIR, "原音频缓存.json")   # 原音频分析缓存（10-28 加）

sys.path.insert(0, HERE)
from logkit import log, out, log_setup, log_done, log_step, log_error
from audio_paths import find_audio, missing_hint, short_name

# 音频格式无关：mp3 / wav / flac / ogg 都认，认不出返回 None
# 原音频可能放在 audio\原音频\ 子目录（10-28 拆缓存后）
REF_WAV = find_audio("原音频", REF_DIR) or find_audio("原音频", SRC_DIR)
MY_WAV = find_audio("模仿", SRC_DIR)


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


def ref_table_gen_version(path):
    """读参考音素.json 里记的「生成器版本」。

    读不到（旧表没有这个字段）→ 返回 None，意思是「未知版本」。
    """
    try:
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("生成器版本")
    except Exception:
        return None


def current_gen_version():
    """当前生成器的版本号（从 gen_ref_phonemes.py 里读，避免两处各写一份）。"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_gen_ref_probe", os.path.join(HERE, "gen_ref_phonemes.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)      # 只加载模块，不跑 main()
        return getattr(mod, "GEN_VERSION", None)
    except Exception:
        return None


def ref_table_is_current(path):
    """参考音素表是不是**当前代码**生成的（版本号一致）。

    为什么不能只看时间戳（10-28 实际踩到）：
      改了 gen_ref_phonemes.py 的生成逻辑之后，旧表的时间戳照样比音频新
      → 一直「跳过生成」→ 跑出来的是**旧逻辑的结果**，而且不报错。
      词边界错位那个 P0 bug 修完后，副本目录就一直在用修复前的表。

    规则：
      版本号一致 → 是当前的
      版本号不同 / 没有版本号（旧表）→ **不是当前的**，要重新生成
    """
    want = current_gen_version()
    if want is None:
        return True              # 读不到当前版本 → 不拦（别因为读不到就乱重生成）
    return ref_table_gen_version(path) == want


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


def ref_table_is_usable(path, ref_audio):
    """参考音素表能不能直接用？返回 (能用, 原因)。

    判定四条，**缺一不可**：
      1. 文件存在
      2. 带句子边界（长音频模式必需）
      3. 比原音频新
      4. 生成器版本号 == 当前代码的版本号（10-28 加）
         —— 只看时间戳不够：改了生成逻辑之后旧表照样「比音频新」，
            会一直跳过生成，跑出旧逻辑的结果且不报错。

    抽成纯函数（不碰文件系统之外的东西）才能被自检覆盖 ——
    放在 ensure_reference 里的话，变异测试根本测不到（10-28 踩过）。
    """
    if not os.path.exists(path):
        return False, "文件不存在"
    if not has_sentence_bounds(path):
        return False, "缺句子边界"
    if not newer_than(path, ref_audio):
        return False, "比原音频旧"
    if not ref_table_is_current(path):
        old = ref_table_gen_version(path)
        return False, ("旧生成器版本（表里 %s / 当前 %s）"
                       % ("缺版本号" if old is None else old, current_gen_version()))
    return True, "就绪"


def ensure_reference():
    """确保参考音素表就绪（优先用原音频缓存，跳过转写+生成）。

    返回 True=就绪，False=没缓存且无法生成（调用方应直接退出）。

    10-28 改：原音频分析结果已缓存（audio\\原音频\\原音频缓存.json）时，
    参考音素表/转写文本都是缓存里现成的，直接跳过讯飞转写和 espeak 生成。
    没缓存才需要先跑「分析原音频.bat」。
    """
    # 有缓存 → 直接复用（参考音素表已在缓存里，无需转写/生成）
    if os.path.exists(REF_CACHE_JSON):
        log("   有原音频缓存 → 跳过转写与参考音素表生成", show=True)
        return True
    # 没缓存：看看参考音素表是否已经在（旧流程残留），在也能跑，但要提示补缓存
    if os.path.exists(REF_PHONEME_JSON):
        log("   没缓存但已有参考音素表，继续（建议先跑 分析原音频.bat 建缓存）", show=True)
        return True
    out("")
    out("X 没找到原音频缓存，也没有参考音素表。")
    out("  请先双击 audio\\原音频\\分析原音频.bat，把原音频分析一遍生成缓存，")
    out("  之后一键跑还原度才会只分析模仿音频。")
    return False


def main():
    log_setup("一键路由")
    out("=" * 62)
    out("   还原度测试 · 一键跑（统一走长文本逐句评测）")
    out("=" * 62)

    if not (REF_WAV and MY_WAV):
        out("X 没找到音频文件，请把两个音频放进 audio\\：")
        out(missing_hint([("原音频", REF_WAV), ("模仿", MY_WAV)], SRC_DIR))
        log_done("（缺音频，提前退出）")
        return

    # 单句 = 长文本切出恰好 1 句的特例，不再有独立的单句链路（10-28 并轨）。
    #   所有音频统一走「讯飞转写 → 按标点切句 → compare_long 逐句评测」。
    #   一句文本切出来就是 1 句，逐句评测循环自然只跑 1 次，无需单独分支。
    dur = audio_duration(REF_WAV) or 0.0
    out("输入音频：%s（%.1fs） vs %s" % (short_name(REF_WAV), dur, short_name(MY_WAV)))
    out("本次模式：长文本逐句评测（单句/多句统一）")

    log_step("[1/2] 讯飞转写 + 逐句评测（发音/音变/节奏/重音）")
    if not ensure_reference():
        log_done("（没识别到文本，提前退出）")
        return
    if run_script("compare_long.py") != 0:
        out("X 逐句评测失败，详见上方日志")
        log_done("（评测失败）")
        return

    log_step("[2/2] 综合评分 + 报告")
    if run_script("overall_score.py") != 0:
        out("X 综合评分失败，详见上方日志")
        log_done("（评分失败）")
        return

    out("")
    out("=" * 62)
    out("全部完成。结果在 output\\：")
    out("    还原度综合报告.txt   ← 先看这份（人看）")
    out("    音素对比_报告.txt    ← 逐句/逐音明细")
    out("    （逐句明细 + 漏句清单见 音素对比_报告.txt）")
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
