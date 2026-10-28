# -*- coding: utf-8 -*-
r"""一键跑还原度 · 纯硬性开关（默认长文本，--单句 走单句）

被 长文本模式一键跑还原度.bat / 单句模式一键跑还原度.bat 调用。

模式判定（10-28 改：删掉「按时长自动路由」，不再拿 20 秒猜单句/长文本）：
  默认（无参数）→ 长文本模式：
      1. 每次先调讯飞转写取台词文本当真值（不依赖手动放 转写文本.txt）
         - 转写空/失败 → 报「没识别到文本」直接退出
      2. 转写文本做轻量清理（andWhat → and What 这类粘连）后生成参考音素表
      3. compare_long.py（四维度逐句评测）
  --单句 / --sentence → 单句模式：compare_phoneme → compare_rhythm
  最后都跑 overall_score.py

用法：python scripts\run_all.py              # 长文本（切句）
     python scripts\run_all.py --单句        # 单句
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

sys.path.insert(0, HERE)
from logkit import log, out, log_setup, log_done, log_step, log_error
from audio_paths import find_audio_pair, missing_hint, short_name

# 音频格式无关：mp3 / wav / flac / ogg 都认，认不出返回 None
REF_WAV, MY_WAV = find_audio_pair(SRC_DIR)


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


def ensure_reference(long_mode):
    """长文本（切句）模式：每次先调讯飞转写当真值，再生成参考音素表。

    返回 True=就绪，False=没识别到文本（调用方应直接退出）。

    为什么每次转写、不缓存（10-28 定）：
      这是要做成「AI 纠音程序」，不能依赖用户手动准备 转写文本.txt；
      台词文本必须程序自己转写、自己判断。转写空 → 报「没识别到文本」退出。
    """
    log_step("讯飞转写（取台词文本当真值）")
    if not os.path.exists(XUNFEI_CONFIG):
        out("")
        out("X 没有 讯飞配置.json，无法自动转写。")
        out("  请在项目根目录放 讯飞配置.json"
            "（{\"app_id\":\"..\",\"secret_key\":\"..\"}），再重跑。")
        return False
    if run_script("xunfei_transcribe.py") != 0:
        out("")
        out("X 没识别到文本（转写失败或音频里没有可识别的语音）。")
        out("  请检查：1) 音频是否正常、有人声；2) 网络与 讯飞配置.json 是否有效。")
        return False
    text = None
    if os.path.exists(TRANSCRIBE_TXT):
        with open(TRANSCRIBE_TXT, encoding="utf-8") as f:
            text = f.read().strip()
    if not text:
        out("")
        out("X 没识别到文本（转写结果为空）。")
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
    out("   还原度测试 · 一键跑（默认长文本，--单句 走单句）")
    out("=" * 62)

    if not (REF_WAV and MY_WAV):
        out("X 没找到音频文件，请把两个音频放进 audio\\：")
        out(missing_hint([("原音频", REF_WAV), ("模仿", MY_WAV)], SRC_DIR))
        log_done("（缺音频，提前退出）")
        return

    # 纯硬性开关（10-28 改）：不再拿时长猜「单句 / 长文本」。
    #   默认走长文本（切句）模式；--单句 / --sentence 强制单句。
    #   理由：一句读慢也可能超过 20 秒，时长不是可靠的判据；做成纠音程序后
    #   该由用户明确指定，而不是程序猜。
    force_single = ("--单句" in sys.argv) or ("--sentence" in sys.argv)
    long_mode = not force_single

    dur = audio_duration(REF_WAV) or 0.0
    mode_name = "单句模式" if force_single else "长文本模式（切句评测）"
    out("输入音频：%s（%.1fs） vs %s" % (short_name(REF_WAV), dur, short_name(MY_WAV)))
    out("本次模式：%s" % mode_name)

    if long_mode:
        log_step("[1/2] 长文本逐句评测（发音/音变/节奏/重音）")
        if not ensure_reference(long_mode):
            log_done("（没识别到文本，提前退出）")
            return
        if run_script("compare_long.py") != 0:
            out("X 长文本评测失败，详见上方日志")
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
        out("    （长文本模式：逐句明细 + 漏句清单见 音素对比_报告.txt）")
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
