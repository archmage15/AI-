# -*- coding: utf-8 -*-
"""离线预生成参考音素表 —— 调 espeak-ng，产出纯数据 JSON。

【为什么必须"离线预生成"】
espeak-ng 是 GPL-3.0（copyleft，有传染性）。按计划书决策 3：
运行时脚本**绝不 import / 链接**它 —— 只在本机离线跑一次，把结果落成
纯数据文件（参考音素.json），之后所有脚本都只读这个 JSON。

【输入】参考文本，必须写**标准形式**（`want to` 而不是 `wanna`）。
    角色用 | 分隔 —— | 之前是角色 A，之后是角色 B（对齐用，见计划书 §5.5）。
    例：  so Where are you working now Terry|oh im still at the bank

【输出】..\audio\参考音素.json
    {
      "文本": "...",
      "音素": ["s", "oʊ", ...],                              # 整句音素序列
      "词边界": [{"word": "so", "a": 0, "b": 2, "角色": "A"}, ...]
    }                                                        # a/b 是音素下标

【用法】
    python gen_ref_phonemes.py --probe              # 只看 espeak 原始输出（换文本时用）
    python gen_ref_phonemes.py                      # 用 DEFAULT_TEXT 生成
    python gen_ref_phonemes.py "your text here"     # 指定文本

【⚠️ 为什么是"整句跑"而不是"逐词跑"（09-24 实测确定）】
espeak-ng 的 `--ipa` 输出**按空格分的是"词"，词内音素是连续串、无分隔符**，
且**重音标记 ˈ ˌ 夹在音素中间**。实测整句跑的输出段数与输入词数一一对应：
    'so Where are you working now'  →  'sˌoʊ wˌɛɹ ɑːɹ juː wˈɜːkɪŋ nˈaʊ'   (6 段对 6 词)
所以整句跑一次就能同时拿到「词边界」和「句中真实发音」——
比逐词调用更准（逐词会把 so 读成 sˈoʊ，句中其实是 sˌoʊ，重音位置都不同）。
词内音素切分用**最长匹配**，词典直接取模型自己的符号表 ——
保证切出来的 token 一定在模型词表里，后续 diff 不会因符号集不同而误配。
"""
import json
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
OUT_JSON = os.path.join(ROOT, "audio", "参考音素.json")
MODEL_DIR = os.path.join(ROOT, "models", "wav2vec2-espeak")
VOCAB_JSON = os.path.join(MODEL_DIR, "vocab.json")

DEFAULT_TEXT = "so Where are you working now Terry|oh im still at the bank"

# espeak-ng 可执行文件的候选位置（msi 装完通常不写 PATH）
CANDIDATES = [
    "espeak-ng",
    r"C:\Program Files\eSpeak NG\espeak-ng.exe",
    r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
]

# 从 espeak 输出里剥掉的"非音素"标记：主重音 / 次重音 / 撇号
STRIP_CHARS = "ˈˌ'"

# espeak 的 -v 音色（美音，与现有素材一致）
VOICE = "en-us"

# 模型词表里的特殊 token，不能当切分词典
SPECIAL_TOKENS = {"<pad>", "<s>", "</s>", "<unk>", "??"}


def find_espeak():
    """返回 (可执行路径, 版本号)。找不到返回 (None, None)。"""
    for cand in CANDIDATES:
        try:
            r = subprocess.run([cand, "--version"], capture_output=True, timeout=15)
            if r.returncode == 0:
                return cand, r.stdout.decode("utf-8", "replace").strip()
        except Exception:
            continue
    return None, None


def raw_phonemize(exe, text):
    """原样返回 espeak-ng 的 IPA 输出（不做解析）—— 供 probe 用。"""
    r = subprocess.run([exe, "-q", "--ipa", "-v", VOICE, text],
                       capture_output=True, timeout=20)
    return (r.returncode,
            r.stdout.decode("utf-8", "replace"),
            r.stderr.decode("utf-8", "replace"))


def load_keyset():
    """读模型符号表，返回 (词典集合, 最长 token 长度, 全集) —— 切分用。"""
    with open(VOCAB_JSON, encoding="utf-8") as f:
        vocab = json.load(f)
    keyset = {k for k in vocab.keys() if k not in SPECIAL_TOKENS and k.strip()}
    max_len = max((len(k) for k in keyset), default=1)
    return keyset, max_len, set(vocab.keys())


def strip_marks(s):
    for ch in STRIP_CHARS:
        s = s.replace(ch, "")
    return s


def split_phonemes(s, keyset, max_len):
    """把"词内连续音素串"切成音素 token —— 动态规划，保证每个 token 都在词典里。

    ⚠️ 为什么不能用贪心最长匹配：实测 "juː" 会被切成 "ju" + "ː"，
       而 "ː" 不在模型符号表里 —— 贪心不会回溯。DP 会退一步试 "j" + "uː" ✓

    无可行切分时返回 (单字符列表, 那些字符)：不静默丢字符 ——
    制造"看起来对、其实漏了音"的假象，比直接报出来危险得多。
    """
    n = len(s)
    memo = {}

    def solve(i):
        if i >= n:
            return ()
        if i in memo:
            return memo[i]
        memo[i] = None
        for L in range(min(max_len, n - i), 0, -1):
            t = s[i:i + L]
            if t in keyset:
                rest = solve(i + L)
                if rest is not None:
                    memo[i] = (t,) + rest
                    return memo[i]
        return None

    res = solve(0)
    if res is None:
        return [c for c in s], list(s)
    return list(res), []


def clean_word(w):
    """去掉首尾标点，只留字母与撇号。"""
    return re.sub(r"^[^\w']+|[^\w']+$", "", w, flags=re.UNICODE)


def no_espeak():
    print("X 找不到 espeak-ng。请先安装，或把它所在目录加进 PATH。候选位置：")
    for c in CANDIDATES:
        print("    " + c)
    return 1


def probe(exe, ver):
    print("espeak-ng : %s" % exe)
    print("version   : %s" % ver)
    print()
    for w in ["so", "working", "Terry", "bank", "the", "think"]:
        rc, out, err = raw_phonemize(exe, w)
        print("  %-9s rc=%d  repr=%s" % (w, rc, repr(out)))
    rc, out, err = raw_phonemize(exe, "so Where are you working now")
    print()
    print("  整句 repr=%s" % repr(out))
    return 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    exe, ver = find_espeak()
    if exe is None:
        return no_espeak()

    if "--probe" in sys.argv:
        return probe(exe, ver)

    print("espeak-ng : %s" % exe)
    print("version   : %s" % ver.strip())

    text = args[0] if args else DEFAULT_TEXT
    print("文本      : %s" % text)
    print()

    keyset, max_len, vocab_all = load_keyset()
    print("切分词典  : 模型符号表 %d 项（已排除特殊 token，最长 %d 字符）" % (len(keyset), max_len))
    print()

    segments = [s.strip() for s in text.split("|") if s.strip()]
    role_names = "ABCDEFGH"

    phonemes, word_bounds = [], []
    all_unknown, problems = [], []
    sentences = []          # 句子边界：[{"句号":1, "文本":"...", "a":0, "b":32, "角色":"A"}]
                          # a/b 是音素下标（闭开区间）。用于长音频按句评测：
                          # 参考序列按句切段，模仿音素按内容对齐进各句区间。

    for si, seg in enumerate(segments):
        role = role_names[si] if si < len(role_names) else "?"
        words = [w for w in (clean_word(x) for x in seg.split()) if w]
        # 按句末标点（. ? ! ；;）把词切成句子 —— 只切文本，不用任何时间戳。
        # ⚠️ 必须在**原始词**（未剥标点）上判断句末：clean_word 已把标点去掉，
        #    拿剥完的词判断会一句都切不出来（实测踩过）。
        raw_words = [x for x in seg.split() if clean_word(x)]
        sent_groups, cur = [], []
        for rw in raw_words:
            cur.append(clean_word(rw))
            if rw and rw[-1] in ".?!;":
                sent_groups.append(cur)
                cur = []
        if cur:
            sent_groups.append(cur)
        rc, out, err = raw_phonemize(exe, seg)
        if rc != 0:
            print("X espeak-ng 失败：%s" % err.strip())
            return 2
        parts = strip_marks(out).split()

        if len(parts) != len(words):
            # ⚠️ 段数对不上：espeak 可能把两个词连读了（如 want to -> wɒntə）
            problems.append("角色 %s：输入 %d 词，espeak 输出 %d 段 —— 对应关系不明"
                            % (role, len(words), len(parts)))
        print("角色 %s：%d 词，%d 句" % (role, len(words), len(sent_groups)))

        # 词 -> 句号 映射（按词序累加，与 espeak 段错位时也保持句子结构）
        word2sent = {}
        wi_sent = 0
        for gi, g in enumerate(sent_groups):
            for _ in g:
                if wi_sent < len(words):
                    word2sent[wi_sent] = gi
                    wi_sent += 1

        for wi, w in enumerate(words):
            if wi < len(parts):
                toks, unk = split_phonemes(parts[wi], keyset, max_len)
                all_unknown.extend(unk)
            else:
                toks, unk = [], []
            a = len(phonemes)
            phonemes.extend(toks)
            b = len(phonemes)
            gi = word2sent.get(wi, len(sent_groups) - 1 if sent_groups else 0)
            word_bounds.append({"word": w, "a": a, "b": b, "角色": role,
                                "句号": gi + 1})
            print("  %-10s %s" % (w, " ".join(toks) if toks else "（无）"))

        # 汇总每句的音素区间
        for gi, g in enumerate(sent_groups):
            idxs = [wi for wi in range(len(words)) if word2sent.get(wi) == gi]
            if not idxs:
                continue
            # 直接用该句首尾词的音素下标
            first_wb = word_bounds[len(word_bounds) - len(words) + idxs[0]]
            last_wb = word_bounds[len(word_bounds) - len(words) + idxs[-1]]
            sentences.append({"句号": len(sentences) + 1,
                              "文本": " ".join(g),
                              "a": first_wb["a"], "b": last_wb["b"],
                              "角色": role})

    payload = {"文本": text, "音素": phonemes, "词边界": word_bounds, "句子边界": sentences}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print()
    print("共 %d 个词，%d 个音素" % (len(word_bounds), len(phonemes)))
    print("已写出：%s" % OUT_JSON)

    # ---- 两道自检 ----
    print()
    missing = sorted(set(phonemes) - vocab_all)
    if missing:
        print("⚠️ 以下符号模型符号表里没有（会影响对比，需 normalize_tok 处理）：")
        print("   " + "  ".join(missing))
    else:
        print("✓ 用到的音素符号模型表里全有")

    if all_unknown:
        print("⚠️ 切分时遇到词典外的字符（已按单字符保留）：")
        print("   " + "  ".join(sorted(set(all_unknown))))

    if problems:
        print()
        print("⚠️ 词/段对应问题（需人工确认）：")
        for p in problems:
            print("   " + p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
