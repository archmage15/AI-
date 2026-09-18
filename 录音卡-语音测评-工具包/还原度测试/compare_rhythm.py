# -*- coding: utf-8 -*-
"""还原度对比 · 第一块：节奏 / 语调

拿「原音频」和「你的模仿录音」做音频对音频的对比，看：
  - 总时长（语速是不是差不多）
  - 停顿：几处、在哪、多长
  - 能量包络：重音落在哪
  - 音高走向（F0）
  - 总体节奏相似度（DTW 对齐后算）

用法：
  1. 把两个音频放进本文件夹，命名成：原音频.wav 和 模仿.wav
     （支持 wav；别的格式请先转成 wav）
  2. 双击 运行节奏对比.bat
  3. 屏幕出报告，并生成 节奏对比.png（两条曲线叠着看）

依赖：soundfile / numpy / matplotlib（都已装好）
"""
import os
import sys
import wave

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REF_FILE = os.path.join(HERE, "原音频.wav")
MY_FILE = os.path.join(HERE, "模仿.wav")
OUT_PNG = os.path.join(HERE, "节奏对比.png")

SR = 16000
FRAME_MS = 25
HOP_MS = 10
PAUSE_MIN_S = 0.12      # 超过这个时长的静音才算"停顿"


def log(msg=""):
    print(str(msg), flush=True)


# ---------------- 读音频 + 变单声道 + 重采样 ----------------
def load_mono(path):
    if not os.path.exists(path):
        return None
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(audio) * SR / sr))
        x_in = np.linspace(0, 1, len(audio), endpoint=False)
        x_out = np.linspace(0, 1, n_out, endpoint=False)
        audio = np.interp(x_out, x_in, audio)
    return audio.astype(np.float32)


# ---------------- 能量包络 ----------------
def frame_rms(y):
    fl = int(SR * FRAME_MS / 1000)
    hl = int(SR * HOP_MS / 1000)
    if len(y) < fl:
        return np.zeros(1)
    n = 1 + (len(y) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    return np.sqrt((y[idx] ** 2).mean(axis=1))


# ---------------- 停顿检测 ----------------
def find_pauses(env):
    thr = 0.12 * env.max() if env.max() > 0 else 0
    silent = env < thr
    hop_s = HOP_MS / 1000.0
    out = []
    i = 0
    while i < len(silent):
        if silent[i]:
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            dur = (j - i) * hop_s
            if dur >= PAUSE_MIN_S:
                out.append((i * hop_s, dur))
            i = j
        else:
            i += 1
    return out


# ---------------- 音高 F0（自相关，粗略） ----------------
def pitch_contour(y, fmin=70, fmax=400):
    fl = int(SR * 0.04)
    hl = int(SR * HOP_MS / 1000)
    lag_min = int(SR / fmax)
    lag_max = int(SR / fmin)
    f0 = []
    for i in range(0, max(1, len(y) - fl), hl):
        fr = y[i:i + fl]
        if len(fr) < fl:
            break
        fr = fr - fr.mean()
        corr = np.correlate(fr, fr, mode="full")[len(fr) - 1:]
        if corr[0] <= 0 or lag_max >= len(corr):
            f0.append(0.0)
            continue
        seg = corr[lag_min:lag_max]
        if len(seg) == 0:
            f0.append(0.0)
            continue
        peak = int(np.argmax(seg)) + lag_min
        if corr[peak] / corr[0] < 0.3:
            f0.append(0.0)
        else:
            f0.append(SR / peak)
    return np.array(f0)


# ---------------- DTW 距离 ----------------
def dtw_distance(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    b = (b - b.min()) / (b.max() - b.min() + 1e-9)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / max(n, m)


# ---------------- 画图 ----------------
def plot(ref, mine, env_r, env_m, f0_r, f0_m, pauses_r, pauses_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    hop = HOP_MS / 1000.0
    t_r = np.arange(len(env_r)) * hop
    t_m = np.arange(len(env_m)) * hop

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, env, f0, pauses, title in (
            (axes[0], env_r, f0_r, pauses_r, "原音频（标准）"),
            (axes[1], env_m, f0_m, pauses_m, "我的模仿")):
        ax.plot(t_r if title.startswith("原") else t_m, env, color="#c0392b" if title.startswith("原") else "#2471a3",
                linewidth=1.2, label="能量（响度）")
        ax.set_ylabel("能量", color="#555")
        ax.grid(alpha=0.25)
        ax.set_title(title, fontsize=12)
        for (st, dur) in pauses:
            ax.axvspan(st, st + dur, color="gray", alpha=0.22)
        ax2 = ax.twinx()
        tt = t_r if title.startswith("原") else t_m
        n = min(len(tt), len(f0))
        if n > 0:
            f0v = np.where(f0[:n] == 0, np.nan, f0[:n])
            ax2.plot(tt[:n], f0v, color="#7d3c98", linewidth=1.0, alpha=0.8, label="音高 F0")
        ax2.set_ylabel("音高 (Hz)", color="#7d3c98")
        ax2.set_ylim(60, 400)

    axes[1].set_xlabel("时间 (秒)   ·   灰色块 = 停顿")
    fig.suptitle("节奏 / 语调 对比", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    log("图已保存：" + OUT_PNG)


def main():
    log("=" * 62)
    log("   还原度对比 · 节奏 / 语调")
    log("=" * 62)

    ref = load_mono(REF_FILE)
    mine = load_mono(MY_FILE)
    if ref is None or mine is None:
        log("")
        log("X 没找到音频文件。请把两个 wav 放进本文件夹：")
        log("    " + REF_FILE)
        log("    " + MY_FILE)
        return

    dur_r = len(ref) / SR
    dur_m = len(mine) / SR
    env_r, env_m = frame_rms(ref), frame_rms(mine)
    f0_r, f0_m = pitch_contour(ref), pitch_contour(mine)
    p_r, p_m = find_pauses(env_r), find_pauses(env_m)
    dist = dtw_distance(env_r, env_m)
    sim = max(0.0, (1 - dist) * 100)

    log("")
    log("总时长：原音频 %.2fs   |   我的 %.2fs   |   差 %+.2fs"
        % (dur_r, dur_m, dur_m - dur_r))
    log("停顿数量：原音频 %d 处   |   我的 %d 处" % (len(p_r), len(p_m)))
    if p_r:
        log("  原音频停顿：" + "，".join("%.1fs处停%.2fs" % (s, d) for s, d in p_r))
    if p_m:
        log("  我的停顿  ：" + "，".join("%.1fs处停%.2fs" % (s, d) for s, d in p_m))
    log("")
    log("节奏相似度（DTW，0-100，越高越像）：%.1f" % sim)
    log("")
    log("怎么看：")
    log("  - 相似度高 → 你的节奏基本跟上了原音频")
    log("  - 相似度低 / 停顿位置对不上 → 这块要练")
    log("")

    try:
        plot(ref, mine, env_r, env_m, f0_r, f0_m, p_r, p_m)
    except Exception as e:
        log("（画图失败，不影响上面数据：%s）" % e)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("[致命错误]")
        log(traceback.format_exc())
    try:
        input("按回车退出 ...")
    except Exception:
        pass
