# -*- coding: utf-8 -*-
r"""还原度对比 · 第一块：节奏 / 语调

拿「原音频」和「你的模仿录音」做音频对音频的对比，看：
  - 总时长（语速是不是差不多）
  - 停顿：几处、在哪、多长
  - 能量包络：重音落在哪
  - 音高走向（F0）
  - 总体节奏相似度（DTW 对齐后算）

用法：
  1. 把两个音频放进 audio\ 文件夹，命名成：原音频.wav 和 模仿.wav
     （支持 wav；别的格式请先转成 wav）
  2. 双击 scripts\run_rhythm.bat（或根目录的 一键跑还原度.bat）
  3. 屏幕出报告，结果写到 output\ 文件夹

目录约定（本脚本在 scripts\ 里，用 ROOT 指回上一层）：
  audio\   放录音（你手动放，只读）
  output\  放本次结果（自动生成，可随时清空）

依赖：soundfile / numpy / matplotlib（都已装好）
"""
import json
import os
import sys
import wave

import numpy as np

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

REF_FILE = os.path.join(SRC_DIR, "原音频.wav")
MY_FILE = os.path.join(SRC_DIR, "模仿.wav")
OUT_PNG = os.path.join(OUT_DIR, "节奏对比.png")
OUT_JSON = os.path.join(OUT_DIR, "节奏对比.json")   # 给机器/综合评分器用

SR = 16000
FRAME_MS = 25
HOP_MS = 10
PAUSE_MIN_S = 0.12      # 超过这个时长的静音才算"停顿"
SIL_RATIO = 0.12        # 能量低于 max*该比例算静音（与 find_pauses 一致）

# ---- 对齐参数：先把两条时间轴"卡齐"再比，避免整体延迟污染 DTW ----
VAD_RATIO = 0.10        # 首尾裁剪阈值：能量低于 max*该比例算静音
MAX_SHIFT_S = 2.0       # 互相关允许搜索的最大平移（秒），防误对齐

# ---- 评分权重（节奏分内部）----
W_DUR, W_PAUSE, W_ENV = 0.35, 0.35, 0.30


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


# ---------------- 时间轴对齐：VAD 裁静音 + 互相关平移 ----------------
def vad_trim(env):
    """用能量阈值裁掉首尾静音，返回 (起帧, 止帧)。
    这是对齐的第一步：模仿者"反应慢半拍"会在开头留一段静音，
    不裁掉的话会被 DTW 当成"节奏不像"。"""
    if len(env) == 0 or env.max() <= 0:
        return 0, len(env)
    thr = VAD_RATIO * env.max()
    voiced = np.where(env > thr)[0]
    if len(voiced) == 0:
        return 0, len(env)
    return int(voiced[0]), int(voiced[-1]) + 1


def cross_corr_shift(a, b, max_shift):
    """互相关求 b 相对 a 的整体平移帧数（正数= b 落后 a）。
    只搜 ±max_shift 帧，取相关系数最大的位移 —— 这是"卡点对齐"。
    与 DTW（管伸缩）互补：互相关管"谁先谁后"。"""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    best_lag, best_score = 0, -np.inf
    lo, hi = -max_shift, max_shift
    for lag in range(lo, hi + 1):
        if lag >= 0:
            x, y = a[lag:], b[:n - lag]
        else:
            x, y = a[:n + lag], b[-lag:]
        if len(x) < 4:
            continue
        denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
        score = float((x * y).sum() / denom) if denom > 0 else 0.0
        if score > best_score:
            best_score, best_lag = score, lag
    return best_lag


def align_pair(env_r, env_m):
    """把两条能量包络对齐：先各自 VAD 裁首尾，再互相关求平移，返回对齐后的两段。"""
    r0, r1 = vad_trim(env_r)
    m0, m1 = vad_trim(env_m)
    er, em = env_r[r0:r1], env_m[m0:m1]
    if len(er) == 0 or len(em) == 0:
        return er, em, 0
    max_shift = int(MAX_SHIFT_S * 1000 / HOP_MS)
    lag = cross_corr_shift(er, em, max_shift)
    # 按 lag 裁掉各自多出的头/尾，使两条起点对齐
    if lag > 0:            # 模仿落后 → 去掉模仿开头 lag 帧
        em = em[lag:]
        er = er[:len(em)]
    elif lag < 0:          # 模仿超前 → 去掉原音频开头 |lag| 帧
        er = er[-lag:]
        em = em[:len(er)]
    n = min(len(er), len(em))
    return er[:n], em[:n], lag


# ---------------- 各子项相似度（0-100） ----------------
def sim_duration(len_r, len_m, hop_s):
    """时长相似度：两条语音净时长（裁静音后）差得越小分越高。"""
    dr, dm = len_r * hop_s, len_m * hop_s
    if max(dr, dm) <= 1e-6:
        return 100.0, dr, dm
    return max(0.0, (1 - abs(dr - dm) / max(dr, dm)) * 100), dr, dm


def sim_pause(p_r, p_m):
    """停顿相似度：先比个数；个数相同再比位置/时长（一一配对）。"""
    if not p_r and not p_m:
        return 100.0
    n = max(len(p_r), len(p_m))
    cnt = max(0.0, (1 - abs(len(p_r) - len(p_m)) / n) * 100)
    if len(p_r) == len(p_m) and p_r:
        diffs = [abs(sr - sm) + abs(dr - dm)
                 for (sr, dr), (sm, dm) in zip(p_r, p_m)]
        pos = max(0.0, (1 - np.mean(diffs) / 0.5) * 100)   # 0.5s 差 → 0 分
        return 0.5 * cnt + 0.5 * pos
    return cnt


def sim_dtw(a, b):
    """DTW 相似度（0-100）：只负责"伸缩对齐"下的形状相似。"""
    d = dtw_distance(a, b)
    if d != d:      # nan
        return 0.0
    return max(0.0, (1 - d) * 100)


def sim_f0(f0_r, f0_m):
    """语调相似度：对 F0 轮廓做 DTW。
    0 值(未检测到浊音)当掩码去掉；取对数尺度（人耳对音高是 log 感知）；
    再比整体音高起伏幅度（std），避免"都能唱但音域差很多"被漏判。"""
    vr = f0_r[f0_r > 0]
    vm = f0_m[f0_m > 0]
    if len(vr) < 4 or len(vm) < 4:
        return 0.0, 0.0
    lr = np.log(vr)
    lm = np.log(vm)
    shape = sim_dtw(lr, lm)
    # 起伏幅度相似度
    sr, sm_ = lr.std(), lm.std()
    amp = max(0.0, (1 - abs(sr - sm_) / (max(sr, sm_) + 1e-9)) * 100)
    return 0.5 * shape + 0.5 * amp, amp


def write_json(dur_r, dur_m, p_r, p_m, env_sim, dur_sim, pause_sim,
               rhythm_score, f0_sim, amp_sim, tone_score, lag_frames, hop_s):
    """写 节奏对比.json —— 给机器/综合评分器读，不再只打印屏幕。"""
    data = {
        "总时长秒": {"原音频": round(dur_r, 3), "我的": round(dur_m, 3),
                     "差": round(dur_m - dur_r, 3)},
        "对齐": {"整体平移帧": lag_frames, "整体平移秒": round(lag_frames * hop_s, 3)},
        "停顿": {"原音频处数": len(p_r), "我的处数": len(p_m),
                 "原音频": [[round(s, 2), round(d, 2)] for s, d in p_r],
                 "我的": [[round(s, 2), round(d, 2)] for s, d in p_m]},
        "子项分": {"时长相似": round(dur_sim, 1), "停顿相似": round(pause_sim, 1),
                   "能量DTW相似": round(env_sim, 1), "F0起伏幅度": round(amp_sim, 1)},
        "节奏分": round(rhythm_score, 1),
        "语调分": round(tone_score, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    hop_s = HOP_MS / 1000.0
    dur_r = len(ref) / SR
    dur_m = len(mine) / SR

    env_r_raw, env_m_raw = frame_rms(ref), frame_rms(mine)
    f0_r, f0_m = pitch_contour(ref), pitch_contour(mine)

    # 先对齐（裁静音 + 互相关平移），再在各子项上算相似度
    env_r, env_m, lag = align_pair(env_r_raw, env_m_raw)
    p_r, p_m = find_pauses(env_r), find_pauses(env_m)

    env_sim = sim_dtw(env_r, env_m)                       # 能量包络 DTW（伸缩对齐）
    dur_sim, net_r, net_m = sim_duration(len(env_r), len(env_m), hop_s)
    pause_sim = sim_pause(p_r, p_m)
    rhythm_score = W_DUR * dur_sim + W_PAUSE * pause_sim + W_ENV * env_sim

    # 语调：F0 轮廓 DTW（截到对齐后的较短长度，避免尾部错位）
    n_f0 = min(len(f0_r), len(f0_m))
    tone_score, amp_sim = sim_f0(f0_r[:n_f0], f0_m[:n_f0])

    log("")
    log("总时长：原音频 %.2fs   |   我的 %.2fs   |   差 %+.2fs"
        % (dur_r, dur_m, dur_m - dur_r))
    log("对齐：模仿相对原音频整体平移 %+.2fs（已自动校正）" % (lag * hop_s))
    log("停顿数量：原音频 %d 处   |   我的 %d 处" % (len(p_r), len(p_m)))
    if p_r:
        log("  原音频停顿：" + "，".join("%.1fs处停%.2fs" % (s, d) for s, d in p_r))
    if p_m:
        log("  我的停顿  ：" + "，".join("%.1fs处停%.2fs" % (s, d) for s, d in p_m))
    log("")
    log("节奏分：%.1f / 100   （时长 %.1f ｜ 停顿 %.1f ｜ 能量DTW %.1f）"
        % (rhythm_score, dur_sim, pause_sim, env_sim))
    log("语调分：%.1f / 100   （F0轮廓+起伏幅度）" % tone_score)
    log("")
    log("怎么看：")
    log("  - 节奏分高 → 语速/停顿基本跟上了原音频")
    log("  - 语调分高 → 音高起伏走向像原音频")
    log("")

    # 画图（用对齐前的原始包络，保留"对人看"的完整时间轴）
    try:
        plot(ref, mine, env_r_raw, env_m_raw, f0_r, f0_m,
             find_pauses(env_r_raw), find_pauses(env_m_raw))
    except Exception as e:
        log("（画图失败，不影响上面数据：%s）" % e)

    try:
        write_json(dur_r, dur_m, p_r, p_m, env_sim, dur_sim, pause_sim,
                   rhythm_score, tone_score, amp_sim, tone_score, lag, hop_s)
        log("数据（机器）：节奏对比.json")
    except Exception as e:
        log("（写 json 失败：%s）" % e)


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
