# -*- coding: utf-8 -*-
"""麦克风音量检测：找出哪个设备真的能收到你的声音。

双击 miclevel.bat 运行。全程约 10 秒，请【一直大声说话】。
"""
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import sounddevice as sd

TARGETS = [1, 2, 3, 4]
DUR = 2.0
W = 68

print("=" * W)
print("   麦克风音量检测")
print("=" * W)
print("接下来约 10 秒，请【一直大声说话】，中途不要停。")
print("（念这段话、数数、随便说什么都行）")
print()
for i in (3, 2, 1):
    print("   倒计时", i, "...")
    time.sleep(1)
print()
print(">>> 开始！请说话！<<<")
print()

results = []
for idx in TARGETS:
    try:
        info = sd.query_devices(idx)
        name = str(info["name"])
        sr = int(info["default_samplerate"])
    except Exception as e:
        print("  跳过设备", idx, ":", e)
        continue
    try:
        rec = sd.rec(int(DUR * sr), samplerate=sr, channels=1, dtype="float32", device=idx)
        sd.wait()
        data = np.asarray(rec).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        results.append((idx, name, rms))
        print("  设备 %-2d %-44s RMS=%.5f  峰值=%.3f" % (idx, name[:44], rms, peak))
    except Exception as e:
        print("  设备 %-2d %-44s 打开失败：%s" % (idx, name[:44], e))

print()
print("=" * W)
print("结果：")
best = None
for idx, name, rms in results:
    if rms > 0.01:
        tag = "   <<< 有声音！就是它"
    elif rms > 0.002:
        tag = "   (有轻微信号)"
    else:
        tag = "   (静音)"
    print("  %-44s RMS=%.5f%s" % (name[:44], rms, tag))
    if best is None or rms > best[2]:
        best = (idx, name, rms)
print()
if best and best[2] > 0.01:
    print("信号最强的设备：", best[1])
    print("=> 你说话真正被听到的是这个麦克风。")
else:
    print("所有设备都没收到明显声音。")
    print("=> 检查：麦克风插好没 / 系统里是否被静音 / 离嘴是不是太远。")
print("=" * W)
