# -*- coding: utf-8 -*-
"""读 Windows 麦克风隐私设置，判断 Python 有没有权限拿麦克风。"""
import sys
import winreg

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def rd(root, path, name):
    try:
        with winreg.OpenKey(root, path) as k:
            v, _ = winreg.QueryValueEx(k, name)
            return v
    except FileNotFoundError:
        return "(该项不存在)"
    except Exception as e:
        return "(读不到: %s)" % e


BASE = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"

print("=" * 52)
print("   麦克风隐私设置实测")
print("=" * 52)
print("机器级  总开关   :", rd(winreg.HKEY_LOCAL_MACHINE, BASE, "Value"))
print("用户级  总开关   :", rd(winreg.HKEY_CURRENT_USER, BASE, "Value"))
print("机器级  桌面应用 :", rd(winreg.HKEY_LOCAL_MACHINE, BASE + r"\NonPackaged", "Value"))
print("用户级  桌面应用 :", rd(winreg.HKEY_CURRENT_USER, BASE + r"\NonPackaged", "Value"))
print("组策略  限制     :", rd(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Policies\Microsoft\Windows\AppPrivacy", "LetAppsAccessMicrophone"))

print("-" * 52)
print("针对 Python 的单独授权记录：")
found = False
for root, tag in ((winreg.HKEY_CURRENT_USER, "HKCU"), (winreg.HKEY_LOCAL_MACHINE, "HKLM")):
    p = BASE + r"\NonPackaged"
    try:
        with winreg.OpenKey(root, p) as k:
            n = winreg.QueryInfoKey(k)[0]
            for i in range(n):
                name = winreg.EnumKey(k, i)
                if "python" in name.lower():
                    try:
                        with winreg.OpenKey(root, p + "\\" + name) as sk:
                            v, _ = winreg.QueryValueEx(sk, "Value")
                            t, _ = winreg.QueryValueEx(sk, "LastUsedTimeStart")
                    except Exception:
                        v, t = "?", "?"
                    print(" ", tag, name, "->", v, "| last:", t)
                    found = True
    except Exception:
        pass
if not found:
    print("  (没有针对 python 的单独设置，走上面的总开关)")

print("-" * 52)
print("Allow = 已允许；Deny = 已禁止；Prompt = 每次询问")
print("=" * 52)
