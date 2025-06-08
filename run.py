#!/usr/bin/env python3
"""
Olympic CamGuard 系統啟動腳本
"""

import subprocess
import sys
import os


def install_dependencies():
    """安裝必要的依賴包"""
    print("📦 安裝系統依賴...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依賴安裝完成")
    except subprocess.CalledProcessError:
        print("❌ 依賴安裝失敗，請手動執行: pip install -r requirements.txt")
        return False
    return True


def main():
    print("🏅 Olympic CamGuard - Alpha 版本")
    print("=" * 50)

    # 檢查 requirements.txt 是否存在
    if not os.path.exists("requirements.txt"):
        print("❌ 找不到 requirements.txt 文件")
        return

    # 安裝依賴
    if not install_dependencies():
        return

    print("\n🚀 啟動系統...")
    print("📱 Web 界面將在 http://localhost:8000 運行")
    print("📍 已設置台北地區測試場館")
    print("🧠 AI 圖像分析已啟用")
    print("\n按 Ctrl+C 停止系統")
    print("=" * 50)

    try:
        # 啟動主程序
        subprocess.run([sys.executable, "olympic_camguard.py"])
    except KeyboardInterrupt:
        print("\n\n👋 系統已停止")
    except FileNotFoundError:
        print("❌ 找不到 olympic_camguard.py 文件")


if __name__ == "__main__":
    main()
