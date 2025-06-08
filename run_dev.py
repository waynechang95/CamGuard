#!/usr/bin/env python3
"""
Olympic CamGuard 開發模式啟動腳本
支持熱重載功能，適合開發調試使用
"""

import subprocess
import sys
import os


def main():
    print("🏅 Olympic CamGuard - 開發模式")
    print("=" * 50)
    print("⚡ 啟用熱重載功能...")
    print("📱 Web 界面將在 http://localhost:8000 運行")
    print("🔄 文件變更將自動重啟服務")
    print("\n按 Ctrl+C 停止系統")
    print("=" * 50)

    try:
        # 使用 uvicorn 命令行啟動，支持熱重載
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "olympic_camguard:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 開發服務器已停止")
    except FileNotFoundError:
        print("❌ 找不到 uvicorn 或 olympic_camguard.py")
        print("請確認已安裝依賴: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
