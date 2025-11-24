#!/usr/bin/env python3
"""
nanobabana-editor Gradio アプリケーションのエントリーポイント

使用例:
    python app.py              # 通常モード
    python app.py --test       # テストモード（APIキー不要）
    python app.py --share      # 公開リンクを生成
"""

from src.ui.gradio_app import main

if __name__ == "__main__":
    main()
