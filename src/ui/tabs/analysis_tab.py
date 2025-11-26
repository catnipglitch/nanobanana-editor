"""
Analysis Tab

画像解析タブ（画像説明生成・ポーズ抽出）
"""

import gradio as gr
import logging
from .base_tab import BaseTab
from ...core.tab_specs import TAB_ANALYSIS

logger = logging.getLogger(__name__)


class AnalysisTab(BaseTab):
    """画像解析タブ"""

    def create_ui(self) -> None:
        """画像解析タブのUIを作成"""
        with gr.Tab(TAB_ANALYSIS.display_name, id=TAB_ANALYSIS.key, elem_id=TAB_ANALYSIS.elem_id):
            gr.Markdown("# 🚧 Coming Soon")
            gr.Markdown("""
            このタブでは以下の機能を提供予定です:
            - Gemini Visionによる画像説明生成
            - キャラクターポーズ抽出
            - オブジェクト検出・分類
            - シーン理解・要素抽出

            **実装予定**: Phase 3.3
            """)
