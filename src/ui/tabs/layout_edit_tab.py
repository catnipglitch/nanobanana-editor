"""
Layout Edit Tab

レイアウト編集タブ（キャラクターシート等の定型レイアウト）
"""

import gradio as gr
import logging
from .base_tab import BaseTab
from ...core.tab_specs import TAB_LAYOUT_EDIT

logger = logging.getLogger(__name__)


class LayoutEditTab(BaseTab):
    """レイアウト編集タブ"""

    def create_ui(self) -> None:
        """レイアウト編集タブのUIを作成"""
        with gr.Tab(TAB_LAYOUT_EDIT.display_name, id=TAB_LAYOUT_EDIT.key, elem_id=TAB_LAYOUT_EDIT.elem_id):
            gr.Markdown("# 🚧 Coming Soon")
            gr.Markdown("""
            このタブでは以下の機能を提供予定です:
            - キャラクターシート生成
            - 定型レイアウトテンプレート
            - 複数画像の配置・合成
            - テキスト・注釈の追加

            **実装予定**: Phase 3.6
            """)
