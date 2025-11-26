"""
Outfit Change Tab

衣装チェンジタブ（キャラクター着替え専用）
"""

import gradio as gr
import logging
from .base_tab import BaseTab
from ...core.tab_specs import TAB_OUTFIT_CHANGE

logger = logging.getLogger(__name__)


class OutfitChangeTab(BaseTab):
    """衣装チェンジタブ"""

    def create_ui(self) -> None:
        """衣装チェンジタブのUIを作成"""
        with gr.Tab(TAB_OUTFIT_CHANGE.display_name, id=TAB_OUTFIT_CHANGE.key, elem_id=TAB_OUTFIT_CHANGE.elem_id):
            gr.Markdown("# 🚧 Coming Soon")
            gr.Markdown("""
            このタブでは以下の機能を提供予定です:
            - キャラクターの衣装変更
            - マスクベースの部分置換
            - スタイル転送
            - 衣装テンプレート選択

            **実装予定**: Phase 3.5
            """)
