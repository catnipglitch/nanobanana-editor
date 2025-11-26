"""
Multi-Turn Edit Tab

マルチターン編集タブ（対話形式での段階的編集）
"""

import gradio as gr
import logging
from .base_tab import BaseTab
from ...core.tab_specs import TAB_MULTITURN_EDIT

logger = logging.getLogger(__name__)


class MultiTurnEditTab(BaseTab):
    """マルチターン編集タブ"""

    def create_ui(self) -> None:
        """マルチターン編集タブのUIを作成"""
        with gr.Tab(TAB_MULTITURN_EDIT.display_name, id=TAB_MULTITURN_EDIT.key, elem_id=TAB_MULTITURN_EDIT.elem_id):
            gr.Markdown("# 🚧 Coming Soon")
            gr.Markdown("""
            このタブでは以下の機能を提供予定です:
            - 対話形式での段階的編集
            - 編集履歴の管理
            - 各ステップの保存・復元
            - プロンプトによる反復改善

            **実装予定**: Phase 3.4
            """)
