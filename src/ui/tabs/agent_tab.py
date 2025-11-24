"""
Agent Tab

エージェント支援編集機能（将来実装予定）のプレースホルダー。
"""

import gradio as gr
from PIL import Image
from .base_tab import BaseTab
from ...core.tab_specs import TAB_AGENT


class AgentTab(BaseTab):
    """エージェント支援編集タブ（プレースホルダー）"""

    def create_ui(self) -> None:
        """エージェント支援タブのUIを構築"""
        with gr.Tab(TAB_AGENT.display_name, id=TAB_AGENT.key, elem_id=TAB_AGENT.elem_id):
            with gr.Row():
                with gr.Column(scale=1):
                    agent_input_image = gr.Image(label="編集する画像", type="pil")
                    agent_instruction = gr.Textbox(
                        label="編集指示",
                        placeholder="例: 画像を左右反転して、明るさを上げてください",
                        lines=4
                    )
                    agent_button = gr.Button("エージェントに依頼", variant="primary")

                with gr.Column(scale=1):
                    agent_output_image = gr.Image(label="編集後の画像", type="pil")
                    agent_output_info = gr.Markdown(label="処理内容")

            agent_button.click(
                fn=self.agent_assisted_edit,
                inputs=[agent_input_image, agent_instruction],
                outputs=[agent_output_image, agent_output_info]
            )

    def agent_assisted_edit(self, input_image: Image.Image, instruction: str):
        """
        エージェント支援で画像を編集する（プレースホルダー）

        Args:
            input_image: 入力画像
            instruction: 編集指示

        Returns:
            (output_image, info_text): 編集後の画像と処理内容
        """
        return input_image, """🚧 実装予定

このタブは将来的にエージェント機能を実装する予定です。
エージェントが指示を理解し、適切な画像編集を行います。
"""
