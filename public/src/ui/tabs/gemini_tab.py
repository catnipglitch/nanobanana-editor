"""
Gemini Tab

Gemini image generation tab.
"""

import gradio as gr
import logging
import io
import json
from PIL import Image
from .base_tab import BaseTab
from ...core.generators import GenerationConfig, ModelType
from ...core.tab_specs import TAB_GEMINI

logger = logging.getLogger(__name__)


class GeminiTab(BaseTab):
    """Gemini image generation tab"""

    def create_ui(self) -> None:
        """Create Gemini tab UI"""
        with gr.Tab(TAB_GEMINI.display_name, id=TAB_GEMINI.key, elem_id=TAB_GEMINI.elem_id):
            with gr.Row():
                with gr.Column(scale=1):
                    # プロンプト入力
                    gemini_template = gr.Dropdown(
                        label="プロンプトテンプレート",
                        choices=self.app.template_manager.get_template_choices(),
                        value="選択してください",
                        info="サンプルプロンプトを選択"
                    )

                    gemini_prompt = gr.Textbox(
                        label="プロンプト",
                        placeholder="生成したい画像を説明してください",
                        lines=4
                    )

                    gemini_model = gr.Dropdown(
                        label="モデル",
                        choices=self.app.gemini_models,
                        value=self.app.gemini_models[0] if self.app.gemini_models else None,
                        info="Gemini画像生成モデル"
                    )

                    # 詳細設定
                    with gr.Accordion("詳細設定", open=True):
                        gemini_aspect_ratio = gr.Dropdown(
                            label="アスペクト比",
                            choices=["1:1", "9:16", "16:9", "4:3", "3:4"],
                            value=self.app.default_aspect_ratio,
                            info="生成画像の縦横比"
                        )

                        gemini_image_size = gr.Dropdown(
                            label="解像度",
                            choices=["1K", "2K", "4K"],
                            value="1K",
                            info="Gemini 3 Pro Image Preview で有効（他モデルでは無視されます）"
                        )

                        gr.Markdown("**注意**: Geminiは1枚のみ生成します")

                    # ボタン
                    with gr.Row():
                        gemini_gen_button = gr.Button("画像を生成", variant="primary")
                        gemini_reset_button = gr.Button("リセット")

                with gr.Column(scale=1):
                    # 出力エリア
                    gemini_output_image = gr.Image(label="生成された画像", type="pil")
                    gemini_output_info = gr.Markdown(label="生成情報")

                    with gr.Accordion("JSONログ", open=False):
                        gemini_output_json = gr.Code(language="json", label="メタデータ")

            # イベントハンドラ
            gemini_template.change(
                fn=self.app.apply_template,
                inputs=[gemini_template],
                outputs=[gemini_prompt]
            )

            gemini_gen_button.click(
                fn=self.generate_gemini_image,
                inputs=[
                    gemini_prompt,
                    gemini_model,
                    gemini_aspect_ratio,
                    gemini_image_size,
                ],
                outputs=[
                    gemini_output_image,
                    gemini_output_info,
                    gemini_output_json
                ]
            )

            gemini_reset_button.click(
                fn=lambda: ("", self.app.default_aspect_ratio, "1K"),
                inputs=[],
                outputs=[gemini_prompt, gemini_aspect_ratio, gemini_image_size]
            )

    def generate_gemini_image(
        self,
        prompt: str,
        model_name: str,
        aspect_ratio: str,
        image_size: str,
    ):
        """
        Geminiで画像を生成（Tab 1: Gemini専用）

        Returns:
            (image, info_text, json_log): 生成された画像、情報テキスト、JSONログ
        """
        logger.info(f"=== Gemini Tab: Image Generation Request ===")

        if self.app.test_mode:
            return None, "⚠ テストモード: 画像生成機能は無効です", ""

        # 認証チェック
        if not self.app.google_api_key or self.app.gemini_generator is None:
            error_text = """❌ エラー: APIキーが設定されていません

**Settings タブ** でGoogle API Keyを設定してください。

1. Settings タブを開く
2. APIキーを入力
3. 「接続テスト」ボタンで確認
4. 「APIキーを適用」ボタンで適用
"""
            logger.error("API key not configured")
            return None, error_text, ""

        try:
            # Gemini固有の設定
            config = GenerationConfig(
                model_type=ModelType.TEST if model_name == "test-model" else ModelType.GEMINI,
                model_name=model_name,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                number_of_images=1,  # Geminiは常に1枚
                image_size=image_size,
            )

            # ジェネレータを選択
            if model_name == "test-model":
                generator = self.app.test_generator
            else:
                generator = self.app.gemini_generator

            # 画像生成
            image_data_list, metadata = generator.generate(config)

            # PIL Imageに変換
            pil_image = Image.open(io.BytesIO(image_data_list[0]))

            # ファイル保存
            image_path, metadata_path = self.app.output_manager.save_image_with_metadata(
                image_data=image_data_list[0],
                metadata=metadata,
                prefix="gemini_gen",
                extension="jpg",
            )

            # 情報テキスト生成
            info_text = f"### 生成完了 ✅\n\n"
            info_text += f"**モデル**: {model_name}\n"
            info_text += f"**画像ファイル**: `{image_path.name}`\n"
            info_text += f"**メタデータ**: `{metadata_path.name}`\n"
            info_text += f"**ファイルサイズ**: {len(image_data_list[0]) / 1024:.1f} KB\n"

            # JSONログ
            json_log = json.dumps(metadata, ensure_ascii=False, indent=2)

            logger.info(f"=== Gemini Tab: Image Generation Complete ===")
            return pil_image, info_text, json_log

        except Exception as e:
            logger.error(f"Gemini image generation failed: {e}", exc_info=True)
            error_text = f"❌ エラー: {str(e)}"
            return None, error_text, ""
