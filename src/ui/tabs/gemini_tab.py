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
from ...core.prompt_optimizer import PromptOptimizer

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

                    # 最適化されたプロンプト表示
                    gemini_optimized_prompt = gr.Textbox(
                        label="最適化されたプロンプト",
                        placeholder="「プロンプト生成」または「編集開始」をクリックすると、最適化されたプロンプトが表示されます",
                        lines=5,
                        interactive=True,
                        info="生成前にプロンプトを確認・編集できます"
                    )

                    # プロンプト最適化レベル
                    gemini_optimization_level = gr.Radio(
                        label="プロンプト最適化レベル",
                        choices=[
                            ("レベル0: 最適化なし（整合性チェックのみ）", 0),
                            ("レベル1: Gemini 3.0 自動最適化（推奨）", 1),
                            ("レベル2: Gemini 3.0 誇張表現追加", 2),
                        ],
                        value=1,
                        info="Gemini 3.0でプロンプトを最適化します"
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

                    # ツールオプション（Phase 3.0）
                    with gr.Accordion("ツールオプション", open=False):
                        gemini_google_search = gr.Checkbox(
                            label="Google Search",
                            value=False,
                            info="Google検索でリアルタイム情報を取得（Gemini 3 Pro Image推奨）"
                        )

                    # ボタン
                    with gr.Row():
                        gemini_gen_button = gr.Button("編集開始", variant="primary")
                        gemini_prompt_button = gr.Button("プロンプト生成", variant="secondary", size="sm")
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

            # プロンプト生成ボタン
            gemini_prompt_button.click(
                fn=self.generate_optimized_prompt,
                inputs=[
                    gemini_prompt,
                    gemini_optimization_level,
                ],
                outputs=[gemini_optimized_prompt]
            )

            # 編集開始ボタン
            gemini_gen_button.click(
                fn=self.generate_gemini_image,
                inputs=[
                    gemini_prompt,
                    gemini_optimization_level,
                    gemini_optimized_prompt,
                    gemini_model,
                    gemini_aspect_ratio,
                    gemini_image_size,
                    gemini_google_search,  # Phase 3.0
                ],
                outputs=[
                    gemini_output_image,
                    gemini_output_info,
                    gemini_output_json,
                    gemini_optimized_prompt,
                ]
            )

            gemini_reset_button.click(
                fn=lambda: ("", "", 1, self.app.default_aspect_ratio, "1K", False, None, "", ""),
                inputs=[],
                outputs=[
                    gemini_prompt,
                    gemini_optimized_prompt,
                    gemini_optimization_level,
                    gemini_aspect_ratio,
                    gemini_image_size,
                    gemini_google_search,
                    gemini_output_image,
                    gemini_output_info,
                    gemini_output_json,
                ]
            )

    def generate_optimized_prompt(
        self,
        prompt_text: str,
        optimization_level: int,
    ) -> str:
        """
        プロンプトのみを生成（画像生成なし）

        Args:
            prompt_text: ユーザーのプロンプト
            optimization_level: プロンプト最適化レベル（0/1/2）

        Returns:
            最適化されたプロンプト文字列
        """
        if not prompt_text or prompt_text.strip() == "":
            return "⚠️ プロンプトを入力してください"

        try:
            optimizer = PromptOptimizer(self.app.google_api_key)

            if optimization_level == 0:
                # レベル0: 整合性チェックのみ（そのまま返す）
                return prompt_text
            else:
                # レベル1,2: Gemini 3.0 最適化
                optimized_prompt, opt_error = optimizer.optimize(
                    "", prompt_text, optimization_level
                )
                if opt_error:
                    return f"⚠️ 最適化エラー: {opt_error}\n\nフォールバックプロンプト:\n{optimized_prompt}"

            return optimized_prompt

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}", exc_info=True)
            return f"⚠️ エラー: {str(e)}\n\n元のプロンプト:\n{prompt_text}"

    def generate_gemini_image(
        self,
        prompt: str,
        optimization_level: int,
        optimized_prompt_from_ui: str,
        model_name: str,
        aspect_ratio: str,
        image_size: str,
        enable_google_search: bool,  # Phase 3.0
    ):
        """
        Geminiで画像を生成（Tab 1: Gemini専用）

        Args:
            prompt: ユーザーの元のプロンプト
            optimization_level: プロンプト最適化レベル（0/1/2）
            optimized_prompt_from_ui: UI経由で既に最適化されたプロンプト（空文字列なら新規生成）
            model_name: モデル名
            aspect_ratio: アスペクト比
            image_size: 画像サイズ
            enable_google_search: Google検索を有効化

        Returns:
            (image, info_text, json_log, optimized_prompt): 生成された画像、情報テキスト、JSONログ、最適化プロンプト
        """
        logger.info(f"=== Gemini Tab: Image Generation Request ===")
        logger.info(f"Google Search: {enable_google_search}")

        if self.app.test_mode:
            return None, "⚠ テストモード: 画像生成機能は無効です", "", ""

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
            return None, error_text, "", ""

        try:
            # プロンプト選択ロジック
            optimized_prompt_info = None

            if optimized_prompt_from_ui and optimized_prompt_from_ui.strip():
                # UI経由で既に最適化されたプロンプトが渡された場合
                final_prompt = optimized_prompt_from_ui
                logger.info("Using pre-optimized prompt from UI")
            else:
                # プロンプト最適化
                if optimization_level > 0:
                    try:
                        optimizer = PromptOptimizer(self.app.google_api_key)
                        final_prompt, opt_error = optimizer.optimize(
                            "", prompt, optimization_level
                        )
                        if opt_error:
                            logger.warning(f"Prompt optimization warning: {opt_error}")
                            optimized_prompt_info = opt_error
                        logger.info(f"Prompt optimized (level {optimization_level})")
                    except Exception as e:
                        logger.error(f"Prompt optimization failed: {e}", exc_info=True)
                        optimized_prompt_info = f"最適化エラー: {str(e)}"
                        final_prompt = prompt  # フォールバック: 元のプロンプトを使用
                else:
                    # レベル0: 最適化なし
                    final_prompt = prompt

            # Gemini固有の設定（Phase 3.0: enable_google_search追加）
            config = GenerationConfig(
                model_type=ModelType.TEST if model_name == "test-model" else ModelType.GEMINI,
                model_name=model_name,
                prompt=final_prompt,
                aspect_ratio=aspect_ratio,
                number_of_images=1,  # Geminiは常に1枚
                image_size=image_size,
                enable_google_search=enable_google_search,  # Phase 3.0
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

            # ファイル保存（HF Spacesでは無効化される可能性あり）
            save_result = self.app.output_manager.save_image_with_metadata(
                image_data=image_data_list[0],
                metadata=metadata,
                prefix="gemini_gen",
                extension="jpg",
            )

            # 情報テキスト生成
            info_text = f"### 生成完了 ✅\n\n"
            info_text += f"**モデル**: {model_name}\n"
            if save_result:
                image_path, metadata_path = save_result
                info_text += f"**画像ファイル**: `{image_path.name}`\n"
                info_text += f"**メタデータ**: `{metadata_path.name}`\n"
            else:
                info_text += f"**画像ファイル**: （保存無効）\n"
            info_text += f"**ファイルサイズ**: {len(image_data_list[0]) / 1024:.1f} KB\n"
            if optimization_level > 0:
                info_text += f"**プロンプト最適化**: レベル{optimization_level}\n"
            if optimized_prompt_info:
                info_text += f"**⚠️ 最適化警告**: {optimized_prompt_info}\n"

            # JSONログ
            json_log = json.dumps(metadata, ensure_ascii=False, indent=2)

            logger.info(f"=== Gemini Tab: Image Generation Complete ===")
            return pil_image, info_text, json_log, final_prompt

        except Exception as e:
            logger.error(f"Gemini image generation failed: {e}", exc_info=True)
            error_text = f"❌ エラー: {str(e)}"
            return None, error_text, "", ""
