"""
Reference Tab

Reference image-based generation tab (Gemini with reference images).
"""

import gradio as gr
import logging
from PIL import Image
from .base_tab import BaseTab
from ...core.generators import GenerationConfig, ModelType
from ...core.tab_specs import TAB_REFERENCE

logger = logging.getLogger(__name__)


class ReferenceTab(BaseTab):
    """Reference image-based generation tab"""

    def create_ui(self) -> None:
        """Create Reference tab UI"""
        with gr.Tab(TAB_REFERENCE.display_name, id=TAB_REFERENCE.key, elem_id=TAB_REFERENCE.elem_id):
            gr.Markdown("""
            ### Gemini による参照画像ベース画像生成

            最大14枚の参照画像と共にプロンプトを入力して、新しい画像を生成します。
            参照画像なしでも生成可能です（text-to-image）。
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    edit_model = gr.Dropdown(
                        label="モデル選択",
                        choices=[
                            "gemini-2.5-flash-image",
                            "gemini-3-pro-image-preview"
                        ],
                        value="gemini-3-pro-image-preview"
                    )
                    edit_prompt = gr.Textbox(
                        label="プロンプト（必須）",
                        placeholder="例: An office group photo of these people, they are making funny faces.",
                        lines=4
                    )

                    # File component for upload (no tuple issue)
                    edit_file_upload = gr.File(
                        label="参照画像をアップロード（最大14枚、任意）",
                        file_count="multiple",
                        file_types=["image"],
                        interactive=True
                    )

                    # Gallery for preview
                    edit_reference_preview = gr.Gallery(
                        label="アップロード済み画像のプレビュー",
                        columns=4,
                        rows=2,
                        height=300,
                        interactive=False,
                        allow_preview=True
                    )

                    edit_image_count = gr.Markdown("**アップロード**: 0/14 枚")

                    with gr.Row():
                        edit_aspect_ratio = gr.Dropdown(
                            label="アスペクト比",
                            choices=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                            value="1:1"
                        )
                        edit_image_size = gr.Dropdown(
                            label="解像度",
                            choices=["1K", "2K", "4K"],
                            value="1K"
                        )

                    edit_button = gr.Button("画像生成", variant="primary")

                with gr.Column(scale=1):
                    edit_output_image = gr.Image(label="生成された画像", type="pil")
                    edit_output_info = gr.Markdown(label="生成情報")
                    edit_text_response = gr.Textbox(
                        label="テキスト応答（Geminiからの説明）",
                        lines=5,
                        interactive=False
                    )

            # File upload時にプレビューを更新
            edit_file_upload.change(
                fn=self.update_preview,
                inputs=[edit_file_upload],
                outputs=[edit_reference_preview, edit_image_count]
            )

            # 画像生成ボタン
            edit_button.click(
                fn=self.generate_with_reference_images,
                inputs=[edit_model, edit_prompt, edit_file_upload, edit_aspect_ratio, edit_image_size],
                outputs=[edit_output_image, edit_output_info, edit_text_response]
            )

    def update_preview(self, files):
        """
        参照画像アップロード時にプレビューを更新（Phase 2.6）

        Args:
            files: アップロードされたファイルのリスト

        Returns:
            (images, count_text): プレビュー画像のリストと画像数テキスト
        """
        if not files or len(files) == 0:
            return None, "**アップロード**: 0/14 枚"

        # 最大14枚に制限
        files = files[:14]

        # PIL Imageをロードしてプレビュー表示
        images = []
        for file in files:
            try:
                img = Image.open(file)
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to load image {file}: {e}")

        count_text = f"**アップロード**: {len(images)}/14 枚"
        return images, count_text

    def generate_with_reference_images(self, model_name: str, prompt: str, file_paths, aspect_ratio: str, image_size: str):
        """
        参照画像をベースに画像生成する（Tab 3用 - Phase 2.6）

        Args:
            model_name: 使用するモデル名
            prompt: 生成プロンプト
            file_paths: アップロードされた参照画像のパス
            aspect_ratio: アスペクト比
            image_size: 画像サイズ

        Returns:
            (generated_image, info_text, text_response): 生成画像、情報テキスト、テキスト応答
        """
        if not prompt or prompt.strip() == "":
            return None, "⚠ プロンプトを入力してください", ""

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
            # 参照画像の処理（最大14枚）- File component returns paths directly
            ref_images_list = None
            if file_paths and len(file_paths) > 0:
                ref_images_list = []
                for file_path in file_paths[:14]:  # 最大14枚に制限
                    try:
                        img = Image.open(file_path)
                        ref_images_list.append(img)
                    except Exception as e:
                        logger.error(f"Failed to load reference image {file_path}: {e}")

                if not ref_images_list:
                    ref_images_list = None
                else:
                    logger.info(f"Using {len(ref_images_list)} reference images")

            # GenerationConfig作成
            config = GenerationConfig(
                model_type=ModelType.GEMINI,
                model_name=model_name,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                reference_images=ref_images_list,
                number_of_images=1,
            )

            # 画像生成
            image_data_list, metadata = self.app.gemini_generator.generate(config)

            if not image_data_list:
                return None, "❌ 画像の生成に失敗しました", ""

            # 画像とメタデータを保存
            image_path, metadata_path = self.app.output_manager.save_image_with_metadata(
                image_data=image_data_list[0],
                metadata=metadata,
                prefix="edit_gen",
                extension="jpg",
            )

            # 生成した画像を読み込み
            generated_image = Image.open(image_path)

            # テキスト応答を取得
            text_response = metadata.get("text_response", "")

            # 情報テキスト作成
            info_text = f"""✓ 画像生成完了!

**モデル**: {model_name}
**プロンプト**: {prompt[:100]}{'...' if len(prompt) > 100 else ''}
**アスペクト比**: {aspect_ratio}
**解像度**: {image_size}
**参照画像数**: {len(ref_images_list) if ref_images_list else 0}
**画像ファイル**: `{image_path.name}`
**メタデータ**: `{metadata_path.name}`
**サイズ**: {len(image_data_list[0]):,} バイト
"""

            logger.info(f"Reference image generation complete: {image_path.name}")
            return generated_image, info_text, text_response

        except Exception as e:
            logger.error(f"Reference image generation failed: {e}", exc_info=True)
            error_text = f"❌ エラー: {str(e)}"
            return None, error_text, ""
