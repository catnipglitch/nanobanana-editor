"""
Imagen Tab

Imagen image generation tab.
"""

import gradio as gr
import logging
import io
import json
from PIL import Image
from .base_tab import BaseTab
from ...core.generators import GenerationConfig, ModelType
from ...core.tab_specs import TAB_IMAGEN

logger = logging.getLogger(__name__)


class ImagenTab(BaseTab):
    """Imagen image generation tab"""

    def create_ui(self) -> None:
        """Create Imagen tab UI"""
        with gr.Tab(TAB_IMAGEN.display_name, id=TAB_IMAGEN.key, elem_id=TAB_IMAGEN.elem_id):
            with gr.Row():
                with gr.Column(scale=1):
                    # プロンプト入力
                    imagen_template = gr.Dropdown(
                        label="プロンプトテンプレート",
                        choices=self.app.template_manager.get_template_choices(),
                        value="選択してください",
                        info="サンプルプロンプトを選択"
                    )

                    imagen_prompt = gr.Textbox(
                        label="プロンプト",
                        placeholder="生成したい画像を説明してください",
                        lines=4
                    )

                    imagen_model = gr.Dropdown(
                        label="モデル",
                        choices=self.app.imagen_models,
                        value=self.app.imagen_models[1] if len(self.app.imagen_models) > 1 else self.app.imagen_models[0],
                        info="Imagen画像生成モデル"
                    )

                    # 詳細設定
                    with gr.Accordion("詳細設定", open=True):
                        imagen_warning = gr.Markdown(value="", visible=True)

                        imagen_aspect_ratio = gr.Dropdown(
                            label="アスペクト比",
                            choices=["1:1", "9:16", "16:9", "4:3", "3:4"],
                            value=self.app.default_aspect_ratio,
                            info="生成画像の縦横比"
                        )

                        imagen_number_of_images = gr.Number(
                            label="画像枚数",
                            value=self.app.default_number_of_images,
                            minimum=1,
                            maximum=8,
                            step=1,
                            info="一度に生成する画像の数（1-8枚）"
                        )

                        imagen_seed_enabled = gr.Checkbox(
                            label="Seed を指定",
                            value=False,
                            info="再現性のための乱数シード（add_watermarkと排他的）"
                        )

                        imagen_seed_value = gr.Number(
                            label="Seed 値",
                            value=0,
                            minimum=0,
                            step=1,
                            visible=False
                        )

                        imagen_enhance_prompt = gr.Checkbox(
                            label="プロンプト拡張",
                            value=False,
                            info="AIがプロンプトを自動的に詳細化・改善"
                        )

                        imagen_add_watermark = gr.Checkbox(
                            label="透かし追加",
                            value=False,
                            info="生成画像に透かしを追加（seedと排他的）"
                        )

                        imagen_safety_filter = gr.Dropdown(
                            label="安全フィルタレベル",
                            choices=[
                                ("最強フィルタ - 最も厳格 (block_low_and_above / 旧:block_most)", "block_low_and_above"),
                                ("標準フィルタ - 一部ブロック (block_medium_and_above / 旧:block_some)", "block_medium_and_above"),
                                ("弱いフィルタ - ブロック少なめ (block_only_high / 旧:block_few)", "block_only_high"),
                                ("最弱フィルタ - アクセス制限あり (block_none / 旧:block_fewest) ※機能しない可能性あり", "block_none")
                            ],
                            value=self.app.default_safety_filter,
                            info="不適切なコンテンツのフィルタリング強度"
                        )

                        imagen_person_generation = gr.Dropdown(
                            label="人物生成",
                            choices=[
                                ("成人のみ許可 (allow_adult)", "allow_adult"),
                                ("全年齢許可 (allow_all)", "allow_all"),
                                ("人物生成禁止 (block_all)", "block_all")
                            ],
                            value=self.app.default_person_generation,
                            info="人物画像の生成許可レベル"
                        )

                    # ボタン
                    with gr.Row():
                        imagen_gen_button = gr.Button("画像を生成", variant="primary")
                        imagen_reset_button = gr.Button("リセット")

                with gr.Column(scale=1):
                    # 出力エリア（単一画像 or Gallery）
                    imagen_output_image = gr.Image(label="生成された画像", type="pil")
                    imagen_output_gallery = gr.Gallery(label="生成された画像（複数）", visible=False)
                    imagen_output_info = gr.Markdown(label="生成情報")

                    with gr.Accordion("JSONログ", open=False):
                        imagen_output_json = gr.Code(language="json", label="メタデータ")

            # イベントハンドラ
            imagen_template.change(
                fn=self.apply_template,
                inputs=[imagen_template],
                outputs=[imagen_prompt]
            )

            # Seed有効化でseed値入力を表示
            imagen_seed_enabled.change(
                fn=lambda enabled: gr.update(visible=enabled),
                inputs=[imagen_seed_enabled],
                outputs=[imagen_seed_value]
            )

            # Seed/Watermark排他制御
            imagen_seed_enabled.change(
                fn=lambda s, w: self.toggle_seed_watermark_imagen(s, w, "seed"),
                inputs=[imagen_seed_enabled, imagen_add_watermark],
                outputs=[imagen_seed_enabled, imagen_add_watermark, imagen_warning]
            )

            imagen_add_watermark.change(
                fn=lambda s, w: self.toggle_seed_watermark_imagen(s, w, "watermark"),
                inputs=[imagen_seed_enabled, imagen_add_watermark],
                outputs=[imagen_seed_enabled, imagen_add_watermark, imagen_warning]
            )

            imagen_gen_button.click(
                fn=self.generate_imagen_image,
                inputs=[
                    imagen_prompt,
                    imagen_model,
                    imagen_aspect_ratio,
                    imagen_number_of_images,
                    imagen_seed_enabled,
                    imagen_seed_value,
                    imagen_enhance_prompt,
                    imagen_add_watermark,
                    imagen_safety_filter,
                    imagen_person_generation,
                ],
                outputs=[
                    imagen_output_image,
                    imagen_output_gallery,
                    imagen_output_image,  # visible更新
                    imagen_output_gallery,  # visible更新
                    imagen_output_info,
                    imagen_output_json
                ]
            )

            imagen_reset_button.click(
                fn=lambda: (
                    "",
                    self.app.default_aspect_ratio,
                    self.app.default_number_of_images,
                    False,
                    0,
                    False,
                    False,
                    self.app.default_safety_filter,
                    self.app.default_person_generation,
                ),
                inputs=[],
                outputs=[
                    imagen_prompt,
                    imagen_aspect_ratio,
                    imagen_number_of_images,
                    imagen_seed_enabled,
                    imagen_seed_value,
                    imagen_enhance_prompt,
                    imagen_add_watermark,
                    imagen_safety_filter,
                    imagen_person_generation,
                ]
            )

    def generate_imagen_image(
        self,
        prompt: str,
        model_name: str,
        aspect_ratio: str,
        number_of_images: int,
        seed_enabled: bool,
        seed_value: int,
        enhance_prompt: bool,
        add_watermark: bool,
        safety_filter_level: str,
        person_generation: str,
    ):
        """
        Imagenで画像を生成（Tab 2: Imagen専用）

        Returns:
            (single_image, gallery_images, single_visible, gallery_visible, info_text, json_log):
            単一画像、複数画像、表示制御、情報テキスト、JSONログ
        """
        logger.info(f"=== Imagen Tab: Image Generation Request ===")

        if self.app.test_mode:
            return None, None, gr.update(visible=True), gr.update(visible=False), "⚠ テストモード: 画像生成機能は無効です", ""

        # 認証チェック
        if not self.app.google_api_key or self.app.imagen_generator is None:
            error_text = """❌ エラー: APIキーが設定されていません

**Settings タブ** でGoogle API Keyを設定してください。

1. Settings タブを開く
2. APIキーを入力
3. 「接続テスト」ボタンで確認
4. 「APIキーを適用」ボタンで適用
"""
            logger.error("API key not configured")
            return None, None, gr.update(visible=True), gr.update(visible=False), error_text, ""

        try:
            # Imagen固有の設定
            config = GenerationConfig(
                model_type=ModelType.IMAGEN if model_name.startswith("imagen") else ModelType.TEST,
                model_name=model_name,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                number_of_images=number_of_images,
                seed=seed_value if seed_enabled else None,
                enhance_prompt=enhance_prompt,
                add_watermark=add_watermark,
                safety_filter_level=safety_filter_level,
                person_generation=person_generation,
            )

            # ジェネレータを選択
            if model_name == "test-model":
                generator = self.app.test_generator
            else:
                generator = self.app.imagen_generator

            # 画像生成
            image_data_list, metadata = generator.generate(config)

            # PIL Imageに変換
            pil_images = [Image.open(io.BytesIO(data)) for data in image_data_list]

            # ファイル保存（HF Spacesでは無効化される可能性あり）
            image_files = []
            metadata_path = None

            if len(image_data_list) == 1:
                save_result = self.app.output_manager.save_image_with_metadata(
                    image_data=image_data_list[0],
                    metadata=metadata,
                    prefix="imagen_gen"
                )
                if save_result:
                    image_path, metadata_path = save_result
                    image_files = [image_path.name]
            else:
                save_result = self.app.output_manager.save_images_with_metadata(
                    image_data_list=image_data_list,
                    metadata=metadata,
                    prefix="imagen_gen"
                )
                if save_result:
                    image_paths, metadata_path = save_result
                    image_files = [p.name for p in image_paths]

            # 情報テキスト生成
            info_text = f"### 生成完了 ✅\n\n"
            info_text += f"**モデル**: {model_name}\n"
            info_text += f"**生成枚数**: {len(pil_images)}枚\n"

            if image_files and len(image_files) == 1:
                info_text += f"**画像ファイル**: `{image_files[0]}`\n"
                info_text += f"**メタデータ**: `{metadata_path.name}`\n"
                info_text += f"**ファイルサイズ**: {len(image_data_list[0]) / 1024:.1f} KB\n"
            elif image_files:
                info_text += f"**画像ファイル**: \n"
                for img_file in image_files:
                    info_text += f"  - `{img_file}`\n"
                info_text += f"**メタデータ**: `{metadata_path.name}`\n"

            if seed_enabled:
                info_text += f"**使用したseed**: {seed_value}\n"

            # JSONログ
            json_log = json.dumps(metadata, ensure_ascii=False, indent=2)

            # 単一画像/複数画像で出力を切り替え
            if len(pil_images) == 1:
                logger.info("Returning single image via Image component")
                return pil_images[0], None, gr.update(visible=True), gr.update(visible=False), info_text, json_log
            else:
                logger.info(f"Returning {len(pil_images)} images via Gallery component")
                return None, pil_images, gr.update(visible=False), gr.update(visible=True), info_text, json_log

        except Exception as e:
            logger.error(f"Imagen image generation failed: {e}", exc_info=True)
            error_text = f"❌ エラー: {str(e)}"
            return None, None, gr.update(visible=True), gr.update(visible=False), error_text, ""

    def apply_template(self, template_name: str):
        """プロンプトテンプレートを適用"""
        if template_name == "選択してください":
            return ""

        template = self.app.template_manager.get_template_by_name(template_name)
        if template:
            return template.prompt
        else:
            return ""

    def toggle_seed_watermark_imagen(self, seed_enabled: bool, add_watermark: bool, changed_component: str):
        """Imagenタブ用: seed と watermark の排他制御"""
        if changed_component == "seed" and seed_enabled and add_watermark:
            return True, False, "⚠ 注意: seedを有効にしたため、add_watermarkを無効化しました"
        elif changed_component == "watermark" and add_watermark and seed_enabled:
            return False, True, "⚠ 注意: add_watermarkを有効にしたため、seedを無効化しました"
        else:
            return seed_enabled, add_watermark, ""
