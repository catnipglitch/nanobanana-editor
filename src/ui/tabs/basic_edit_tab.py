"""
Basic Edit Tab

ベーシック編集タブ（シンプルな1回編集・アップスケール）
"""

import gradio as gr
import logging
import json
import base64
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

from .base_tab import BaseTab
from ...core.tab_specs import TAB_BASIC_EDIT
from ...core.prompt_optimizer import PromptOptimizer
from .prompts.basic_edit_prompts import UPSCALE_ONLY_PROMPT

if TYPE_CHECKING:
    from ..gradio_app import NanobananaApp

logger = logging.getLogger(__name__)

class BasicEditTab(BaseTab):
    """ベーシック編集タブ（シンプルな1回編集・アップスケール）"""

    def __init__(self, app: "NanobananaApp"):
        super().__init__(app)

    @staticmethod
    def calculate_target_resolution(
        aspect_ratio: str, resolution: str
    ) -> tuple[int, int]:
        """アスペクト比と解像度から目標解像度を計算する"""
        aspect_map = {
            "1:1": (1, 1),
            "2:3": (2, 3),
            "3:2": (3, 2),
            "3:4": (3, 4),
            "4:3": (4, 3),
            "4:5": (4, 5),
            "5:4": (5, 4),
            "9:16": (9, 16),
            "16:9": (16, 9),
            "21:9": (21, 9),
        }

        resolution_base = {"1K": 1024, "2K": 2048, "4K": 4096}

        w_ratio, h_ratio = aspect_map[aspect_ratio]
        base = resolution_base[resolution]

        # アスペクト比に応じて解像度を計算
        if w_ratio >= h_ratio:
            width = base
            height = int(base * h_ratio / w_ratio)
        else:
            height = base
            width = int(base * w_ratio / h_ratio)

        return width, height

    @staticmethod
    def compose_alpha_channel(rgb_img: Image.Image, alpha_bytes: bytes) -> bytes:
        """
        RGB画像とアルファマット画像を合成してRGBA PNG画像を生成する

        Args:
            rgb_img: RGB画像
            alpha_bytes: アルファマット画像のバイナリデータ

        Returns:
            RGBA PNG形式のバイナリデータ
        """
        # アルファマット画像を読み込み
        alpha_img = Image.open(BytesIO(alpha_bytes))

        # グレースケールに変換
        if alpha_img.mode != "L":
            alpha_img = alpha_img.convert("L")

        # サイズが一致しない場合はリサイズ
        if rgb_img.size != alpha_img.size:
            alpha_img = alpha_img.resize(rgb_img.size, Image.Resampling.LANCZOS)

        # RGBA画像を作成
        rgba_img = rgb_img.convert("RGBA")
        rgba_img.putalpha(alpha_img)

        # PNG形式でバイナリ出力
        output_buffer = BytesIO()
        rgba_img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()

    def _validate_inputs(
        self, input_image: Optional[Image.Image]
    ) -> Optional[tuple[None, None, None, str, str]]:
        """
        入力検証（APIキーと入力画像）

        Returns:
            エラーがある場合はエラーレスポンスタプル、問題なければNone
        """
        # 1. 入力検証: APIキー
        if not self.app.google_api_key or self.app.gemini_generator is None:
            error_text = """❌ エラー: APIキーが設定されていません

**Settings タブ** でGoogle API Keyを設定してください。

1. Settings タブを開く
2. APIキーを入力
3. 「接続テスト」ボタンで確認
4. 「APIキーを適用」ボタンで適用
"""
            logger.error("API key not configured")
            return None, None, None, error_text, ""

        # 2. 入力検証: 入力画像
        if input_image is None:
            return None, None, None, "⚠ 入力画像をアップロードしてください", ""

        return None

    def _execute_multi_turn_edit(
        self,
        input_image: Image.Image,
        model_name: str,
        prompt_text: str,
        optimization_level: int,
        process_num: int,
        aspect_ratio: str,
        resolution: str,
        lighting_enabled: bool,
        alpha_prompt_choice: str,
        pre_optimized_prompt: Optional[str] = None,  # NEW PARAMETER
    ) -> tuple[
        Optional[tuple[Image.Image, bytes, int, int]],  # (edited_img, edited_img_data, w, h)
        Optional[tuple[Image.Image, bytes]],  # (alpha_matte_img, alpha_matte_data)
        Optional[tuple[Image.Image, bytes]],  # (rgba_img, rgba_bytes)
        tuple[int, int],  # (input_w, input_h)
        tuple[int, int],  # (target_w, target_h)
        str,  # edit_type
        Optional[str],  # optimized_prompt_info (最適化エラーメッセージ or None)
    ]:
        """
        マルチターン編集を実行（Gemini API呼び出し + RGBA合成）

        Args:
            process_num: 処理タイプ番号（1/2/3）
            optimization_level: プロンプト最適化レベル（0/1/2）
            pre_optimized_prompt: 既に最適化されたプロンプト（UI経由）

        Returns:
            (edited_result, alpha_result, rgba_result, input_size, target_size, edit_type, optimized_prompt_info)
            エラー時は各resultがNoneになる可能性あり
        """
        # RGB変換（アルファチャンネル削除）
        if input_image.mode == "RGBA":
            background = Image.new("RGB", input_image.size, (255, 255, 255))
            background.paste(input_image, mask=input_image.split()[3])
            input_image = background
        elif input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        input_w, input_h = input_image.size

        # 目標解像度計算
        target_w, target_h = self.calculate_target_resolution(aspect_ratio, resolution)

        # プロンプト選択ロジック
        optimized_prompt_error = None

        if pre_optimized_prompt:
            # UI経由で既に最適化されたプロンプトが渡された場合
            edit_prompt = pre_optimized_prompt
            edit_type = "PRE_OPTIMIZED (UI経由)"
            logger.info("Using pre-optimized prompt from UI")
        else:
            # 処理タイプに応じた基本プロンプト選択
            if process_num == 1:
                # 処理1: 従来のロジック（determine_edit_promptを使用）
                edit_prompt, edit_type = self.determine_edit_prompt(
                    (input_w, input_h), (target_w, target_h), lighting_enabled
                )
            elif process_num == 2:
                # 処理2: ポーズ変更なしの背景除去
                edit_prompt = PRESERVE_POSE_PROMPT
                edit_type = "PRESERVE_POSE (ポーズ維持)"
            elif process_num == 3:
                # 処理3: 生成アップスケールのみ
                edit_prompt = UPSCALE_ONLY_PROMPT
                edit_type = "UPSCALE_ONLY (アップスケール)"
            else:
                # フォールバック（通常ここには来ない）
                edit_prompt, edit_type = self.determine_edit_prompt(
                    (input_w, input_h), (target_w, target_h), lighting_enabled
                )

            # プロンプト最適化（pre_optimized_promptが無い場合のみ）
            if optimization_level > 0:
                try:
                    optimizer = PromptOptimizer(self.app.google_api_key)
                    optimized_prompt, opt_error = optimizer.optimize(
                        edit_prompt, prompt_text, optimization_level
                    )
                    if opt_error:
                        logger.warning(f"Prompt optimization warning: {opt_error}")
                        optimized_prompt_error = opt_error
                    edit_prompt = optimized_prompt
                    logger.info(f"Prompt optimized (level {optimization_level})")
                except Exception as e:
                    logger.error(f"Prompt optimization failed: {e}", exc_info=True)
                    optimized_prompt_error = f"最適化エラー: {str(e)}"
                    # フォールバック: 追加指示を単純結合
                    if prompt_text and prompt_text.strip():
                        edit_prompt += f"\n\nAdditional instructions: {prompt_text.strip()}"
            else:
                # レベル0: 追加指示があれば単純結合
                if prompt_text and prompt_text.strip():
                    edit_prompt += f"\n\nAdditional instructions: {prompt_text.strip()}"

        # 入力画像をバイナリ化
        img_buffer = BytesIO()
        input_image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        # Geminiチャット作成
        client = genai.Client(api_key=self.app.google_api_key, vertexai=False)

        chat = client.chats.create(
            model=model_name,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio, image_size=resolution
                ),
            ),
        )

        # 1ターン目: 画像編集
        logger.info(f"Sending turn 1: Image editing (type: {edit_type})")
        response_1 = chat.send_message(
            [
                edit_prompt,
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            ]
        )

        # 編集画像を取得
        edited_img_data = None
        for part in response_1.parts:
            if part.text is not None:
                logger.info(f"API response (turn 1): {part.text}")
            elif hasattr(part, "inline_data") and part.inline_data:
                data_field = part.inline_data.data
                if isinstance(data_field, str):
                    edited_img_data = base64.b64decode(data_field)
                else:
                    edited_img_data = data_field

        if edited_img_data is None:
            return None, None, None, (input_w, input_h), (target_w, target_h), edit_type, optimized_prompt_error

        # 編集画像をPIL Imageに変換
        edited_img = Image.open(BytesIO(edited_img_data))
        edited_w, edited_h = edited_img.size
        logger.info(f"Edited image size: {edited_w}x{edited_h}")

        # 処理3の場合はアルファマット生成をスキップ
        if process_num == 3:
            # アップスケールのみ：アルファマットなし
            logger.info("Process 3: Skipping alpha matte generation (upscale only)")
            return (
                (edited_img, edited_img_data, edited_w, edited_h),
                None,
                None,
                (input_w, input_h),
                (target_w, target_h),
                edit_type,
                optimized_prompt_error,
            )

        # 2ターン目: アルファマット生成（処理1,2のみ）
        alpha_prompt = self.ALPHA_MATTE_PROMPTS[alpha_prompt_choice]
        logger.info("Sending turn 2: Alpha matte generation")
        response_2 = chat.send_message(alpha_prompt)

        # アルファマット画像を取得
        alpha_matte_data = None
        for part in response_2.parts:
            if part.text is not None:
                logger.info(f"API response (turn 2): {part.text}")
            elif hasattr(part, "inline_data") and part.inline_data:
                data_field = part.inline_data.data
                if isinstance(data_field, str):
                    alpha_matte_data = base64.b64decode(data_field)
                else:
                    alpha_matte_data = data_field

        if alpha_matte_data is None:
            return (
                (edited_img, edited_img_data, edited_w, edited_h),
                None,
                None,
                (input_w, input_h),
                (target_w, target_h),
                edit_type,
                optimized_prompt_error,
            )

        # アルファマット画像をPIL Imageに変換（表示用）
        alpha_matte_img = Image.open(BytesIO(alpha_matte_data))

        # RGBA合成（ローカル処理）
        rgba_bytes = self.compose_alpha_channel(edited_img, alpha_matte_data)
        rgba_img = Image.open(BytesIO(rgba_bytes))

        return (
            (edited_img, edited_img_data, edited_w, edited_h),
            (alpha_matte_img, alpha_matte_data),
            (rgba_img, rgba_bytes),
            (input_w, input_h),
            (target_w, target_h),
            edit_type,
            optimized_prompt_error,
        )

    def _save_outputs(
        self,
        edited_img_data: bytes,
        alpha_matte_data: bytes,
        rgba_bytes: bytes,
        save_edited_image: bool,
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
        """
        ファイル保存処理

        Returns:
            (edited_path, matte_path, rgba_path, json_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        import random
        import string

        unique_id = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )
        base_filename = f"alpha_matte_{timestamp}_{unique_id}"

        # ファイル保存処理（HF Spacesでは無効化）
        edited_path = None
        matte_path = None
        rgba_path = None
        json_path = None

        if not self.app.output_manager.disable_save:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            # 編集画像を保存（オプション）
            if save_edited_image:
                edited_path = output_dir / f"{base_filename}_edited.png"
                with open(edited_path, "wb") as f:
                    f.write(edited_img_data)

            # アルファマット保存
            matte_path = output_dir / f"{base_filename}_matte.png"
            with open(matte_path, "wb") as f:
                f.write(alpha_matte_data)

            # RGBA画像保存
            rgba_path = output_dir / f"{base_filename}_rgba.png"
            with open(rgba_path, "wb") as f:
                f.write(rgba_bytes)

            # メタデータは _build_response で生成されるので、ここではパスだけ返す
            json_path = output_dir / f"{base_filename}.json"

        return edited_path, matte_path, rgba_path, json_path

    def _build_response(
        self,
        model_name: str,
        aspect_ratio: str,
        resolution: str,
        lighting_enabled: bool,
        alpha_prompt_choice: str,
        edited_w: int,
        edited_h: int,
        input_w: int,
        input_h: int,
        target_w: int,
        target_h: int,
        edit_type: str,
        rgba_bytes: bytes,
        edited_path: Optional[Path],
        matte_path: Optional[Path],
        rgba_path: Optional[Path],
        json_path: Optional[Path],
    ) -> tuple[str, str]:
        """
        レスポンステキストとJSONログを構築

        Returns:
            (info_text, json_log)
        """
        # 生成情報テキスト
        if self.app.output_manager.disable_save:
            file_info = "- ファイル保存無効（クラウドデプロイモード）"
        else:
            file_info = f"""- 編集後RGB: {edited_path if edited_path else "(保存なし)"}
- アルファマット: {matte_path}
- RGBA合成: {rgba_path}
- メタデータ: {json_path}"""

        info_text = f"""### 処理完了 ✅

**モデル**: {model_name}
**編集タイプ**: {edit_type}
**解像度**: {edited_w} x {edited_h} ({resolution})
**アスペクト比**: {aspect_ratio}

**出力ファイル**:
{file_info}
"""

        # JSONログ（メタデータ構築）
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "input_resolution": {"width": input_w, "height": input_h},
            "target_resolution": {"width": target_w, "height": target_h},
            "edited_resolution": {"width": edited_w, "height": edited_h},
            "edit_type": edit_type,
            "generation": {
                "model": model_name,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "lighting_enabled": lighting_enabled,
                "alpha_prompt_choice": alpha_prompt_choice,
                "file_save_enabled": not self.app.output_manager.disable_save,
            },
        }

        # メタデータJSONファイルを保存
        if json_path is not None and not self.app.output_manager.disable_save:
            metadata_for_file = metadata.copy()
            metadata_for_file["generation"]["edited_image_path"] = (
                str(edited_path) if edited_path else None
            )
            metadata_for_file["generation"]["alpha_matte_path"] = (
                str(matte_path) if matte_path else None
            )
            metadata_for_file["generation"]["rgba_output_path"] = (
                str(rgba_path) if rgba_path else None
            )
            metadata_for_file["generation"]["rgba_size_bytes"] = len(rgba_bytes)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata_for_file, f, ensure_ascii=False, indent=2)

        json_log = json.dumps(metadata, ensure_ascii=False, indent=2)
        return info_text, json_log

    def _get_process_type_number(self, process_type: str) -> int:
        """処理タイプ文字列から番号を抽出"""
        if "処理1" in process_type:
            return 1
        elif "処理2" in process_type:
            return 2
        elif "処理3" in process_type:
            return 3
        elif "処理4" in process_type:
            return 4
        else:
            return 1  # デフォルト

    def generate_optimized_prompt(
        self,
        prompt_text: str,
        optimization_level: int,
    ) -> str:
        """
        プロンプトのみを生成（画像生成なし）

        プレビュー専用メソッド。最適化されたプロンプトを生成して返す。

        Args:
            prompt_text: ユーザーの追加指示
            optimization_level: プロンプト最適化レベル（0/1/2）

        Returns:
            最適化されたプロンプト文字列
        """
        # 基本プロンプトはUPSCALE_ONLY_PROMPTのみ
        edit_prompt = UPSCALE_ONLY_PROMPT

        # プロンプト最適化（レベル0でも整合性チェックを実行）
        try:
            optimizer = PromptOptimizer(self.app.google_api_key)

            if optimization_level == 0:
                # レベル0: 整合性チェックのみ
                optimized_prompt, _ = optimizer._level_0_consistency_check(
                    edit_prompt, prompt_text
                )
            else:
                # レベル1,2: Gemini 3.0 最適化
                optimized_prompt, opt_error = optimizer.optimize(
                    edit_prompt, prompt_text, optimization_level
                )
                if opt_error:
                    return f"⚠️ 最適化エラー: {opt_error}\n\nフォールバックプロンプト:\n{optimized_prompt}"

            return optimized_prompt

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}", exc_info=True)
            # フォールバック: 整合性チェックのみ
            try:
                optimizer = PromptOptimizer(self.app.google_api_key)
                fallback_prompt, _ = optimizer._level_0_consistency_check(
                    edit_prompt, prompt_text
                )
                return f"⚠️ 最適化エラー: {str(e)}\n\nフォールバックプロンプト:\n{fallback_prompt}"
            except:
                return f"⚠️ エラー: {str(e)}\n\n基本プロンプト:\n{edit_prompt}"

    def simple_upscale(
        self,
        model_name: str,
        input_image: Optional[Image.Image],
        prompt_text: str,
        optimization_level: int,
        optimized_prompt_from_ui: str,
        aspect_ratio: str,
        resolution: str,
        enable_google_search: bool = False,
    ) -> tuple[Optional[Image.Image], str, str]:
        """
        シンプルなアップスケール処理（1回のAPI呼び出しのみ）

        Args:
            model_name: Geminiモデル名
            input_image: 入力画像
            prompt_text: ユーザーの追加指示
            optimization_level: プロンプト最適化レベル（0/1/2）
            optimized_prompt_from_ui: UI経由で既に最適化されたプロンプト（空文字列なら新規生成）
            aspect_ratio: アスペクト比
            resolution: 解像度
            enable_google_search: Google検索を有効化

        Returns:
            (output_image, output_info, optimized_prompt)
        """
        optimized_prompt_info = ""

        # 1. 入力検証
        error_result = self._validate_inputs(input_image)
        if error_result:
            return error_result

        try:
            # 2. 目標解像度を計算
            target_w, target_h = self.calculate_target_resolution(aspect_ratio, resolution)

            # 3. プロンプト最適化（UI経由で既に最適化済みの場合はスキップ）
            if optimized_prompt_from_ui and optimized_prompt_from_ui.strip():
                final_prompt = optimized_prompt_from_ui
                logger.info("Using pre-optimized prompt from UI")
            else:
                optimizer = PromptOptimizer(self.app.google_api_key)
                final_prompt, opt_error = optimizer.optimize(
                    UPSCALE_ONLY_PROMPT, prompt_text, optimization_level
                )
                if opt_error:
                    optimized_prompt_info = f"⚠️ プロンプト最適化エラー: {opt_error}"
                    logger.warning(optimized_prompt_info)

            # 4. 入力画像をRGBに変換
            if input_image.mode == "RGBA":
                background = Image.new("RGB", input_image.size, (255, 255, 255))
                background.paste(input_image, mask=input_image.split()[3])
                input_image = background
            elif input_image.mode != "RGB":
                input_image = input_image.convert("RGB")

            input_w, input_h = input_image.size

            # 5. 画像をバイナリに変換
            img_buffer = BytesIO()
            input_image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            # 6. Gemini API呼び出し
            # Google Search対応
            gen_config_dict = {
                "response_modalities": ["image"],
                "image_generation_config": types.ImageGenerationConfig(
                    aspect_ratio=aspect_ratio,
                    output_resolution=resolution,
                ),
            }

            if enable_google_search:
                gen_config_dict["tools"] = [{"google_search": {}}]
                logger.info("Google Search tool enabled")

            response = self.app.gemini_generator.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes, mime_type="image/png"
                    ),
                    types.Part.from_text(text=final_prompt),
                ],
                config=types.GenerateContentConfig(**gen_config_dict),
            )

            # 7. 出力画像を取得
            output_img_data = None
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    output_img_data = base64.b64decode(part.inline_data.data)
                    break

            if not output_img_data:
                return None, "❌ エラー: 画像生成に失敗しました", final_prompt

            output_img = Image.open(BytesIO(output_img_data))
            output_w, output_h = output_img.size

            # 8. ファイル保存
            output_path = None
            if not self.app.output_manager.disable_save:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                unique_id = "".join(
                    random.choices(string.ascii_lowercase + string.digits, k=6)
                )
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"upscale_{timestamp}_{unique_id}.png"
                with open(output_path, "wb") as f:
                    f.write(output_img_data)

            # 9. レスポンス構築
            info_text = f"""### 処理完了 ✅

**モデル**: {model_name}
**処理タイプ**: 生成アップスケール
**入力解像度**: {input_w} x {input_h}
**出力解像度**: {output_w} x {output_h} ({resolution})
**アスペクト比**: {aspect_ratio}
{f"**プロンプト最適化**: レベル{optimization_level}" if optimization_level > 0 else ""}
{"**Google Search**: 有効" if enable_google_search else ""}
{f"**⚠️ 最適化警告**: {optimized_prompt_info}" if optimized_prompt_info else ""}

**出力ファイル**:
- アップスケール画像: {output_path if output_path else "（ファイル保存無効）"}
"""

            return output_img, info_text, final_prompt

        except Exception as e:
            logger.error(f"Upscale failed: {e}", exc_info=True)
            error_text = f"❌ エラー: {str(e)}"
            return None, error_text, ""

    def edit_with_alpha_matte(
        self,
        process_type: str,
        model_name: str,
        input_image: Optional[Image.Image],
        prompt_text: str,
        optimization_level: int,
        optimized_prompt_from_ui: str,  # NEW PARAMETER
        aspect_ratio: str,
        resolution: str,
        lighting_enabled: bool,
        alpha_prompt_choice: str,
        save_edited_image: bool,
    ) -> tuple[
        Optional[Image.Image], Optional[Image.Image], Optional[Image.Image], str, str
    ]:
        """
        マルチターン編集 + RGBA合成（2段階実行対応）

        2つのモードをサポート:
        1. プロンプト生成のみ（optimized_prompt_from_ui が空）
        2. 画像生成実行（optimized_prompt_from_ui に値がある）

        Args:
            process_type: 処理タイプ（"処理1: ...", "処理2: ...", etc.）
            optimization_level: プロンプト最適化レベル（0/1/2）
            optimized_prompt_from_ui: UIから渡された最適化済みプロンプト

        Returns:
            (output_img1, output_img2, output_img3, info_text, json_log)
        """
        # 0. プロンプト生成のみモード（画像生成なし）
        if not optimized_prompt_from_ui or optimized_prompt_from_ui.strip() == "":
            # 入力検証（画像は不要）
            optimized_prompt = self.generate_optimized_prompt(
                process_type,
                prompt_text,
                optimization_level,
                lighting_enabled,
                alpha_prompt_choice,
            )
            # 画像生成せず、プロンプトのみを返す
            return (
                None,  # edited_image
                None,  # alpha_matte
                None,  # rgba_image
                "✅ プロンプトを生成しました。内容を確認して「編集開始」をクリックしてください。",  # info_text
                optimized_prompt,  # json_log → optimized_prompt display
            )

        # 1. 入力検証（画像生成モードの場合）
        validation_error = self._validate_inputs(input_image)
        if validation_error is not None:
            return validation_error

        # 処理タイプ番号を取得
        process_num = self._get_process_type_number(process_type)

        # 処理4は未実装
        if process_num == 4:
            return (
                None,
                None,
                None,
                "❌ 処理4（色温度調整）は現在未実装です。他の処理タイプを選択してください。",
                "",
            )

        try:
            # 2. マルチターン編集実行（最適化済みプロンプトを使用）
            (
                edited_result,
                alpha_result,
                rgba_result,
                input_size,
                target_size,
                edit_type,
                optimized_prompt_info,
            ) = self._execute_multi_turn_edit(
                input_image,
                model_name,
                "",  # prompt_text: 空文字列（既に最適化済み）
                0,  # optimization_level: 0（最適化スキップ）
                process_num,
                aspect_ratio,
                resolution,
                lighting_enabled,
                alpha_prompt_choice,
                pre_optimized_prompt=optimized_prompt_from_ui,  # NEW PARAMETER
            )

            # 3. エラーチェック
            if edited_result is None:
                return None, None, None, "エラー: 編集画像を取得できませんでした", optimized_prompt_from_ui

            # 処理3の場合はアルファマットなしが正常
            if alpha_result is None and process_num != 3:
                edited_img, _, _, _ = edited_result
                return (
                    edited_img,
                    None,
                    None,
                    "エラー: アルファマット画像を取得できませんでした",
                    optimized_prompt_from_ui,
                )

            # 4. 結果の展開
            edited_img, edited_img_data, edited_w, edited_h = edited_result
            input_w, input_h = input_size
            target_w, target_h = target_size

            # 処理3（アップスケールのみ）の場合は簡易処理
            if process_num == 3:
                # アルファマットなし：edited_imgのみ保存
                if not self.app.output_manager.disable_save:
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    import random
                    import string
                    unique_id = "".join(
                        random.choices(string.ascii_lowercase + string.digits, k=6)
                    )
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    edited_path = output_dir / f"upscale_{timestamp}_{unique_id}.png"
                    with open(edited_path, "wb") as f:
                        f.write(edited_img_data)

                    info_text = f"""### 処理完了 ✅

**モデル**: {model_name}
**処理タイプ**: 処理3（生成アップスケール）
**解像度**: {edited_w} x {edited_h} ({resolution})
**アスペクト比**: {aspect_ratio}
{f"**プロンプト最適化**: レベル{optimization_level}" if optimization_level > 0 else ""}
{f"**⚠️ 最適化警告**: {optimized_prompt_info}" if optimized_prompt_info else ""}

**出力ファイル**:
- アップスケール画像: {edited_path}
"""
                else:
                    info_text = f"""### 処理完了 ✅

**モデル**: {model_name}
**処理タイプ**: 処理3（生成アップスケール）
**解像度**: {edited_w} x {edited_h} ({resolution})
**ファイル保存**: 無効（クラウドデプロイモード）
"""

                return edited_img, None, None, info_text, optimized_prompt_from_ui

            # 処理1,2の場合：アルファマット処理
            alpha_matte_img, alpha_matte_data = alpha_result
            rgba_img, rgba_bytes = rgba_result

            # 5. ファイル保存
            edited_path, matte_path, rgba_path, json_path = self._save_outputs(
                edited_img_data, alpha_matte_data, rgba_bytes, save_edited_image
            )

            # 6. レスポンス構築
            info_text, json_log = self._build_response(
                model_name,
                aspect_ratio,
                resolution,
                lighting_enabled,
                alpha_prompt_choice,
                edited_w,
                edited_h,
                input_w,
                input_h,
                target_w,
                target_h,
                edit_type,
                rgba_bytes,
                edited_path,
                matte_path,
                rgba_path,
                json_path,
            )

            return edited_img, alpha_matte_img, rgba_img, info_text, optimized_prompt_from_ui

        except Exception as e:
            logger.error(f"Alpha matte generation failed: {e}", exc_info=True)
            error_text = f"❌ エラー: {str(e)}"
            return None, None, None, error_text, optimized_prompt_from_ui

    def create_ui(self) -> None:
        """ベーシック編集タブのUIを作成"""
        with gr.Tab(
            TAB_BASIC_EDIT.display_name,
            id=TAB_BASIC_EDIT.key,
            elem_id=TAB_BASIC_EDIT.elem_id,
        ):
            gr.Markdown("""
            # ベーシック編集

            シンプルな1回編集に特化したタブ。入力画像を指定解像度にアップスケールします。

            **処理フロー**:
            1. 入力画像を読み込み
            2. プロンプト最適化（アップスケール + ユーザー追加指示）
            3. Gemini API呼び出し（1回のみ）
            4. 出力画像を保存・表示
            """)

            with gr.Row():
                # 左カラム: 入力エリア
                with gr.Column(scale=1):
                    # モデル選択
                    # デフォルトモデルを選択（gemini-3-pro-image-preview を優先）
                    default_model = "gemini-3-pro-image-preview"
                    if default_model in self.app.gemini_models:
                        model_default_value = default_model
                    elif len(self.app.gemini_models) > 1:
                        model_default_value = self.app.gemini_models[1]
                    else:
                        model_default_value = self.app.gemini_models[0]

                    model_name = gr.Dropdown(
                        label="モデル",
                        choices=self.app.gemini_models,
                        value=model_default_value,
                        info="Gemini画像生成モデル",
                    )

                    # 入力画像
                    input_image = gr.Image(
                        label="入力画像",
                        type="pil",
                        sources=["upload", "clipboard"],
                    )

                    # 追加指示
                    prompt_text = gr.Textbox(
                        label="追加指示（オプション）",
                        placeholder="例: 髪の毛の細かい部分を重視してください",
                        lines=3,
                    )

                    # 最適化されたプロンプト表示 (NEW)
                    optimized_prompt_display = gr.Textbox(
                        label="最適化されたプロンプト",
                        placeholder="「プロンプト生成」または「編集開始」をクリックすると、最適化されたプロンプトが表示されます",
                        lines=5,
                        interactive=True,
                        info="生成前にプロンプトを確認・編集できます",
                    )

                    # プロンプト最適化レベル
                    optimization_level = gr.Radio(
                        label="プロンプト最適化レベル",
                        choices=[
                            ("レベル0: 最適化なし（整合性チェックのみ）", 0),
                            ("レベル1: Gemini 3.0 自動最適化（推奨）", 1),
                            ("レベル2: Gemini 3.0 誇張表現追加", 2),
                        ],
                        value=1,
                        info="Gemini 3.0でプロンプトを最適化します",
                    )

                    # アスペクト比
                    aspect_ratio = gr.Dropdown(
                        label="アスペクト比",
                        choices=[
                            "1:1",
                            "2:3",
                            "3:2",
                            "3:4",
                            "4:3",
                            "4:5",
                            "5:4",
                            "9:16",
                            "16:9",
                            "21:9",
                        ],
                        value=self.app.default_aspect_ratio,
                        info="出力画像の縦横比",
                    )

                    # 解像度
                    resolution = gr.Dropdown(
                        label="解像度",
                        choices=["1K", "2K", "4K"],
                        value="1K",
                        info="出力画像の解像度",
                    )

                    # ツールオプション
                    with gr.Accordion("ツールオプション", open=False):
                        enable_google_search = gr.Checkbox(
                            label="Google Search",
                            value=False,
                            info="Google検索でリアルタイム情報を取得（Gemini 3 Pro Image推奨）"
                        )

                    # ボタン
                    with gr.Row():
                        edit_button = gr.Button("編集開始", variant="primary")
                        generate_prompt_button = gr.Button("プロンプト生成", variant="secondary", size="sm")  # NEW
                        reset_button = gr.Button("リセット")

                # 右カラム: 出力エリア
                with gr.Column(scale=1):
                    output_img = gr.Image(label="出力画像（アップスケール後）", type="pil")
                    output_info = gr.Markdown(label="生成情報")

            # イベントハンドラ
            # プロンプト生成ボタン
            generate_prompt_button.click(
                fn=self.generate_optimized_prompt,
                inputs=[
                    prompt_text,
                    optimization_level,
                ],
                outputs=[optimized_prompt_display],
            )

            # 編集開始ボタン
            edit_button.click(
                fn=self.simple_upscale,
                inputs=[
                    model_name,
                    input_image,
                    prompt_text,
                    optimization_level,
                    optimized_prompt_display,
                    aspect_ratio,
                    resolution,
                    enable_google_search,
                ],
                outputs=[
                    output_img,
                    output_info,
                    optimized_prompt_display,
                ],
            )

            reset_button.click(
                fn=lambda: (
                    None,  # input_image
                    "",  # prompt_text
                    "",  # optimized_prompt_display
                    "1:1",  # aspect_ratio
                    "1K",  # resolution
                    False,  # enable_google_search
                    None,  # output_img
                    "",  # output_info
                ),
                inputs=[],
                outputs=[
                    input_image,
                    prompt_text,
                    optimized_prompt_display,
                    aspect_ratio,
                    resolution,
                    enable_google_search,
                    output_img,
                    output_info,
                ],
            )
