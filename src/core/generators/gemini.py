"""
Gemini Image Generator

Gemini専用の画像生成実装。
"""

import base64
import logging
from typing import Any, Dict, List, Tuple

from google import genai
from google.genai import types

from .base import BaseImageGenerator, extract_usage_metadata
from .config import GenerationConfig
from ..model_specs import ModelRegistry

logger = logging.getLogger(__name__)


class GeminiImageGenerator(BaseImageGenerator):
    """Gemini専用画像生成クラス"""

    def generate(self, config: GenerationConfig) -> Tuple[List[bytes], Dict[str, Any]]:
        """
        Geminiで画像を生成

        Args:
            config: 生成設定

        Returns:
            ([image_data], metadata): 画像バイトデータとメタデータ（常に1枚）
        """
        logger.info(f"=== Gemini Image Generation Start ===")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Prompt: '{config.prompt[:50]}...'")
        logger.info(f"Aspect ratio: {config.aspect_ratio}")

        # Geminiは常に1枚のみ生成
        if config.number_of_images != 1:
            logger.warning(f"Gemini supports only 1 image per request. Ignoring number_of_images={config.number_of_images}")

        # Google GenAI クライアント作成（APIキー認証）
        client = genai.Client(
            api_key=self.google_api_key,
            vertexai=False  # Gemini Developer APIを使用
        )

        # Phase 2.6: image_sizeと参照画像のサポート
        # モデルがimage_sizeをサポートしているか確認
        model_spec = ModelRegistry.get_model_spec(config.model_name)
        supports_image_size = model_spec and model_spec.supports_image_size

        image_config_params = {"aspect_ratio": config.aspect_ratio}
        if config.image_size and supports_image_size:
            image_config_params["image_size"] = config.image_size
            logger.info(f"Using image_size: {config.image_size}")
        elif config.image_size and not supports_image_size:
            logger.warning(f"Model {config.model_name} does not support image_size parameter. Ignoring image_size={config.image_size}")

        # Phase 2.6: 参照画像がある場合はcontentsを拡張
        contents = []
        if config.reference_images:
            logger.info(f"Using {len(config.reference_images)} reference images")
            contents.append(config.prompt)
            contents.extend(config.reference_images)
        else:
            contents = config.prompt

        # Phase 3.0: Google検索ツールの設定
        gen_config_dict = {
            "response_modalities": ["TEXT", "IMAGE"],  # Phase 2.6: TEXTも取得
            "image_config": types.ImageConfig(**image_config_params),
        }

        if config.enable_google_search:
            gen_config_dict["tools"] = [{"google_search": {}}]
            logger.info("Google Search tool enabled")

        logger.info(f"Gemini API parameters: model={config.model_name}, prompt='{config.prompt[:50]}...', aspect_ratio={config.aspect_ratio}, image_size={config.image_size}, reference_images_count={len(config.reference_images) if config.reference_images else 0}, enable_google_search={config.enable_google_search}, response_modalities=['TEXT', 'IMAGE']")

        try:
            # Geminiで画像生成（アスペクト比、解像度、参照画像、Google検索を指定）
            response = client.models.generate_content(
                model=config.model_name,
                contents=contents,
                config=types.GenerateContentConfig(**gen_config_dict),
            )

            logger.info(f"Gemini API call successful")

            usage_metadata = extract_usage_metadata(response)

            # 画像データとテキスト応答を抽出
            image_data_list = []
            text_responses = []  # Phase 2.6: テキスト応答を保存
            for candidate in response.candidates:
                # candidate.content が None の場合をチェック
                if candidate.content is None:
                    logger.warning(f"Candidate content is None, skipping")
                    continue

                for part in candidate.content.parts:
                    # Phase 2.6: テキスト部分を抽出
                    if hasattr(part, 'text') and part.text:
                        text_responses.append(part.text)
                        logger.info(f"Extracted text response: {part.text[:100]}...")

                    # 画像データを抽出
                    if hasattr(part, 'inline_data') and part.inline_data:
                        data_field = part.inline_data.data

                        # データがバイトまたはBase64文字列の場合
                        if isinstance(data_field, bytes):
                            image_data = data_field
                        elif isinstance(data_field, str):
                            # Base64文字列をデコード
                            image_data = base64.b64decode(data_field)
                        else:
                            raise ValueError(f"Unexpected data type: {type(data_field)}")

                        image_data_list.append(image_data)
                        logger.info(f"Extracted image data: {len(image_data)} bytes")

            if not image_data_list:
                raise ValueError("No image data found in response. Response may have been blocked by safety filters.")

            # メタデータ作成
            metadata = self._create_metadata_base(config)
            metadata.update({
                "generator": "gemini",
                "number_of_images": 1,
                "number_of_generated_images": len(image_data_list),
            })

            if usage_metadata:
                metadata["usage_metadata"] = usage_metadata

            # Phase 2.6: テキスト応答をメタデータに追加
            if text_responses:
                metadata["text_response"] = "\n".join(text_responses)

            logger.info(f"=== Gemini Image Generation Complete ===")
            return image_data_list, metadata

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}", exc_info=True)
            raise
