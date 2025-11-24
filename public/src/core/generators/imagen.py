"""
Imagen Image Generator

Imagen専用の画像生成実装。
"""

import logging
from typing import Any, Dict, List, Tuple

from google import genai
from google.genai import types

from .base import BaseImageGenerator, extract_usage_metadata
from .config import GenerationConfig

logger = logging.getLogger(__name__)


class ImagenImageGenerator(BaseImageGenerator):
    """Imagen専用画像生成クラス"""

    def generate(self, config: GenerationConfig) -> Tuple[List[bytes], Dict[str, Any]]:
        """
        Imagenで画像を生成

        Args:
            config: 生成設定

        Returns:
            (image_data_list, metadata): 画像バイトデータのリストとメタデータ
        """
        logger.info(f"=== Imagen Image Generation Start ===")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Prompt: '{config.prompt[:50]}...'")
        logger.info(f"Number of images: {config.number_of_images}")
        logger.info(f"Aspect ratio: {config.aspect_ratio}")

        # seed と add_watermark は排他的
        if config.seed is not None and config.add_watermark:
            logger.warning("seed and add_watermark are mutually exclusive. Disabling add_watermark.")
            config.add_watermark = False

        # Google GenAI クライアント作成（APIキー認証）
        client = genai.Client(
            api_key=self.google_api_key,
            vertexai=False  # Gemini Developer APIを使用
        )

        # GenerateImagesConfig を作成
        config_dict = {
            "number_of_images": config.number_of_images,
            "aspect_ratio": config.aspect_ratio,
        }

        # オプション: seed
        if config.seed is not None:
            config_dict["seed"] = config.seed

        logger.info(f"Imagen API parameters: model={config.model_name}, prompt='{config.prompt[:50]}...', config={config_dict}")

        try:
            # Imagenで画像生成
            response = client.models.generate_images(
                model=config.model_name,
                prompt=config.prompt,
                config=types.GenerateImagesConfig(**config_dict),
            )

            logger.info(f"Imagen API call successful: {len(response.generated_images)} images generated")

            usage_metadata = extract_usage_metadata(response)

            # 画像データを抽出
            image_data_list = []
            for generated_image in response.generated_images:
                image_bytes = generated_image.image.image_bytes
                image_data_list.append(image_bytes)
                logger.info(f"Extracted image data: {len(image_bytes)} bytes")

            # メタデータ作成
            metadata = self._create_metadata_base(config)
            metadata.update({
                "generator": "imagen",
                "number_of_images": config.number_of_images,
                "number_of_generated_images": len(image_data_list),
                "safety_filter_level": config.safety_filter_level,
                "person_generation": config.person_generation,
                "enhance_prompt": config.enhance_prompt,
                "add_watermark": config.add_watermark,
            })

            if config.seed is not None:
                metadata["seed"] = config.seed

            if usage_metadata:
                metadata["usage_metadata"] = usage_metadata

            logger.info(f"=== Imagen Image Generation Complete ===")
            return image_data_list, metadata

        except Exception as e:
            logger.error(f"Imagen API call failed: {e}", exc_info=True)
            raise
