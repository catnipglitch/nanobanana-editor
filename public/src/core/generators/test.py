"""
Test Image Generator

テスト用のダミー画像生成実装（APIコールなし）。
"""

import logging
import time
from io import BytesIO
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from .base import BaseImageGenerator
from .config import GenerationConfig

logger = logging.getLogger(__name__)


class TestImageGenerator(BaseImageGenerator):
    """テスト用画像生成クラス（APIコールなし）"""

    def generate(self, config: GenerationConfig) -> Tuple[List[bytes], Dict[str, Any]]:
        """
        テスト用のダミー画像を生成

        Args:
            config: 生成設定

        Returns:
            (image_data_list, metadata): ダミー画像バイトデータのリストとメタデータ
        """
        logger.info(f"=== Test Mode Image Generation Start ===")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Prompt: '{config.prompt[:50]}...'")
        logger.info(f"Number of images: {config.number_of_images}")

        # 3秒のダミー生成処理（本番に近いテスト）
        logger.info("Test Mode: Simulating image generation (3 seconds)...")
        time.sleep(3)

        # PIL Imageでテスト画像を生成
        image_data_list = []
        for i in range(config.number_of_images):
            # 画像サイズを計算（アスペクト比に基づく）
            aspect_ratio = config.aspect_ratio
            if aspect_ratio == "1:1":
                size = (512, 512)
            elif aspect_ratio == "16:9":
                size = (512, 288)
            elif aspect_ratio == "9:16":
                size = (288, 512)
            elif aspect_ratio == "4:3":
                size = (512, 384)
            elif aspect_ratio == "3:4":
                size = (384, 512)
            else:
                size = (512, 512)

            # テスト画像を生成
            img = Image.new('RGB', size, color=(70, 130, 180))
            draw = ImageDraw.Draw(img)

            # テキストを描画
            try:
                # システムフォントを使用（Linux/macOS/Windows共通）
                font = ImageFont.load_default()
            except Exception:
                font = None

            text_lines = [
                f"TEST MODEL OUTPUT",
                f"Image {i+1}/{config.number_of_images}",
                f"Prompt: {config.prompt[:30]}...",
                f"Aspect Ratio: {config.aspect_ratio}",
                f"Size: {size[0]}x{size[1]}",
            ]

            if config.seed is not None:
                text_lines.append(f"Seed: {config.seed}")

            y_offset = size[1] // 4
            for line in text_lines:
                draw.text((20, y_offset), line, fill=(255, 255, 255), font=font)
                y_offset += 30

            # 画像をバイトデータに変換
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            image_data_list.append(img_byte_arr.getvalue())

        # メタデータ作成
        metadata = self._create_metadata_base(config)
        metadata.update({
            "generator": "test",
            "number_of_images": config.number_of_images,
            "number_of_generated_images": len(image_data_list),
            "note": "This is a TEST MODEL output for development/debugging purposes"
        })

        if config.seed is not None:
            metadata["seed"] = config.seed

        logger.info(f"Test Mode: Generated {len(image_data_list)} images")
        logger.info(f"=== Test Mode Image Generation Complete ===")
        return image_data_list, metadata
