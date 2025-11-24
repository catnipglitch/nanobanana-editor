"""
Base Image Generator

画像生成の抽象基底クラスと共通ユーティリティ。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .config import GenerationConfig

# ロギング設定
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def extract_usage_metadata(response: Any) -> Optional[Dict[str, Optional[int]]]:
    """
    Gemini / Imagen レスポンスからトークン使用量メタデータを抽出する。

    Args:
        response: Google GenAI APIレスポンス

    Returns:
        使用量メタデータの辞書、または存在しない場合はNone
    """
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None

    return {
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
        "total_token_count": getattr(usage, "total_token_count", None),
    }


class BaseImageGenerator(ABC):
    """画像生成の抽象基底クラス"""

    def __init__(self, google_api_key: str):
        """
        Args:
            google_api_key: Google API Key
        """
        if not google_api_key:
            raise ValueError("Google API Key is required")

        self.google_api_key = google_api_key

    @abstractmethod
    def generate(self, config: GenerationConfig) -> Tuple[List[bytes], Dict[str, Any]]:
        """
        画像を生成する（抽象メソッド）

        Args:
            config: 生成設定

        Returns:
            (image_data_list, metadata): 画像バイトデータのリストとメタデータ
        """
        pass

    def _create_metadata_base(self, config: GenerationConfig) -> Dict[str, Any]:
        """
        共通メタデータの基本構造を作成

        Args:
            config: 生成設定

        Returns:
            メタデータの基本辞書
        """
        return {
            "model_type": config.model_type.value,
            "model_name": config.model_name,
            "prompt": config.prompt,
            "aspect_ratio": config.aspect_ratio,
            "auth_method": "api_key",
        }
