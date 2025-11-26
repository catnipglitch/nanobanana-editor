"""
Generation Configuration

画像生成の設定パラメータを管理するモジュール。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from PIL import Image

from ..model_specs import ModelType


@dataclass
class GenerationConfig:
    """画像生成の設定パラメータ"""
    model_type: ModelType
    model_name: str
    prompt: str

    # Imagen固有のパラメータ
    number_of_images: int = 1
    aspect_ratio: str = "1:1"
    safety_filter_level: str = "block_only_high"
    person_generation: str = "allow_adult"
    seed: Optional[int] = None
    enhance_prompt: bool = False
    add_watermark: bool = False

    # Gemini固有のパラメータ
    image_size: Optional[str] = None  # Phase 2.6: 解像度選択（1K/2K/4K）
    reference_images: Optional[List[Image.Image]] = None  # Phase 2.6: 参照画像（最大14枚）
    enable_google_search: bool = False  # Phase 3.0: Google検索ツール有効化

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換"""
        config_dict = {
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "number_of_images": self.number_of_images,
            "aspect_ratio": self.aspect_ratio,
            "safety_filter_level": self.safety_filter_level,
            "person_generation": self.person_generation,
            "enhance_prompt": self.enhance_prompt,
            "add_watermark": self.add_watermark,
        }
        if self.seed is not None:
            config_dict["seed"] = self.seed
        if self.image_size is not None:
            config_dict["image_size"] = self.image_size
        if self.reference_images is not None:
            config_dict["reference_images_count"] = len(self.reference_images)
        if self.enable_google_search:
            config_dict["enable_google_search"] = self.enable_google_search
        return config_dict
