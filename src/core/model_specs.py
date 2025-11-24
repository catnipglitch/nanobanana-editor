"""
Model Specifications

画像生成モデルのAPI仕様を一元管理するモジュール。
新しいモデルの追加や仕様変更を容易にするための設計。
"""

from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass


class ModelType(Enum):
    """画像生成モデルのタイプ"""
    IMAGEN = "imagen"
    GEMINI = "gemini"
    TEST = "test"  # テスト用モデル（開発・デバッグ用）


class AuthMethod(Enum):
    """認証方法"""
    GCP_PROJECT_ID = "gcp_project_id"  # GCP プロジェクトID（Vertex AI SDK）
    VERTEX_AI_API_KEY = "vertex_ai_api_key"  # Vertex AI API キー
    GEMINI_API_KEY = "gemini_api_key"  # Gemini API キー
    NONE = "none"  # 認証不要（テストモード用）


@dataclass
class AspectRatioSpec:
    """アスペクト比の仕様"""
    ratio: str  # "1:1", "16:9", etc.
    label: str  # UI表示用ラベル
    width: Optional[int] = None  # 幅（ピクセル、モデルによっては不要）
    height: Optional[int] = None  # 高さ（ピクセル、モデルによっては不要）


@dataclass
class ModelSpec:
    """モデルの仕様"""
    model_id: str  # モデルID（API呼び出し時に使用）
    model_type: ModelType  # モデルタイプ
    display_name: str  # UI表示名
    auth_methods: List[AuthMethod]  # サポートする認証方法（優先順位順）

    # サポートするパラメータ
    supports_aspect_ratio: bool = True
    supports_number_of_images: bool = False
    supports_seed: bool = False
    supports_enhance_prompt: bool = False
    supports_add_watermark: bool = False
    supports_safety_filter: bool = False
    supports_person_generation: bool = False
    supports_image_size: bool = False  # Phase 2.6: 解像度選択サポート

    # サポートするアスペクト比
    aspect_ratios: List[AspectRatioSpec] = None

    # サポートする解像度（Gemini 3 Pro Image用）
    image_sizes: List[str] = None  # ["1K", "2K", "4K"]

    # デフォルトパラメータ
    default_aspect_ratio: str = "1:1"
    default_number_of_images: int = 1
    max_number_of_images: int = 1
    default_image_size: str = "1K"  # Phase 2.6: デフォルト解像度

    def __post_init__(self):
        """初期化後の処理"""
        if self.aspect_ratios is None:
            # デフォルトのアスペクト比リスト
            self.aspect_ratios = [
                AspectRatioSpec("1:1", "1:1 (Square)"),
                AspectRatioSpec("16:9", "16:9 (Landscape)"),
                AspectRatioSpec("9:16", "9:16 (Portrait)"),
                AspectRatioSpec("4:3", "4:3 (Landscape)"),
                AspectRatioSpec("3:4", "3:4 (Portrait)"),
            ]


# ===================================================================
# アスペクト比定義
# ===================================================================

# Gemini 3 Pro Image の拡張アスペクト比（10種類）
GEMINI_3_PRO_ASPECT_RATIOS = [
    AspectRatioSpec("1:1", "1:1 (Square)"),
    AspectRatioSpec("2:3", "2:3 (Portrait)"),
    AspectRatioSpec("3:2", "3:2 (Landscape)"),
    AspectRatioSpec("3:4", "3:4 (Portrait)"),
    AspectRatioSpec("4:3", "4:3 (Landscape)"),
    AspectRatioSpec("4:5", "4:5 (Portrait)"),
    AspectRatioSpec("5:4", "5:4 (Landscape)"),
    AspectRatioSpec("9:16", "9:16 (Portrait)"),
    AspectRatioSpec("16:9", "16:9 (Landscape)"),
    AspectRatioSpec("21:9", "21:9 (Ultra Wide)"),
]

# ===================================================================
# モデル仕様定義
# ===================================================================

# Imagen 4.0 系モデル
IMAGEN_4_0_GENERATE_001 = ModelSpec(
    model_id="imagen-4.0-generate-001",
    model_type=ModelType.IMAGEN,
    display_name="Imagen 4.0 (Standard)",
    auth_methods=[AuthMethod.VERTEX_AI_API_KEY, AuthMethod.GCP_PROJECT_ID],
    supports_aspect_ratio=True,
    supports_number_of_images=True,
    supports_seed=True,
    supports_enhance_prompt=False,
    supports_add_watermark=True,
    supports_safety_filter=True,
    supports_person_generation=True,
    max_number_of_images=4,  # Vertex AI Imagen API の最大値
    aspect_ratios=[
        AspectRatioSpec("1:1", "1:1 (Square)"),
        AspectRatioSpec("16:9", "16:9 (Landscape)"),
        AspectRatioSpec("9:16", "9:16 (Portrait)"),
        AspectRatioSpec("4:3", "4:3 (Landscape)"),
        AspectRatioSpec("3:4", "3:4 (Portrait)"),
    ]
)

IMAGEN_4_0_FAST_GENERATE_001 = ModelSpec(
    model_id="imagen-4.0-fast-generate-001",
    model_type=ModelType.IMAGEN,
    display_name="Imagen 4.0 Fast",
    auth_methods=[AuthMethod.VERTEX_AI_API_KEY, AuthMethod.GCP_PROJECT_ID],
    supports_aspect_ratio=True,
    supports_number_of_images=True,
    supports_seed=True,
    supports_enhance_prompt=False,
    supports_add_watermark=True,
    supports_safety_filter=True,
    supports_person_generation=True,
    max_number_of_images=4,  # Vertex AI Imagen API の最大値
)

IMAGEN_4_0_ULTRA_GENERATE_001 = ModelSpec(
    model_id="imagen-4.0-ultra-generate-001",
    model_type=ModelType.IMAGEN,
    display_name="Imagen 4.0 Ultra",
    auth_methods=[AuthMethod.VERTEX_AI_API_KEY, AuthMethod.GCP_PROJECT_ID],
    supports_aspect_ratio=True,
    supports_number_of_images=True,
    supports_seed=True,
    supports_enhance_prompt=False,
    supports_add_watermark=True,
    supports_safety_filter=True,
    supports_person_generation=True,
    max_number_of_images=4,  # Vertex AI Imagen API の最大値
)

# Imagen 3.0 系モデル
IMAGEN_3_0_GENERATE_002 = ModelSpec(
    model_id="imagen-3.0-generate-002",
    model_type=ModelType.IMAGEN,
    display_name="Imagen 3.0 (v2)",
    auth_methods=[AuthMethod.VERTEX_AI_API_KEY, AuthMethod.GCP_PROJECT_ID],
    supports_aspect_ratio=True,
    supports_number_of_images=True,
    supports_seed=True,
    supports_enhance_prompt=False,
    supports_add_watermark=True,
    supports_safety_filter=True,
    supports_person_generation=True,
    max_number_of_images=4,
)

IMAGEN_3_0_GENERATE_001 = ModelSpec(
    model_id="imagen-3.0-generate-001",
    model_type=ModelType.IMAGEN,
    display_name="Imagen 3.0 (v1)",
    auth_methods=[AuthMethod.VERTEX_AI_API_KEY, AuthMethod.GCP_PROJECT_ID],
    supports_aspect_ratio=True,
    supports_number_of_images=True,
    supports_seed=True,
    supports_enhance_prompt=False,
    supports_add_watermark=True,
    supports_safety_filter=True,
    supports_person_generation=True,
    max_number_of_images=4,
)

IMAGEN_3_0_FAST_GENERATE_001 = ModelSpec(
    model_id="imagen-3.0-fast-generate-001",
    model_type=ModelType.IMAGEN,
    display_name="Imagen 3.0 Fast",
    auth_methods=[AuthMethod.VERTEX_AI_API_KEY, AuthMethod.GCP_PROJECT_ID],
    supports_aspect_ratio=True,
    supports_number_of_images=True,
    supports_seed=True,
    supports_enhance_prompt=False,
    supports_add_watermark=True,
    supports_safety_filter=True,
    supports_person_generation=True,
    max_number_of_images=4,
)

IMAGEN_3_0_CAPABILITY_001 = ModelSpec(
    model_id="imagen-3.0-capability-001",
    model_type=ModelType.IMAGEN,
    display_name="Imagen 3.0 Capability",
    auth_methods=[AuthMethod.VERTEX_AI_API_KEY, AuthMethod.GCP_PROJECT_ID],
    supports_aspect_ratio=True,
    supports_number_of_images=True,
    supports_seed=True,
    supports_enhance_prompt=False,
    supports_add_watermark=True,
    supports_safety_filter=True,
    supports_person_generation=True,
    max_number_of_images=4,
)

# Gemini 2.5 Flash Image
GEMINI_2_5_FLASH_IMAGE = ModelSpec(
    model_id="gemini-2.5-flash-image",
    model_type=ModelType.GEMINI,
    display_name="Gemini 2.5 Flash Image",
    auth_methods=[AuthMethod.GEMINI_API_KEY],
    supports_aspect_ratio=True,
    supports_number_of_images=False,
    supports_seed=False,
    supports_enhance_prompt=False,
    supports_add_watermark=False,
    supports_safety_filter=False,
    supports_person_generation=False,
    max_number_of_images=1,
)

# Gemini 3 Pro Image Preview
GEMINI_3_PRO_IMAGE_PREVIEW = ModelSpec(
    model_id="gemini-3-pro-image-preview",
    model_type=ModelType.GEMINI,
    display_name="Gemini 3 Pro Image Preview",
    auth_methods=[AuthMethod.GEMINI_API_KEY],
    supports_aspect_ratio=True,
    supports_number_of_images=False,
    supports_seed=False,
    supports_enhance_prompt=False,
    supports_add_watermark=False,
    supports_safety_filter=False,
    supports_person_generation=False,
    supports_image_size=True,  # Phase 2.6: 解像度選択対応
    max_number_of_images=1,
    aspect_ratios=GEMINI_3_PRO_ASPECT_RATIOS,  # 10種類の拡張アスペクト比
    image_sizes=["1K", "2K", "4K"],  # Phase 2.6: サポートする解像度
    default_image_size="1K",  # Phase 2.6: デフォルト解像度
)

# テストモデル
TEST_MODEL = ModelSpec(
    model_id="test-model",
    model_type=ModelType.TEST,
    display_name="Test Model (Development)",
    auth_methods=[AuthMethod.NONE],
    supports_aspect_ratio=True,
    supports_number_of_images=True,
    supports_seed=True,
    supports_enhance_prompt=True,
    supports_add_watermark=True,
    supports_safety_filter=True,
    supports_person_generation=True,
    max_number_of_images=8,
)


# ===================================================================
# モデルレジストリ
# ===================================================================

class ModelRegistry:
    """モデル仕様のレジストリ"""

    # 全モデルのリスト（表示順）
    _ALL_MODELS = [
        TEST_MODEL,
        GEMINI_2_5_FLASH_IMAGE,
        GEMINI_3_PRO_IMAGE_PREVIEW,
        IMAGEN_4_0_GENERATE_001,
        IMAGEN_4_0_FAST_GENERATE_001,
        IMAGEN_4_0_ULTRA_GENERATE_001,
        IMAGEN_3_0_GENERATE_002,
        IMAGEN_3_0_GENERATE_001,
        IMAGEN_3_0_FAST_GENERATE_001,
        IMAGEN_3_0_CAPABILITY_001,
    ]

    # モデルID -> ModelSpec のマッピング
    _MODEL_MAP: Dict[str, ModelSpec] = {
        spec.model_id: spec for spec in _ALL_MODELS
    }

    @classmethod
    def get_all_models(cls) -> List[ModelSpec]:
        """全モデルのリストを取得"""
        return cls._ALL_MODELS.copy()

    @classmethod
    def get_model_spec(cls, model_id: str) -> Optional[ModelSpec]:
        """モデルIDからModelSpecを取得"""
        return cls._MODEL_MAP.get(model_id)

    @classmethod
    def get_models_by_type(cls, model_type: ModelType) -> List[ModelSpec]:
        """モデルタイプ別にモデルを取得"""
        return [spec for spec in cls._ALL_MODELS if spec.model_type == model_type]

    @classmethod
    def get_model_ids(cls) -> List[str]:
        """全モデルIDのリストを取得（UI用）"""
        return [spec.model_id for spec in cls._ALL_MODELS]

    @classmethod
    def get_aspect_ratios(cls, model_id: str) -> List[str]:
        """モデルがサポートするアスペクト比のリストを取得"""
        spec = cls.get_model_spec(model_id)
        if spec and spec.aspect_ratios:
            return [ar.ratio for ar in spec.aspect_ratios]
        return ["1:1"]  # デフォルト

    @classmethod
    def get_default_aspect_ratio(cls, model_id: str) -> str:
        """モデルのデフォルトアスペクト比を取得"""
        spec = cls.get_model_spec(model_id)
        return spec.default_aspect_ratio if spec else "1:1"

    @classmethod
    def get_max_number_of_images(cls, model_id: str) -> int:
        """モデルの最大生成枚数を取得"""
        spec = cls.get_model_spec(model_id)
        return spec.max_number_of_images if spec else 1

    @classmethod
    def supports_parameter(cls, model_id: str, parameter: str) -> bool:
        """モデルが特定のパラメータをサポートしているか確認"""
        spec = cls.get_model_spec(model_id)
        if not spec:
            return False
        return getattr(spec, f"supports_{parameter}", False)


class GeminiModelRegistry:
    """Gemini専用モデルレジストリ"""

    # Geminiモデルのリスト（test-modelを含む）
    _MODELS = [
        GEMINI_3_PRO_IMAGE_PREVIEW,
        GEMINI_2_5_FLASH_IMAGE,
        TEST_MODEL,
    ]

    # モデルID -> ModelSpec のマッピング
    _MODEL_MAP: Dict[str, ModelSpec] = {
        spec.model_id: spec for spec in _MODELS
    }

    @classmethod
    def get_all_models(cls) -> List[ModelSpec]:
        """全Geminiモデルのリストを取得"""
        return cls._MODELS.copy()

    @classmethod
    def get_model_spec(cls, model_id: str) -> Optional[ModelSpec]:
        """モデルIDからModelSpecを取得"""
        return cls._MODEL_MAP.get(model_id)

    @classmethod
    def get_model_ids(cls) -> List[str]:
        """全モデルIDのリストを取得（UI用）"""
        return [spec.model_id for spec in cls._MODELS]

    @classmethod
    def get_aspect_ratios(cls, model_id: str) -> List[str]:
        """モデルがサポートするアスペクト比のリストを取得"""
        spec = cls.get_model_spec(model_id)
        if spec and spec.aspect_ratios:
            return [ar.ratio for ar in spec.aspect_ratios]
        return ["1:1"]  # デフォルト

    @classmethod
    def get_default_aspect_ratio(cls, model_id: str) -> str:
        """モデルのデフォルトアスペクト比を取得"""
        spec = cls.get_model_spec(model_id)
        return spec.default_aspect_ratio if spec else "1:1"

    @classmethod
    def get_max_number_of_images(cls, model_id: str) -> int:
        """モデルの最大生成枚数を取得（Geminiは常に1）"""
        return 1

    @classmethod
    def supports_parameter(cls, model_id: str, parameter: str) -> bool:
        """モデルが特定のパラメータをサポートしているか確認"""
        spec = cls.get_model_spec(model_id)
        if not spec:
            return False
        return getattr(spec, f"supports_{parameter}", False)


class ImagenModelRegistry:
    """Imagen専用モデルレジストリ"""

    # Imagenモデルのリスト（test-modelを含む）
    _MODELS = [
        TEST_MODEL,
        IMAGEN_4_0_GENERATE_001,
        IMAGEN_4_0_FAST_GENERATE_001,
        IMAGEN_4_0_ULTRA_GENERATE_001,
        IMAGEN_3_0_GENERATE_002,
        IMAGEN_3_0_GENERATE_001,
        IMAGEN_3_0_FAST_GENERATE_001,
        IMAGEN_3_0_CAPABILITY_001,
    ]

    # モデルID -> ModelSpec のマッピング
    _MODEL_MAP: Dict[str, ModelSpec] = {
        spec.model_id: spec for spec in _MODELS
    }

    @classmethod
    def get_all_models(cls) -> List[ModelSpec]:
        """全Imagenモデルのリストを取得"""
        return cls._MODELS.copy()

    @classmethod
    def get_model_spec(cls, model_id: str) -> Optional[ModelSpec]:
        """モデルIDからModelSpecを取得"""
        return cls._MODEL_MAP.get(model_id)

    @classmethod
    def get_model_ids(cls) -> List[str]:
        """全モデルIDのリストを取得（UI用）"""
        return [spec.model_id for spec in cls._MODELS]

    @classmethod
    def get_aspect_ratios(cls, model_id: str) -> List[str]:
        """モデルがサポートするアスペクト比のリストを取得"""
        spec = cls.get_model_spec(model_id)
        if spec and spec.aspect_ratios:
            return [ar.ratio for ar in spec.aspect_ratios]
        return ["1:1"]  # デフォルト

    @classmethod
    def get_default_aspect_ratio(cls, model_id: str) -> str:
        """モデルのデフォルトアスペクト比を取得"""
        spec = cls.get_model_spec(model_id)
        return spec.default_aspect_ratio if spec else "1:1"

    @classmethod
    def get_max_number_of_images(cls, model_id: str) -> int:
        """モデルの最大生成枚数を取得"""
        spec = cls.get_model_spec(model_id)
        return spec.max_number_of_images if spec else 1

    @classmethod
    def supports_parameter(cls, model_id: str, parameter: str) -> bool:
        """モデルが特定のパラメータをサポートしているか確認"""
        spec = cls.get_model_spec(model_id)
        if not spec:
            return False
        return getattr(spec, f"supports_{parameter}", False)
