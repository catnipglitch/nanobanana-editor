"""
Tab Specifications

アプリケーション内の全タブの仕様を一元管理。
タブの追加・変更時はこのファイルのみを編集する。
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class TabCategory(Enum):
    """タブのカテゴリ"""
    GENERATION = "generation"      # 画像生成系
    EDITING = "editing"            # 画像編集系
    MANAGEMENT = "management"      # 管理・設定系


@dataclass(frozen=True)
class TabSpec:
    """タブの仕様定義"""

    # 識別子（内部処理・設定ファイル用）
    key: str                    # 例: "gemini"

    # UI表示名（Gradio の label）
    display_name: str          # 例: "画像生成（Gemini）"

    # HTML要素ID（Gradio の elem_id）
    elem_id: str               # 例: "tab-gemini"

    # タブクラス名
    class_name: str            # 例: "GeminiTab"

    # カテゴリ
    category: TabCategory

    # 説明（将来的なツールチップ用）
    description: Optional[str] = None

    # 表示順序
    order: int = 0


# タブ定義（表示順序に従って定義）
TAB_GEMINI = TabSpec(
    key="gemini",
    display_name="画像生成（Gemini）",
    elem_id="tab-gemini",
    class_name="GeminiTab",
    category=TabCategory.GENERATION,
    description="Gemini モデルで画像を生成",
    order=1
)

TAB_IMAGEN = TabSpec(
    key="imagen",
    display_name="画像生成（Imagen）",
    elem_id="tab-imagen",
    class_name="ImagenTab",
    category=TabCategory.GENERATION,
    description="Imagen モデルで画像を生成",
    order=2
)

TAB_REFERENCE = TabSpec(
    key="reference",
    display_name="参照画像ベース生成",
    elem_id="tab-reference",
    class_name="ReferenceTab",
    category=TabCategory.GENERATION,
    description="参照画像を基に新しい画像を生成",
    order=3
)

TAB_AGENT = TabSpec(
    key="agent",
    display_name="エージェント支援編集",
    elem_id="tab-agent",
    class_name="AgentTab",
    category=TabCategory.EDITING,
    description="AI エージェントによる画像編集",
    order=4
)

TAB_SETTINGS = TabSpec(
    key="settings",
    display_name="Settings",
    elem_id="tab-settings",
    class_name="SettingsTab",
    category=TabCategory.MANAGEMENT,
    description="API キー管理とアプリ設定",
    order=5
)


class TabRegistry:
    """タブレジストリ（全タブの管理）"""

    _tabs: List[TabSpec] = [
        TAB_GEMINI,
        TAB_IMAGEN,
        TAB_REFERENCE,
        TAB_AGENT,
        TAB_SETTINGS,
    ]

    @classmethod
    def get_all_tabs(cls) -> List[TabSpec]:
        """全タブを表示順で取得"""
        return sorted(cls._tabs, key=lambda t: t.order)

    @classmethod
    def get_tab_by_key(cls, key: str) -> Optional[TabSpec]:
        """key からタブ仕様を取得"""
        return next((t for t in cls._tabs if t.key == key), None)

    @classmethod
    def get_tab_by_elem_id(cls, elem_id: str) -> Optional[TabSpec]:
        """elem_id からタブ仕様を取得"""
        return next((t for t in cls._tabs if t.elem_id == elem_id), None)

    @classmethod
    def get_default_tab_elem_id(cls) -> str:
        """デフォルトタブの elem_id を取得"""
        return TAB_GEMINI.elem_id

    @classmethod
    def get_key_to_elem_id_map(cls) -> dict:
        """key → elem_id のマッピングを取得"""
        return {t.key: t.elem_id for t in cls._tabs}
