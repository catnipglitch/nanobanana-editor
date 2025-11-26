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
    ANALYSIS = "analysis"          # 画像解析系
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
# Phase 3.1: 11タブ構成への拡張

TAB_GEMINI = TabSpec(
    key="gemini_gen01",
    display_name="画像生成（Gemini）",
    elem_id="tab-gemini-gen01",
    class_name="GeminiTab",
    category=TabCategory.GENERATION,
    description="Gemini モデルで画像を生成（Google検索統合済み）",
    order=1
)

TAB_BASIC_EDIT = TabSpec(
    key="gemini_edit01",
    display_name="ベーシック編集",
    elem_id="tab-gemini-edit01",
    class_name="BasicEditTab",
    category=TabCategory.EDITING,
    description="回転・反転・リサイズなど基本操作",
    order=2
)

TAB_REFERENCE = TabSpec(
    key="gemini_edit02",
    display_name="参照画像ベース生成",
    elem_id="tab-gemini-edit02",
    class_name="ReferenceTab",
    category=TabCategory.GENERATION,
    description="参照画像を基に新しい画像を生成",
    order=3
)

TAB_MULTITURN_EDIT = TabSpec(
    key="gemini_edit03",
    display_name="マルチターン編集",
    elem_id="tab-gemini-edit03",
    class_name="MultiTurnEditTab",
    category=TabCategory.EDITING,
    description="対話形式での段階的編集",
    order=4
)

TAB_LAYOUT_EDIT = TabSpec(
    key="gemini_edit04",
    display_name="レイアウト編集",
    elem_id="tab-gemini-edit04",
    class_name="LayoutEditTab",
    category=TabCategory.EDITING,
    description="キャラクターシート等の定型レイアウト",
    order=5
)

TAB_OUTFIT_CHANGE = TabSpec(
    key="gemini_edit05",
    display_name="衣装チェンジ",
    elem_id="tab-gemini-edit05",
    class_name="OutfitChangeTab",
    category=TabCategory.EDITING,
    description="キャラクター着替え専用",
    order=6
)

TAB_ADVANCED_EDIT = TabSpec(
    key="gemini_edit06",
    display_name="高度な編集",
    elem_id="tab-gemini-edit06",
    class_name="AdvancedEditTab",
    category=TabCategory.EDITING,
    description="アルファ処理・マスク編集等",
    order=7
)

TAB_ANALYSIS = TabSpec(
    key="gemini_other01",
    display_name="画像解析",
    elem_id="tab-gemini-other01",
    class_name="AnalysisTab",
    category=TabCategory.ANALYSIS,
    description="画像説明生成・ポーズ抽出",
    order=8
)

TAB_AGENT = TabSpec(
    key="gemini_other02",
    display_name="Chat / エージェント",
    elem_id="tab-gemini-other02",
    class_name="AgentTab",
    category=TabCategory.EDITING,
    description="対話型エージェントによる画像編集",
    order=9
)

TAB_IMAGEN = TabSpec(
    key="imagen_gen01",
    display_name="画像生成（Imagen）",
    elem_id="tab-imagen-gen01",
    class_name="ImagenTab",
    category=TabCategory.GENERATION,
    description="Imagen モデルで画像を生成",
    order=10
)

TAB_SETTINGS = TabSpec(
    key="settings",
    display_name="Settings",
    elem_id="tab-settings",
    class_name="SettingsTab",
    category=TabCategory.MANAGEMENT,
    description="API キー管理とアプリ設定",
    order=11
)


class TabRegistry:
    """タブレジストリ（全タブの管理）"""

    _tabs: List[TabSpec] = [
        TAB_GEMINI,
        TAB_BASIC_EDIT,
        TAB_REFERENCE,
        TAB_MULTITURN_EDIT,
        TAB_LAYOUT_EDIT,
        TAB_OUTFIT_CHANGE,
        TAB_ADVANCED_EDIT,
        TAB_ANALYSIS,
        TAB_AGENT,
        TAB_IMAGEN,
        TAB_SETTINGS,
    ]

    # Phase 3.1: 旧key → 新key マッピング（後方互換性）
    _legacy_key_mapping = {
        "gemini": "gemini_gen01",
        "imagen": "imagen_gen01",
        "reference": "gemini_edit02",
        "agent": "gemini_other02",
    }

    @classmethod
    def get_all_tabs(cls) -> List[TabSpec]:
        """全タブを表示順で取得"""
        return sorted(cls._tabs, key=lambda t: t.order)

    @classmethod
    def get_tab_by_key(cls, key: str) -> Optional[TabSpec]:
        """key からタブ仕様を取得（旧keyも認識）"""
        # 旧keyの場合は新keyに変換
        actual_key = cls._legacy_key_mapping.get(key, key)
        return next((t for t in cls._tabs if t.key == actual_key), None)

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
