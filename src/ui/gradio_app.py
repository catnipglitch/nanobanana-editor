"""
Gradio UI Application (5-Tab Architecture with Tab Classes)

nanobabana-editorのGradioインターフェース（タブクラス分離版）。
5つのタブで構成:
- Tab 1: 画像生成（Gemini）
- Tab 2: 画像生成（Imagen）
- Tab 3: 参照画像ベース生成
- Tab 4: エージェント支援編集
- Tab 5: Settings（APIキー管理）
"""

import gradio as gr
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# ロギング設定
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.generators import (
    GenerationConfig, ModelType,
    GeminiImageGenerator, ImagenImageGenerator, TestImageGenerator
)
from src.core.output_manager import OutputManager
from src.core.prompt_templates import PromptTemplateManager
from src.core.model_specs import GeminiModelRegistry, ImagenModelRegistry

# タブクラスをインポート
from src.ui.tabs import (
    GeminiTab,
    ImagenTab,
    ReferenceTab,
    AgentTab,
    SettingsTab
)

# 環境変数を読み込み
load_dotenv()


class NanobabanaApp:
    """nanobabana-editorアプリケーション（4タブ版）"""

    def __init__(self, test_mode: bool = False):
        """
        初期化

        Args:
            test_mode: テストモード（APIキーなしでUIのみ起動）
        """
        self.test_mode = test_mode

        # 認証情報を取得
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        # APIキー未設定フラグ（警告表示用）
        self.api_key_missing = not self.google_api_key and not test_mode

        # テストモードでない場合のみ、ジェネレーターを初期化
        if not test_mode:
            # APIキーがある場合のみジェネレーターを初期化
            if self.google_api_key:
                try:
                    # 分離されたジェネレータ
                    self.gemini_generator = GeminiImageGenerator(google_api_key=self.google_api_key)
                    self.imagen_generator = ImagenImageGenerator(google_api_key=self.google_api_key)
                    self.test_generator = TestImageGenerator(google_api_key="dummy_key_for_test")
                    logger.info("Image generators initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize generators: {e}")
                    self.gemini_generator = None
                    self.imagen_generator = None
                    self.test_generator = None
                    self.api_key_missing = True
            else:
                logger.warning("GOOGLE_API_KEY not found in environment variables")
                self.gemini_generator = None
                self.imagen_generator = None
                self.test_generator = None
        else:
            self.gemini_generator = None
            self.imagen_generator = None
            self.test_generator = None
            print("⚠ テストモード: 画像生成機能は無効です")

        self.output_manager = OutputManager(output_dir="output")
        self.template_manager = PromptTemplateManager()

        # Geminiタブ用モデルリスト
        self.gemini_models = GeminiModelRegistry.get_model_ids()

        # Imagenタブ用モデルリスト
        self.imagen_models = ImagenModelRegistry.get_model_ids()

        # デフォルト設定
        self.default_aspect_ratio = "1:1"
        self.default_number_of_images = 1
        self.default_safety_filter = "block_only_high"
        self.default_person_generation = "allow_all"  # Issue #8: 全年齢可をデフォルトに

        # タブクラスをインスタンス化
        self.gemini_tab = GeminiTab(self)
        self.imagen_tab = ImagenTab(self)
        self.reference_tab = ReferenceTab(self)
        self.agent_tab = AgentTab(self)
        self.settings_tab = SettingsTab(self)

        # 初期タブを読み込み（APIキー未設定時は強制的にSettings）
        self.default_tab = self._load_default_tab()

    def _load_default_tab(self) -> str:
        """
        app_config.tomlから初期タブを読み込む
        APIキー未設定時は強制的にSettings

        Returns:
            タブの key 文字列（Gradio 5.x の selected パラメータ用、gr.Tab の id に対応）
        """
        try:
            import sys
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                try:
                    import tomli as tomllib
                except ImportError:
                    logger.warning("tomli not available, using default tab")
                    from src.core.tab_specs import TAB_SETTINGS, TAB_GEMINI
                    return TAB_SETTINGS.key if self.api_key_missing else TAB_GEMINI.key

            with open(project_root / "app_config.toml", "rb") as f:
                config = tomllib.load(f)

            tab_key = config.get("ui", {}).get("default_tab", "gemini")

            # TabRegistry でキーの有効性を確認
            from src.core.tab_specs import TabRegistry, TAB_SETTINGS, TAB_GEMINI
            tab_spec = TabRegistry.get_tab_by_key(tab_key)
            default_key = tab_spec.key if tab_spec else TAB_GEMINI.key

            # APIキー未設定時は強制的にSettings
            if self.api_key_missing:
                logger.info("API key missing, forcing Settings tab")
                return TAB_SETTINGS.key

            logger.info(f"Default tab set to: {default_key} (key: {default_key})")
            return default_key

        except Exception as e:
            logger.warning(f"Failed to load default tab config: {e}")
            from src.core.tab_specs import TAB_SETTINGS, TAB_GEMINI
            return TAB_SETTINGS.key if self.api_key_missing else TAB_GEMINI.key

    def apply_template(self, template_name: str):
        """プロンプトテンプレートを適用（Gemini/Imagenタブで共有）"""
        if template_name == "選択してください":
            return ""

        template = self.template_manager.get_template_by_name(template_name)
        if template:
            return template.prompt
        else:
            return ""

    def create_ui(self):
        """Gradio UIを作成（5タブ構成：タブクラス分離版）"""
        with gr.Blocks(title="nanobabana-editor", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# nanobabana-editor")
            gr.Markdown("画像生成・編集ツール - Gemini & Imagen 完全分離アーキテクチャ")

            # APIキー未設定時の警告バナー
            if self.api_key_missing:
                gr.Markdown("""
                ⚠️ **警告**: GOOGLE_API_KEY が設定されていません。

                画像生成を行うには、**Settings タブ** でAPIキーを入力してください。
                """, elem_id="warning-banner")

            # デフォルトタブを設定（APIキー未設定時はSettings強制）
            with gr.Tabs(selected=self.default_tab):
                # 各タブクラスのcreate_ui()を呼び出し
                self.gemini_tab.create_ui()
                self.imagen_tab.create_ui()
                self.reference_tab.create_ui()
                self.agent_tab.create_ui()
                self.settings_tab.create_ui()

        return demo

    def launch(self, **kwargs):
        """Gradio UIを起動"""
        demo = self.create_ui()
        demo.launch(**kwargs)


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="nanobabana-editor Gradio UI (4-Tab Architecture)")
    parser.add_argument("--test", action="store_true", help="テストモード（APIキーなし）")
    parser.add_argument("--share", action="store_true", help="公開リンクを生成")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="サーバー名")
    parser.add_argument("--server-port", type=int, default=7860, help="サーバーポート")

    args = parser.parse_args()

    app = NanobabanaApp(test_mode=args.test)
    app.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port
    )


if __name__ == "__main__":
    main()
