"""
Prompt Templates

プロンプトテンプレートの読み込みと管理を行うモジュール。
app_config.toml からテンプレートを読み込み、UI で使用できる形式で提供する。
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Python 3.11+ では標準ライブラリの tomllib を使用
if sys.version_info >= (3, 11):
    import tomllib
else:
    # Python 3.10以下の場合は tomli を使用
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "Python 3.10 以下では tomli パッケージが必要です。\n"
            "インストール: uv add tomli"
        )


@dataclass
class PromptTemplate:
    """プロンプトテンプレート"""
    name: str
    prompt: str
    model: str

    def to_dict(self) -> Dict[str, str]:
        """辞書形式に変換"""
        return {
            "name": self.name,
            "prompt": self.prompt,
            "model": self.model
        }


class PromptTemplateManager:
    """プロンプトテンプレートの管理クラス"""

    def __init__(self, template_file: Path = None):
        """
        Args:
            template_file: テンプレートファイルのパス（デフォルト: プロジェクトルート/app_config.toml）
        """
        if template_file is None:
            # プロジェクトルートを取得
            project_root = Path(__file__).parent.parent.parent
            template_file = project_root / "app_config.toml"

        self.template_file = template_file
        self.templates: List[PromptTemplate] = []
        self._load_templates()

    def _load_templates(self):
        """テンプレートファイルを読み込む"""
        if not self.template_file.exists():
            print(f"Warning: Template file not found: {self.template_file}")
            # デフォルトテンプレートを作成
            self.templates = self._get_default_templates()
            return

        try:
            with open(self.template_file, "rb") as f:
                data = tomllib.load(f)

            # テンプレートをパース
            if "templates" in data:
                for template_data in data["templates"]:
                    template = PromptTemplate(
                        name=template_data.get("name", "Unknown"),
                        prompt=template_data.get("prompt", ""),
                        model=template_data.get("model", "gemini-2.5-flash-image")
                    )
                    self.templates.append(template)

            if not self.templates:
                print("Warning: No templates found in file. Using defaults.")
                self.templates = self._get_default_templates()

        except Exception as e:
            print(f"Error loading templates: {e}")
            self.templates = self._get_default_templates()

    def _get_default_templates(self) -> List[PromptTemplate]:
        """デフォルトのテンプレートを返す"""
        return [
            PromptTemplate(
                name="風景 - 夕焼けの山",
                prompt="A beautiful sunset over mountains",
                model="gemini-2.5-flash-image"
            ),
            PromptTemplate(
                name="動物 - 猫",
                prompt="A cute cat sleeping on a cozy sofa",
                model="gemini-2.5-flash-image"
            ),
        ]

    def get_template_names(self) -> List[str]:
        """テンプレート名のリストを取得"""
        return [template.name for template in self.templates]

    def get_template_by_name(self, name: str) -> Optional[PromptTemplate]:
        """
        名前からテンプレートを取得

        Args:
            name: テンプレート名

        Returns:
            見つかった場合は PromptTemplate、見つからない場合は None
        """
        for template in self.templates:
            if template.name == name:
                return template
        return None

    def get_all_templates(self) -> List[PromptTemplate]:
        """全テンプレートを取得"""
        return self.templates

    def get_template_choices(self) -> List[str]:
        """
        Gradio の Dropdown で使用するための選択肢リストを取得
        先頭に「選択してください」を追加
        """
        return ["選択してください"] + self.get_template_names()
