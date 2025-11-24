"""
Base Tab Class

全てのタブクラスの基底クラス。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..gradio_app import NanobabanaApp


class BaseTab(ABC):
    """タブの抽象基底クラス"""

    def __init__(self, app: 'NanobabanaApp'):
        """
        Args:
            app: NanobabanaApp インスタンス（ジェネレーター、設定などにアクセス）
        """
        self.app = app

    @abstractmethod
    def create_ui(self) -> None:
        """
        タブのUIを構築する（抽象メソッド）

        Gradio コンポーネントを使用してタブのUIを定義する。
        """
        pass
