"""
Gradio UI Tabs Package

各タブを独立したモジュールとして管理。
"""

from .base_tab import BaseTab
from .gemini_tab import GeminiTab
from .basic_edit_tab import BasicEditTab
from .reference_tab import ReferenceTab
from .multiturn_edit_tab import MultiTurnEditTab
from .layout_edit_tab import LayoutEditTab
from .outfit_change_tab import OutfitChangeTab
from .advanced_edit_tab import AdvancedEditTab
from .analysis_tab import AnalysisTab
from .agent_tab import AgentTab
from .imagen_tab import ImagenTab
from .settings_tab import SettingsTab

__all__ = [
    "BaseTab",
    "GeminiTab",
    "BasicEditTab",
    "ImagenTab",
    "ReferenceTab",
    "MultiTurnEditTab",
    "LayoutEditTab",
    "OutfitChangeTab",
    "AdvancedEditTab",
    "AnalysisTab",
    "AgentTab",
    "SettingsTab",
]
