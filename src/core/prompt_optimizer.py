"""
Prompt Optimizer

Gemini 3.0を使用したプロンプト最適化サービス
"""

import logging
from datetime import datetime
from typing import Optional

from google import genai

try:
    from zoneinfo import ZoneInfo
    ZONEINFO_AVAILABLE = True
except ImportError:
    ZONEINFO_AVAILABLE = False

logger = logging.getLogger(__name__)

if not ZONEINFO_AVAILABLE:
    logger.warning("zoneinfo not available, datetime context will use UTC for fallback")


class PromptOptimizer:
    """Gemini 3.0を使用したプロンプト最適化サービス"""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: Google API Key
        """
        self.api_key = api_key
        self.model = "gemini-3-pro-preview"  # Text-only model for prompt optimization

    def optimize(
        self, base_prompt: str, user_instructions: str, level: int
    ) -> tuple[str, Optional[str]]:
        """
        プロンプトを最適化する（日時コンテキスト対応）

        Args:
            base_prompt: 処理タイプから選択された基本プロンプト
            user_instructions: ユーザーの追加指示
            level: 最適化レベル (0=なし, 1=自動最適化, 2=誇張表現)

        Returns:
            (最適化後のプロンプト, エラーメッセージ or None)
        """
        # 時間参照を検出
        has_temporal_ref = self._detect_temporal_references(base_prompt, user_instructions)

        # レベル0: 整合性チェックのみ
        if level == 0:
            prompt, error = self._level_0_consistency_check(base_prompt, user_instructions)
            # 日時コンテキストを追加
            datetime_str = self._generate_datetime_context()
            final_prompt = self._append_datetime_to_prompt(prompt, datetime_str)
            return final_prompt, error

        # レベル1,2: Gemini 3.0で最適化
        try:
            client = genai.Client(api_key=self.api_key, vertexai=False)

            # システムプロンプト選択（時間参照フラグを渡す）
            system_prompt = self._get_system_prompt(level, has_temporal_ref)

            # 最適化リクエスト（XMLコンテキスト構造）
            prompt = f"""
{system_prompt}

<context>
<base_prompt>
{base_prompt}
</base_prompt>

<user_instructions>
{user_instructions.strip() if user_instructions.strip() else "(none)"}
</user_instructions>
</context>

<instruction>
Based on the context above, generate the optimized prompt.
</instruction>
"""

            # Thinking levelをレベルに応じて設定
            # レベル1: low（高速・低コスト）
            # レベル2: デフォルト（high、創造性重視）
            # NOTE: thinking_level parameter is not yet supported in current SDK version
            # TODO: Re-enable when SDK supports thinking_level parameter
            # config = {}
            # if level == 1:
            #     config["thinking_level"] = "low"

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            # テキスト応答を取得
            optimized_prompt = None
            for part in response.candidates[0].content.parts:
                if part.text:
                    optimized_prompt = part.text.strip()
                    break

            if not optimized_prompt:
                logger.warning("Gemini 3.0 returned no text response for prompt optimization")
                fallback_prompt, _ = self._level_0_consistency_check(base_prompt, user_instructions)
                # 日時コンテキストを追加
                datetime_str = self._generate_datetime_context()
                final_prompt = self._append_datetime_to_prompt(fallback_prompt, datetime_str)
                return final_prompt, None

            # 日時コンテキストを追加
            datetime_str = self._generate_datetime_context()
            final_prompt = self._append_datetime_to_prompt(optimized_prompt, datetime_str)

            logger.info(f"Prompt optimized (level {level}, temporal_ref={has_temporal_ref}): {len(final_prompt)} chars")
            return final_prompt, None

        except Exception as e:
            logger.error(f"Prompt optimization failed (level {level}): {e}", exc_info=True)
            error_msg = f"プロンプト最適化エラー: {str(e)}"
            # フォールバック: レベル0 + 日時追加
            fallback_prompt, _ = self._level_0_consistency_check(base_prompt, user_instructions)
            datetime_str = self._generate_datetime_context()
            final_prompt = self._append_datetime_to_prompt(fallback_prompt, datetime_str)
            return final_prompt, error_msg

    def _level_0_consistency_check(
        self, base_prompt: str, user_instructions: str
    ) -> tuple[str, None]:
        """
        レベル0: 整合性チェックのみ

        基本プロンプトとユーザー指示を単純に結合。
        将来的には矛盾チェックを追加可能。

        Args:
            base_prompt: 基本プロンプト
            user_instructions: ユーザー指示

        Returns:
            (結合されたプロンプト, None)
        """
        if user_instructions and user_instructions.strip():
            combined = f"{base_prompt}\n\nAdditional instructions: {user_instructions.strip()}"
        else:
            combined = base_prompt

        logger.info(f"Prompt combined (level 0): {len(combined)} chars")
        return combined, None

    def _generate_datetime_context(self) -> str:
        """現在の日時情報を生成する（ローカルタイムゾーン）

        システムのローカルタイムゾーンを使用。
        取得できない場合はJST（Asia/Tokyo）にフォールバック。

        Returns:
            str: フォーマット例: "2025-11-28 Fri 14:30"
        """
        try:
            # まずシステムのローカルタイムゾーンで取得を試みる
            now = datetime.now()
            datetime_str = now.strftime("%Y-%m-%d %a %H:%M")
            logger.debug(f"Generated datetime context (local): {datetime_str}")
            return datetime_str

        except Exception as e:
            # ローカルタイムゾーン取得失敗時、JSTにフォールバック
            logger.warning(f"Failed to get local datetime, falling back to JST: {e}")
            try:
                if ZONEINFO_AVAILABLE:
                    jst = ZoneInfo("Asia/Tokyo")
                    now = datetime.now(jst)
                    datetime_str = now.strftime("%Y-%m-%d %a %H:%M")
                    logger.debug(f"Generated datetime context (JST fallback): {datetime_str}")
                    return datetime_str
                else:
                    # zoneinfo利用不可の場合、UTCを使用
                    now = datetime.utcnow()
                    datetime_str = now.strftime("%Y-%m-%d %a %H:%M UTC")
                    logger.warning("zoneinfo not available, using UTC")
                    return datetime_str
            except Exception as e2:
                # 最終フォールバック: UTC
                logger.error(f"All datetime context generation failed: {e2}")
                now = datetime.utcnow()
                return now.strftime("%Y-%m-%d %a %H:%M UTC")

    def _detect_temporal_references(
        self, base_prompt: str, user_instructions: str
    ) -> bool:
        """プロンプト内の時間参照を検出する

        Args:
            base_prompt: 基本プロンプト
            user_instructions: ユーザーの追加指示

        Returns:
            bool: 時間参照が検出された場合True
        """
        temporal_keywords = [
            # 日本語
            "明日", "今日", "昨日", "今夜", "今朝", "今晩",
            "午前", "午後", "朝", "昼", "夕方", "夜",
            "来週", "今週", "先週", "来月", "今月", "先月",
            "来年", "今年", "去年",
            # 英語
            "tomorrow", "today", "yesterday", "tonight",
            "morning", "afternoon", "evening", "night",
            "next week", "this week", "last week",
            "next month", "this month", "last month",
            "next year", "this year", "last year",
        ]

        combined_text = f"{base_prompt} {user_instructions}".lower()

        for keyword in temporal_keywords:
            if keyword.lower() in combined_text:
                logger.info(f"Temporal reference detected: '{keyword}'")
                return True

        return False

    def _append_datetime_to_prompt(self, prompt: str, datetime_str: str) -> str:
        """プロンプト末尾に日時情報を追加する

        Args:
            prompt: 元のプロンプト
            datetime_str: 日時文字列

        Returns:
            str: 日時情報が追加されたプロンプト
        """
        # 重複チェック: 既に "Current datetime:" が含まれている場合はスキップ
        if "Current datetime:" in prompt:
            logger.debug("Datetime info already present in prompt, skipping append")
            return prompt

        # 末尾に日時情報を追加
        final_prompt = f"{prompt}\n\nCurrent datetime: {datetime_str}"
        logger.debug(f"Appended datetime to prompt: {datetime_str}")

        return final_prompt

    def _get_system_prompt(self, level: int, has_temporal_ref: bool) -> str:
        """システムプロンプトを取得する

        Args:
            level: 最適化レベル（0/1/2）
            has_temporal_ref: 時間参照が含まれている場合True

        Returns:
            str: システムプロンプト
        """
        base_prompt = OPTIMIZATION_SYSTEM_PROMPTS.get(level, OPTIMIZATION_SYSTEM_PROMPTS[1])

        # 時間参照がない場合は、既存のシステムプロンプトをそのまま返す
        if not has_temporal_ref:
            return base_prompt

        # 時間参照がある場合、temporal_context_noteを挿入
        temporal_note = """
<temporal_context_note>
The user's prompt contains temporal references (e.g., "tomorrow", "next week").
A current datetime will be appended to the final prompt as: "Current datetime: YYYY-MM-DD DDD HH:MM"
Please ensure the optimized prompt can reference this datetime information appropriately.
</temporal_context_note>
"""

        # </output_format>の直前に挿入
        if "</output_format>" in base_prompt:
            return base_prompt.replace("</output_format>", f"{temporal_note}\n</output_format>")
        else:
            # フォールバック: 末尾に追加
            return f"{base_prompt}\n\n{temporal_note}"


# =============================================================================
# プロンプト最適化用システムプロンプト
# =============================================================================

OPTIMIZATION_SYSTEM_PROMPTS = {
    0: "",  # レベル0は最適化なし（_level_0_consistency_checkが処理）
    # レベル1: 自動最適化（基本プロンプトを保持しつつ自然に統合）
    # Level 1: Auto-optimization (Preserves base prompt while naturally integrating)
    1: """<role>
You are an expert prompt optimizer for NanoBanana image editing system powered by Gemini API.
# あなたはGemini APIを使用したNanoBanana画像編集システムの専門プロンプト最適化AIです
</role>

<task>
Optimize the provided base prompt and user instructions into a single, effective English prompt for Gemini image generation/editing API.
# 提供された基本プロンプトとユーザー指示を、Gemini画像生成/編集API向けの単一の効果的な英語プロンプトに最適化してください
</task>

<constraints>
# 実行すべきこと / DO:
DO:
- Preserve the exact technical specifications (aspect ratio, resolution, lighting settings)
  # 技術仕様を正確に保持（アスペクト比、解像度、ライティング設定）
- Integrate user instructions naturally into the base prompt
  # ユーザー指示を基本プロンプトに自然に統合
- Use clear, concise English suitable for image generation
  # 画像生成に適した明確で簡潔な英語を使用
- Prioritize base prompt intent when conflicts arise
  # 矛盾がある場合は基本プロンプトの意図を優先

# 実行してはいけないこと / DO NOT:
DO NOT:
- Add explanations, commentary, or meta-text
  # 説明、コメント、メタテキストを追加しない
- Remove or modify technical parameters
  # 技術パラメータを削除・変更しない
- Use overly verbose or flowery language
  # 過度に冗長または華美な表現を使用しない
- Output anything except the optimized prompt itself
  # 最適化されたプロンプト以外は出力しない
</constraints>

<examples>
Example 1:
Base prompt: "Upscale the input image to high resolution. Maintain color accuracy."
User instructions: "髪の毛の細かい部分を重視してください"
Optimized prompt: Upscale the input image to high resolution with enhanced detail preservation, especially for fine hair strands. Maintain color accuracy and natural texture.

Example 2:
Base prompt: "Extract person from background. Apply studio lighting."
User instructions: "自然な立ち姿で"
Optimized prompt: Extract person from background in a natural standing pose. Apply professional studio lighting with balanced exposure.
</examples>

<output_format>
Return only the optimized prompt with no prefix or explanation.
# プロンプトのみを出力（プレフィックスや説明なし）
</output_format>""",
    # レベル2: 誇張表現追加（視覚的に印象的な詳細な表現）
    # Level 2: Enhanced expressions (Visually impressive detailed descriptions)
    2: """<role>
You are an expert prompt optimizer for NanoBanana image editing system, specializing in enhanced, vivid descriptions.
# あなたはNanoBanana画像編集システムの専門プロンプト最適化AIで、拡張された鮮明な表現を専門としています
</role>

<task>
Transform the provided base prompt and user instructions into a highly detailed, visually impressive English prompt with enhanced descriptive language.
# 提供された基本プロンプトとユーザー指示を、拡張された説明的言語を用いて、非常に詳細で視覚的に印象的な英語プロンプトに変換してください
</task>

<constraints>
# 実行すべきこと / DO:
DO:
- Amplify visual details, lighting, texture, and quality descriptors
  # 視覚的ディテール、照明、質感、品質の記述を増幅
- Use professional photography terminology (e.g., "pixel-perfect", "high-fidelity", "meticulously")
  # プロ写真用語を使用（例: "pixel-perfect", "high-fidelity", "meticulously"）
- Expand and enrich user instructions with vivid language
  # ユーザー指示を鮮明な言語で拡張・強化
- Preserve all technical specifications exactly
  # すべての技術仕様を正確に保持

# 実行してはいけないこと / DO NOT:
DO NOT:
- Modify aspect ratio, resolution, or lighting parameters
  # アスペクト比、解像度、ライティングパラメータを変更しない
- Add explanations or commentary
  # 説明やコメントを追加しない
- Output anything except the optimized prompt itself
  # 最適化されたプロンプト以外は出力しない
</constraints>

<examples>
Example 1:
Base prompt: "Remove background. Apply studio lighting."
User instructions: "きれいに切り抜いて"
Optimized prompt: Meticulously remove all background elements with pixel-perfect precision, extracting the subject with clean, sharp edges. Apply professional studio lighting with controlled highlights, soft shadows, and balanced exposure for a high-quality result.

Example 2:
Base prompt: "Upscale to 2K resolution."
User instructions: "ディテールを保持"
Optimized prompt: Upscale to 2K resolution with exceptional detail preservation, maintaining crisp textures, fine patterns, and natural color fidelity throughout the enhanced image.

Example 3:
Base prompt: "Generate full-body portrait with natural pose."
User instructions: "プロフェッショナルな仕上がりで"
Optimized prompt: Generate a meticulously crafted full-body portrait featuring a naturally balanced pose with professional modeling stance. Render with studio-grade quality, enhanced clarity, and refined details for a polished, high-end result.
</examples>

<output_format>
Return only the optimized prompt with no prefix or explanation.
# プロンプトのみを出力（プレフィックスや説明なし）
</output_format>""",
}
