"""
Basic Edit Tab

ベーシック編集タブ（人物のアルファチャンネル切り抜き・背景除去）
"""

import gradio as gr
import logging
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

from .base_tab import BaseTab
from ...core.tab_specs import TAB_BASIC_EDIT

if TYPE_CHECKING:
    from ..gradio_app import NanobananaApp

logger = logging.getLogger(__name__)







#入力画像1は人物のポートレートです。  
#広告素材として利用できるよう、人物を丁寧に切り抜き、背景の要素をすべて取り除きなさい。  
#ライティングは元の写真の状態を忠実に再現し、光の方向、強さ、色温度、肌の反射などを変更してはならない。  
#人物の姿勢や構図は元のまま保持し、体型・表情・衣服の色を変えずに自然に保ちなさい。  
#人物の輪郭や髪の毛の細部は滑らかに処理し、透明背景でノイズのない広告向けの素材として仕上げなさい。  
#最終的な画像は指定された出力サイズに合わせて高品質にアップスケールしなさい。


#✦ パターン１
#元の写真を忠実に活かす・切り抜き＋元のライティング＋姿勢そのまま

UPSCALE_PROMPT_1 = (
    "Input Image 1 is a portrait of a person.  "
    "Prepare it as an advertising asset by carefully extracting the person and removing all background elements.  "
    "Reproduce the original lighting faithfully, without altering the direction, intensity, color temperature, or reflections on the skin.  "
    "Keep the posture and composition exactly as in the original image, preserving the person’s body shape, expression, and clothing colors.  "
    "Maintain smooth edges and detailed hair strands, and output a clean, transparent background suitable for compositing.  "
    "Finally, upscale the image to match the required output resolution with high quality."
)


#入力画像1は人物のポートレートです。  
#広告素材として利用できるよう、人物のみを丁寧に切り抜き、背景を完全に取り除きなさい。  
#ライティングは均一でフラットなスタジオ光に整え、影の方向性が強まる照明やレンブラントライトのような演出を避けること。  
#人物はスタジオで自然に立っている姿勢として再構築しなさい。  
#両足に重心を均等に置き、腕は身体から少し離し、表情はナチュラルでニュートラルな素材向けとすること。  
#元の写真に写っていない下半身や手足などがある場合は、人物の体格と雰囲気に合わせて自然に想像し補完しなさい。  
#人物の特徴や衣服の色は変えず、広告合成に適した透明背景で仕上げること。  
#最終画像は出力サイズに合わせて高品質にアップスケールしなさい。



#✦ パターン２
#切り抜き＋スタジオ風ライティング化（姿勢はそのまま）

UPSCALE_PROMPT_2 = (
    "Input Image 1 is a portrait of a person.  "
    "Extract the person cleanly and remove the background to prepare the subject for advertising use.  "
    "Adjust the lighting to a flat, even studio style, avoiding dramatic shadows or directional highlights.  "
    "Ensure the illumination and color temperature remain neutral so the figure blends naturally with any presentation or design material.  "
    "Preserve the original posture and composition without altering the subject’s features or color tones.  "
    "Finally, upscale the image to the required output size with high resolution."
)



#入力画像1は人物のポートレートです。  
#広告素材として利用できるよう、人物のみを丁寧に切り抜き、背景を完全に取り除きなさい。  
#ライティングは均一でフラットなスタジオ光に整え、影の方向性が強まる照明やレンブラントライトのような演出を避けること。  
#人物はスタジオで自然に立っている姿勢として再構築しなさい。  
#両足に重心を均等に置き、腕は身体から少し離し、表情はナチュラルでニュートラルな素材向けとすること。  
#元の写真に写っていない下半身や手足などがある場合は、人物の体格と雰囲気に合わせて自然に想像し補完しなさい。  
#人物の特徴や衣服の色は変えず、広告合成に適した透明背景で仕上げること。  
#最終画像は出力サイズに合わせて高品質にアップスケールしなさい。

#パターン３
#**切り抜き＋スタジオ立ち姿（ポーズ補完あり）
#※見えない部分は自然に再構築する広告特化モデル**
UPSCALE_PROMPT_3 = (
    "Input Image 1 is a portrait of a person.  "
    "Extract the subject cleanly and remove all background elements to prepare the figure for advertising use.  "
    "Adjust the lighting to a flat, uniform studio style, avoiding dramatic shadows or Rembrandt-style highlights.  "
    "Reconstruct the person as standing naturally in a studio setting, with weight evenly distributed on both feet, arms slightly away from the body, and a neutral, natural expression suitable for commercial assets.  "
    "If any body parts are missing—such as the lower body, hands, or legs—extend and complete them naturally based on the subject’s physique and appearance.  "
    "Preserve the subject’s inherent features and clothing colors, and output a clean transparent background for easy compositing.  "
    "Finally, upscale the image to the required output resolution with high quality."
)

# シンプルなアップスケールプロンプト定義（未使用、サンプルコードから移植）
UPSCALE_PROMPT_4 = (
""
  )



# アルファマットプロンプト定義（サンプルコードから移植）

# キャラクター用（イラスト）
# 参考訳：
# 入力画像のキャラクターの、プロフェッショナルな高精度グレースケールアルファマットを生成してください。
# 背景は純粋な黒（#000000）である必要があります。
# キャラクターの体、髪、装着しているアクセサリーは白でレンダリングしてください。
# 重要：半透明部分（細い髪の毛、レース生地、透明プラスチックなど）は、
# 不透明度レベルを示す正確なグレーの階調を使用して表現してください。
# 影、床の反射、床に置かれた切り離されたオブジェクト（かばんや被っていない帽子など）は
# 厳密に除外してマスクアウトしてください。
# 出力は、プロフェッショナルな合成に適した、クリーンでノイズのないセグメンテーションマスクである必要があります。
ALPHA_MATTE_PROMPT_CHARACTER = (
    "A professional high-precision grayscale alpha matte of the character from the input image. "
    "The background must be pure black (Hex #000000). "
    "The character's body, hair, and worn accessories must be rendered in white. "
    "Crucially, represent semi-transparency (e.g., fine hair strands, lace fabric, transparent plastic) "
    "using accurate shades of gray to indicate opacity levels. "
    "Strictly exclude and mask out any cast shadows, floor reflections, and detached objects placed on the floor "
    "(such as bags or hats not being worn). "
    "The output should be a clean, noise-free segmentation mask suitable for professional compositing."
)

# 人物用（実写）- V1
# 参考訳：
# 入力画像の人物の、プロフェッショナルな高精度グレースケールアルファマットを生成してください。
# 背景は純粋な黒（#000000）である必要があります。
# 人物の肌（顔、腕、脚、体）は、ソリッドな不透明の白でレンダリングしてください - 肌は決して透明ではありません。
# 髪、衣服、アクセサリーは白でレンダリングしてください。
# グレーの階調を使用した半透明表現は、次の場合のみ適用してください：
# - エッジの細い髪の毛
# - 透明またはシアーな生地（レース、チュール、オーガンザ）
# - 透明なアクセサリー（眼鏡、プラスチック製品）
# 肌、ソリッドな衣服、メインボディは常にソリッドな不透明の白である必要があります。
# 影、床の反射、床に置かれた切り離されたオブジェクトは厳密に除外してマスクアウトしてください。
# 出力は、プロフェッショナルな合成に適した、クリーンでノイズのないセグメンテーションマスクである必要があります。
ALPHA_MATTE_PROMPT_HUMAN = (
    "A professional high-precision grayscale alpha matte of the person from the input image. "
    "The background must be pure black (Hex #000000). "
    "The person's skin (face, arms, legs, body) must be rendered as solid opaque white - skin is never transparent. "
    "Hair, clothing, and accessories must be rendered in white. "
    "Semi-transparency using gray shades should ONLY be applied to: "
    "- Fine hair strands at the edges "
    "- Transparent or sheer fabric (lace, tulle, organza) "
    "- Transparent accessories (glasses, plastic items). "
    "Skin, solid clothing, and the main body must always be solid opaque white. "
    "Strictly exclude and mask out any cast shadows, floor reflections, and detached objects on the floor. "
    "The output should be a clean, noise-free segmentation mask suitable for professional compositing."
)

# 人物用（実写）- V2
# 参考訳：
# 人物のプロフェッショナルな高精度グレースケールアルファマットを生成してください。
# 背景は純粋な黒（#000000）である必要があります。
# 【最重要】色と透明度の分離指示：
# 被写体の元の色や影を透明度と混同しないでください。
# 暗い色（衣服の黒いストライプや暗い髪など）は透明ではなく、ソリッドな白でレンダリングする必要があります。
# 絶対的な不透明（白）領域の定義：
# 以下の領域は、内部に灰色のピクセルを一切含まない、完全に均一なソリッドな不透明の白（#FFFFFF）領域としてレンダリングする必要があります：
# 1. 肌の全領域（顔、腕、体）。肌は決して透明ではありません。
# 2. 衣服の全領域（すべてのパターンやストライプを含む）。生地はソリッドです。
# 3. 髪の塊のメインボディ。
# 4. ソリッドなアクセサリーや眼鏡のフレーム。
# 眼鏡の特別ルール：
# 肌の上に位置する透明なレンズは、下の肌が不透明であるため、ソリッドな白でレンダリングする必要があります。
# 半透明（グレー）の限定的な適用：
# エッジの部分でのみ、グレーの階調を使用して半透明を正確に表現してください：
# - 背景に移行する細い個別の髪の毛
# - 本当にシアーな生地（レースなど）が背景と接するエッジ
# 最終出力は、メイン被写体がソリッドな白いシルエットで、グレーがエッジの精細化にのみ使用される、
# クリーンなセグメンテーションマスクである必要があります。
ALPHA_MATTE_PROMPT_HUMAN_V2 = (
    "A professional high-precision grayscale alpha matte of the person. "
    "Background must be pure black (#000000). "
    # --- 【最重要】色と透明度の分離指示 ---
    "CRITICAL: Do NOT confuse the subject's original colors or shadows with transparency. "
    "Dark colors (like black stripes on clothes or dark hair) are NOT transparent and must be rendered as solid white. "
    # --- 絶対的な不透明（白）領域の定義 ---
    "The following areas must be rendered as a completely uniform, solid opaque white (#FFFFFF) area with absolutely NO gray pixels inside: "
    "1. The entire area of skin (face, arms, body). Skin is never transparent. "
    "2. The entire area of clothing, INCLUDING all patterns and stripes. The fabric is solid. "
    "3. The main body of the hair mass. "
    "4. Solid accessories and frames of glasses. "
    # --- 眼鏡の特別ルール（ユーザー意図への対応） ---
    "Special Rule for Glasses: Transparent lenses located ON TOP OF the skin area must be rendered as solid white, because the skin underneath is opaque. "
    # --- 半透明（グレー）の限定的な適用 ---
    "Only use shades of gray for accurately representing semi-transparency at the very edges: "
    "- Fine, individual hair strands transitioning to the background. "
    "- Edges of truly sheer fabrics (like lace) where they meet the background. "
    "The final output must be a clean segmentation mask where the main subject is a solid white silhouette, and gray is used only for edge refinement."
)

# 人物用（実写）- 推奨（汎用版）
# 参考訳：
# 入力画像の人物の、プロフェッショナルな高精度グレースケールアルファマットを生成してください。
# 背景は純粋な黒（#000000）である必要があります。
# 【汎用化】色と透明度の分離指示：
# 特定の柄ではなく「暗い色や影、柄」全般を対象にします
# 被写体の元の色、照明の影、暗いテクスチャを透明度と混同しないでください。
# 暗い領域（黒い衣服、暗い髪、深い影、プリントパターンなど）は透明ではなく、
# ソリッドな白でレンダリングする必要があります。
# 絶対的な不透明（白）領域の定義：
# 以下の領域は、内部に灰色のピクセルを一切含まない、完全に均一なソリッドな不透明の白（#FFFFFF）領域としてレンダリングする必要があります：
# 1. 肌の全領域。肌は決して透明ではありません。
# 2. 衣服の全領域（色、パターン、プリント、ロゴに関係なく）。生地自体はソリッドです。
# 3. 髪の塊のメインボディ。
# 4. ソリッドなアクセサリーや眼鏡のフレーム。
# 眼鏡・透過物の汎用ルール：
# 肌や衣服の上に位置する透明なアイテム（眼鏡のレンズなど）は、
# 下のオブジェクトが不透明であるため、ソリッドな白でレンダリングする必要があります。
# 半透明（グレー）の限定的な適用：
# エッジの部分でのみ、グレーの階調を使用して半透明を正確に表現してください：
# - 背景に移行する細い個別の髪の毛
# - 本当にシアーな生地（レースなど）が背景と接するエッジ
# 最終出力は、メイン被写体がソリッドな白いシルエットで、グレーがエッジの精細化にのみ使用される、
# クリーンなセグメンテーションマスクである必要があります。
ALPHA_MATTE_PROMPT_HUMAN_GENERIC_BOTU = (
    "A professional high-precision grayscale alpha matte of the person from the input image. "
    "Background must be pure black (#000000). "
    # --- 【汎用化】色と透明度の分離指示 ---
    # 特定の柄ではなく「暗い色や影、柄」全般を対象にします
    "CRITICAL: Do NOT confuse the subject's original colors, lighting shadows, or dark textures with transparency. "
    "Dark areas (such as black clothing, dark hair, deep shadows, or printed patterns) are NOT transparent and must be rendered as solid white. "
    # --- 絶対的な不透明（白）領域の定義 ---
    "The following areas must be rendered as a completely uniform, solid opaque white (#FFFFFF) area with absolutely NO gray pixels inside: "
    "1. The entire area of skin. Skin is never transparent. "
    "2. The entire area of clothing, regardless of its color, pattern, print, or logo. The fabric itself is solid. "
    "3. The main body of the hair mass. "
    "4. Solid accessories and frames of glasses. "
    # --- 眼鏡・透過物の汎用ルール ---
    "Rule for Transparent Objects: Transparent items (like glasses lenses) located ON TOP OF the skin or clothing must be rendered as solid white, because the object underneath is opaque. "
    # --- 半透明（グレー）の限定的な適用 ---
    "Only use shades of gray for accurately representing semi-transparency at the very edges: "
    "- Fine, individual hair strands transitioning to the background. "
    "- Edges of truly sheer fabrics (like lace) where they meet the background. "
    "The final output must be a clean segmentation mask where the main subject is a solid white silhouette, and gray is used only for edge refinement."
)
ALPHA_MATTE_PROMPT_HUMAN_GENERIC = (
    "A professional high-precision grayscale alpha matte of the person from the input image. "
    "Background must be pure black (#000000). "
    # --- 【汎用化】色と透明度の分離指示 ---
    # 特定の柄ではなく「暗い色や影、柄」全般を対象にします
    "CRITICAL: Do NOT confuse the subject's original colors, lighting shadows, or dark textures with transparency. "
    "Dark areas (such as black clothing, dark hair, deep shadows, or printed patterns) are NOT transparent and must be rendered as solid white. "
    # --- 絶対的な不透明（白）領域の定義 ---
    "The following areas must be rendered as a completely uniform, solid opaque white (#FFFFFF) area with absolutely NO gray pixels inside: "
    "1. The entire area of skin. Skin is never transparent. "
    "2. The entire area of clothing, regardless of its color, pattern, print, or logo. The fabric itself is solid. "
    "3. The main body of the hair mass. "
    # --- 半透明（グレー）の限定的な適用 ---
    "Only use shades of gray for accurately representing semi-transparency at the very edges: "
    "- Fine, individual hair strands transitioning to the background. "
    "- Edges of truly sheer fabrics (like lace) where they meet the background. "
    "The final output must be a clean segmentation mask where the main subject is a solid white silhouette, and gray is used only for edge refinement."
)


class BasicEditTab(BaseTab):
    """ベーシック編集タブ（人物のアルファチャンネル切り抜き）"""

    def __init__(self, app: "NanobananaApp"):
        super().__init__(app)
        # アルファマットプロンプトマップ
        self.ALPHA_MATTE_PROMPTS = {
            "人物用（実写）- 推奨": ALPHA_MATTE_PROMPT_HUMAN_GENERIC,
            "人物用（実写）- V2": ALPHA_MATTE_PROMPT_HUMAN_V2,
            "人物用（実写）- V1": ALPHA_MATTE_PROMPT_HUMAN,
            "キャラクター用（イラスト）": ALPHA_MATTE_PROMPT_CHARACTER,
        }

    @staticmethod
    def calculate_target_resolution(
        aspect_ratio: str, resolution: str
    ) -> tuple[int, int]:
        """アスペクト比と解像度から目標解像度を計算する"""
        aspect_map = {
            "1:1": (1, 1),
            "2:3": (2, 3),
            "3:2": (3, 2),
            "3:4": (3, 4),
            "4:3": (4, 3),
            "4:5": (4, 5),
            "5:4": (5, 4),
            "9:16": (9, 16),
            "16:9": (16, 9),
            "21:9": (21, 9),
        }

        resolution_base = {"1K": 1024, "2K": 2048, "4K": 4096}

        w_ratio, h_ratio = aspect_map[aspect_ratio]
        base = resolution_base[resolution]

        # アスペクト比に応じて解像度を計算
        if w_ratio >= h_ratio:
            width = base
            height = int(base * h_ratio / w_ratio)
        else:
            height = base
            width = int(base * w_ratio / h_ratio)

        return width, height

    @staticmethod
    def determine_edit_prompt(
        input_size: tuple[int, int],
        target_size: tuple[int, int],
        enable_lighting: bool = True,
    ) -> tuple[str, str]:
        """
        入力画像サイズと目標サイズを比較して、適切な編集プロンプトを生成する

        Args:
            input_size: 入力画像サイズ (width, height)
            target_size: 目標サイズ (width, height)
            enable_lighting: ライティング調整を有効にするか（現在は未使用、UPSCALE_PROMPT_3に統合済み）

        Returns:
            (編集プロンプト, 編集タイプ)
        """
        input_w, input_h = input_size
        target_w, target_h = target_size

        # アスペクト比を計算（小数点4桁で比較）
        input_ratio = round(input_w / input_h, 4)
        target_ratio = round(target_w / target_h, 4)

        # サイズが完全に一致 - UPSCALE_PROMPT_3を使用（シャープニングも含む）
        if input_w == target_w and input_h == target_h:
            prompt = UPSCALE_PROMPT_3
            return prompt, "sharpen"

        # 縮小が必要（入力が目標より大きい）- UPSCALE_PROMPT_3を使用
        if input_w > target_w or input_h > target_h:
            prompt = UPSCALE_PROMPT_3
            return prompt, "downscale"

        # 拡大が必要（入力が目標より小さく、比率が同じ）- UPSCALE_PROMPT_3を使用
        if abs(input_ratio - target_ratio) < 0.01:  # ほぼ同じ比率
            prompt = UPSCALE_PROMPT_3
            return prompt, "upscale"

        # 生成拡張が必要（入力が目標より小さく、比率が異なる）- UPSCALE_PROMPT_3を使用
        prompt = UPSCALE_PROMPT_3
        return prompt, "generative_expand"

    @staticmethod
    def compose_alpha_channel(rgb_img: Image.Image, alpha_bytes: bytes) -> bytes:
        """
        RGB画像とアルファマット画像を合成してRGBA PNG画像を生成する

        Args:
            rgb_img: RGB画像
            alpha_bytes: アルファマット画像のバイナリデータ

        Returns:
            RGBA PNG形式のバイナリデータ
        """
        # アルファマット画像を読み込み
        alpha_img = Image.open(BytesIO(alpha_bytes))

        # グレースケールに変換
        if alpha_img.mode != "L":
            alpha_img = alpha_img.convert("L")

        # サイズが一致しない場合はリサイズ
        if rgb_img.size != alpha_img.size:
            alpha_img = alpha_img.resize(rgb_img.size, Image.Resampling.LANCZOS)

        # RGBA画像を作成
        rgba_img = rgb_img.convert("RGBA")
        rgba_img.putalpha(alpha_img)

        # PNG形式でバイナリ出力
        output_buffer = BytesIO()
        rgba_img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()

    def edit_with_alpha_matte(
        self,
        process_type: str,
        model_name: str,
        input_image: Optional[Image.Image],
        prompt_text: str,
        aspect_ratio: str,
        resolution: str,
        lighting_enabled: bool,
        alpha_prompt_choice: str,
        save_edited_image: bool,
    ) -> tuple[
        Optional[Image.Image], Optional[Image.Image], Optional[Image.Image], str, str
    ]:
        """
        マルチターン編集 + RGBA合成

        Returns:
            (output_img1, output_img2, output_img3, info_text, json_log)
        """
        # 1. 入力検証: APIキー
        if not self.app.google_api_key or self.app.gemini_generator is None:
            error_text = """❌ エラー: APIキーが設定されていません

**Settings タブ** でGoogle API Keyを設定してください。

1. Settings タブを開く
2. APIキーを入力
3. 「接続テスト」ボタンで確認
4. 「APIキーを適用」ボタンで適用
"""
            logger.error("API key not configured")
            return None, None, None, error_text, ""

        # 2. 入力検証: 入力画像
        if input_image is None:
            return None, None, None, "⚠ 入力画像をアップロードしてください", ""

        try:
            # 3. RGB変換（アルファチャンネル削除）
            if input_image.mode == "RGBA":
                background = Image.new("RGB", input_image.size, (255, 255, 255))
                background.paste(input_image, mask=input_image.split()[3])
                input_image = background
            elif input_image.mode != "RGB":
                input_image = input_image.convert("RGB")

            input_w, input_h = input_image.size

            # 4. 目標解像度計算
            target_w, target_h = self.calculate_target_resolution(
                aspect_ratio, resolution
            )

            # 5. 編集プロンプト生成
            edit_prompt, edit_type = self.determine_edit_prompt(
                (input_w, input_h), (target_w, target_h), lighting_enabled
            )

            # 追加指示があれば追加
            if prompt_text and prompt_text.strip():
                edit_prompt += f"\n\nAdditional instructions: {prompt_text.strip()}"

            # 6. 入力画像をバイナリ化
            img_buffer = BytesIO()
            input_image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            # 7. Geminiチャット作成
            client = genai.Client(api_key=self.app.google_api_key, vertexai=False)

            chat = client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio, image_size=resolution
                    ),
                ),
            )

            # 8. 1ターン目: 画像編集
            logger.info(f"Sending turn 1: Image editing (type: {edit_type})")
            response_1 = chat.send_message(
                [
                    edit_prompt,
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                ]
            )

            # 編集画像を取得
            edited_img_data = None
            for part in response_1.parts:
                if part.text is not None:
                    logger.info(f"API response (turn 1): {part.text}")
                elif hasattr(part, "inline_data") and part.inline_data:
                    data_field = part.inline_data.data
                    if isinstance(data_field, str):
                        edited_img_data = base64.b64decode(data_field)
                    else:
                        edited_img_data = data_field

            if edited_img_data is None:
                return None, None, None, "エラー: 編集画像を取得できませんでした", ""

            # 編集画像をPIL Imageに変換
            edited_img = Image.open(BytesIO(edited_img_data))
            edited_w, edited_h = edited_img.size
            logger.info(f"Edited image size: {edited_w}x{edited_h}")

            # 9. 2ターン目: アルファマット生成
            alpha_prompt = self.ALPHA_MATTE_PROMPTS[alpha_prompt_choice]
            logger.info("Sending turn 2: Alpha matte generation")
            response_2 = chat.send_message(alpha_prompt)

            # アルファマット画像を取得
            alpha_matte_data = None
            for part in response_2.parts:
                if part.text is not None:
                    logger.info(f"API response (turn 2): {part.text}")
                elif hasattr(part, "inline_data") and part.inline_data:
                    data_field = part.inline_data.data
                    if isinstance(data_field, str):
                        alpha_matte_data = base64.b64decode(data_field)
                    else:
                        alpha_matte_data = data_field

            if alpha_matte_data is None:
                return (
                    edited_img,
                    None,
                    None,
                    "エラー: アルファマット画像を取得できませんでした",
                    "",
                )

            # アルファマット画像をPIL Imageに変換（表示用）
            alpha_matte_img = Image.open(BytesIO(alpha_matte_data))

            # 10. RGBA合成（ローカル処理）
            rgba_bytes = self.compose_alpha_channel(edited_img, alpha_matte_data)
            rgba_img = Image.open(BytesIO(rgba_bytes))

            # 11. ファイル保存
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            import random
            import string

            unique_id = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=6)
            )
            base_filename = f"alpha_matte_{timestamp}_{unique_id}"

            # ファイル保存処理（HF Spacesでは無効化）
            edited_path = None
            matte_path = None
            rgba_path = None
            json_path = None

            if not self.app.output_manager.disable_save:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)

                # 編集画像を保存（オプション）
                if save_edited_image:
                    edited_path = output_dir / f"{base_filename}_edited.png"
                    with open(edited_path, "wb") as f:
                        f.write(edited_img_data)

                # アルファマット保存
                matte_path = output_dir / f"{base_filename}_matte.png"
                with open(matte_path, "wb") as f:
                    f.write(alpha_matte_data)

                # RGBA画像保存
                rgba_path = output_dir / f"{base_filename}_rgba.png"
                with open(rgba_path, "wb") as f:
                    f.write(rgba_bytes)

                # メタデータ保存
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "input_resolution": {"width": input_w, "height": input_h},
                    "target_resolution": {"width": target_w, "height": target_h},
                    "edited_resolution": {"width": edited_w, "height": edited_h},
                    "edit_type": edit_type,
                    "generation": {
                        "model": model_name,
                        "aspect_ratio": aspect_ratio,
                        "resolution": resolution,
                        "lighting_enabled": lighting_enabled,
                        "alpha_prompt_choice": alpha_prompt_choice,
                        "edited_image_path": str(edited_path) if edited_path else None,
                        "alpha_matte_path": str(matte_path) if matte_path else None,
                        "rgba_output_path": str(rgba_path) if rgba_path else None,
                        "rgba_size_bytes": len(rgba_bytes),
                    },
                }

                # メタデータJSONを保存
                json_path = output_dir / f"{base_filename}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

            # 12. 生成情報テキスト
            if self.app.output_manager.disable_save:
                file_info = "- ファイル保存無効（クラウドデプロイモード）"
            else:
                file_info = f"""- 編集後RGB: {edited_path if edited_path else "(保存なし)"}
- アルファマット: {matte_path}
- RGBA合成: {rgba_path}
- メタデータ: {json_path}"""

            info_text = f"""### 処理完了 ✅

**モデル**: {model_name}
**編集タイプ**: {edit_type}
**解像度**: {edited_w} x {edited_h} ({resolution})
**アスペクト比**: {aspect_ratio}

**出力ファイル**:
{file_info}
"""

            # 13. JSONログ（メタデータ構築）
            metadata_for_display = {
                "timestamp": datetime.now().isoformat(),
                "input_resolution": {"width": input_w, "height": input_h},
                "target_resolution": {"width": target_w, "height": target_h},
                "edited_resolution": {"width": edited_w, "height": edited_h},
                "edit_type": edit_type,
                "generation": {
                    "model": model_name,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "lighting_enabled": lighting_enabled,
                    "alpha_prompt_choice": alpha_prompt_choice,
                    "file_save_enabled": not self.app.output_manager.disable_save,
                },
            }
            json_log = json.dumps(metadata_for_display, ensure_ascii=False, indent=2)

            return edited_img, alpha_matte_img, rgba_img, info_text, json_log

        except Exception as e:
            logger.error(f"Alpha matte generation failed: {e}", exc_info=True)
            error_text = f"❌ エラー: {str(e)}"
            return None, None, None, error_text, ""

    def create_ui(self) -> None:
        """ベーシック編集タブのUIを作成"""
        with gr.Tab(
            TAB_BASIC_EDIT.display_name,
            id=TAB_BASIC_EDIT.key,
            elem_id=TAB_BASIC_EDIT.elem_id,
        ):
            gr.Markdown("""
            # ベーシック編集

            人物のアルファチャンネル切り抜き（背景除去）機能を提供します。

            **処理フロー**:
            1. 入力画像を指定解像度にアップスケール + 明瞭度向上
            2. グレースケールアルファマットを生成
            3. RGBA画像として合成・保存
            """)

            with gr.Row():
                # 左カラム: 入力エリア
                with gr.Column(scale=1):
                    # 処理選択
                    process_type = gr.Dropdown(
                        label="処理選択",
                        choices=["人物のアルファチャンネル切り抜き（背景除去）"],
                        value="人物のアルファチャンネル切り抜き（背景除去）",
                        info="現在は1種類のみ（将来拡張予定）",
                    )

                    # モデル選択
                    model_name = gr.Dropdown(
                        label="モデル",
                        choices=self.app.gemini_models,
                        value=self.app.gemini_models[1]
                        if len(self.app.gemini_models) > 1
                        else self.app.gemini_models[0],
                        info="Gemini画像生成モデル",
                    )

                    # 入力画像
                    input_image = gr.Image(
                        label="入力画像",
                        type="pil",
                        sources=["upload", "clipboard"],
                    )

                    # 追加指示
                    prompt_text = gr.Textbox(
                        label="追加指示（オプション）",
                        placeholder="例: 髪の毛の細かい部分を重視してください",
                        lines=3,
                    )

                    # アスペクト比
                    aspect_ratio = gr.Dropdown(
                        label="アスペクト比",
                        choices=[
                            "1:1",
                            "2:3",
                            "3:2",
                            "3:4",
                            "4:3",
                            "4:5",
                            "5:4",
                            "9:16",
                            "16:9",
                            "21:9",
                        ],
                        value=self.app.default_aspect_ratio,
                        info="出力画像の縦横比",
                    )

                    # 解像度
                    resolution = gr.Dropdown(
                        label="解像度",
                        choices=["1K", "2K", "4K"],
                        value="1K",
                        info="出力画像の解像度",
                    )

                    # 詳細設定
                    with gr.Accordion("詳細設定", open=True):
                        lighting_enabled = gr.Checkbox(
                            label="ライティング調整を有効化",
                            value=True,
                            info="背景をグレーに置き換え、フラットな照明を適用",
                        )

                        alpha_prompt_choice = gr.Dropdown(
                            label="アルファマット生成方式",
                            choices=list(self.ALPHA_MATTE_PROMPTS.keys()),
                            value="人物用（実写）- 推奨",
                            info="アルファマット生成のプロンプト選択",
                        )

                        save_edited_image = gr.Checkbox(
                            label="編集画像（RGB）を保存",
                            value=True,
                            info="中間生成物のRGB画像をファイルに保存",
                        )

                    # ボタン
                    with gr.Row():
                        edit_button = gr.Button("編集開始", variant="primary")
                        reset_button = gr.Button("リセット")

                # 右カラム: 出力エリア
                with gr.Column(scale=1):
                    output_img1 = gr.Image(label="出力画像1: 編集後RGB画像", type="pil")
                    output_img2 = gr.Image(
                        label="出力画像2: アルファマット（グレースケール）", type="pil"
                    )
                    output_img3 = gr.Image(
                        label="出力画像3: RGBA合成画像（最終出力）", type="pil"
                    )

                    output_info = gr.Markdown(label="生成情報")

                    with gr.Accordion("JSONログ", open=False):
                        output_json = gr.Code(language="json", label="メタデータ")

            # イベントハンドラ
            edit_button.click(
                fn=self.edit_with_alpha_matte,
                inputs=[
                    process_type,
                    model_name,
                    input_image,
                    prompt_text,
                    aspect_ratio,
                    resolution,
                    lighting_enabled,
                    alpha_prompt_choice,
                    save_edited_image,
                ],
                outputs=[
                    output_img1,
                    output_img2,
                    output_img3,
                    output_info,
                    output_json,
                ],
            )

            reset_button.click(
                fn=lambda: (
                    None,
                    "",
                    "1:1",
                    "1K",
                    True,
                    "人物用（実写）- 推奨",
                    True,
                    None,
                    None,
                    None,
                    "",
                    "",
                ),
                inputs=[],
                outputs=[
                    input_image,
                    prompt_text,
                    aspect_ratio,
                    resolution,
                    lighting_enabled,
                    alpha_prompt_choice,
                    save_edited_image,
                    output_img1,
                    output_img2,
                    output_img3,
                    output_info,
                    output_json,
                ],
            )
