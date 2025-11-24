"""
Nanobanana アルファマット生成サンプル (マルチターン版)

マルチターンチャットで画像編集とアルファマット生成を行います。

■ 機能
  - 入力: JPG/PNG画像
  - 処理1: Gemini APIで画像編集（サイズ調整・鮮明化）
  - 処理2: 同じチャットでアルファマット生成
  - 出力: RGB画像 + アルファマット → RGBA PNG

■ 画像編集処理（1ターン目）
  入力画像サイズと目標解像度を比較して、適切な編集を依頼:

  【縮小が必要】
  入力画像が目標解像度より大きい場合:
    - 「目標サイズに合わせて縮小してください」

  【拡大が必要】
  入力画像が目標解像度より小さく、比率が同じ場合:
    - 「目標サイズに合わせて拡大してください」

  【生成拡張が必要】
  入力画像が目標解像度より小さく、比率が異なる場合:
    - 「画像の外側を自然に生成拡張して目標サイズにしてください」

  【サイズ一致】
  入力画像が目標解像度と一致する場合:
    - 「この画像を少し鮮明にしてください」

■ アルファマット生成（2ターン目）
  同じチャットで1ターン目の画像に対してアルファマット生成を依頼

■ 設定
  - GOOGLE_API_KEY: .envファイルに設定
  - INPUT_IMAGE_PATH: 入力画像パス
  - ASPECT_RATIO: アスペクト比 (1:1, 16:9等)
  - RESOLUTION: 解像度 (1K, 2K, 4K)

■ 出力ファイル (output/ディレクトリ)
  - *_edited.png: 編集後のRGB画像
  - *_matte.png: 生成されたアルファマット
  - *_rgba.png: 最終的なRGBA合成画像
  - *.json: メタデータ

■ 実行例
  python samples/gemini_api_sample/nanobanana_edit_alpha_multi_turn.py
"""

import os
import json
from datetime import datetime
from pathlib import Path
from io import BytesIO
import base64
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# 環境変数を読み込み
load_dotenv()

# .env から設定を読み込み
API_KEY = os.getenv("GOOGLE_API_KEY")

# ============================================================
# 設定 (ここで変更)
# ============================================================

# 入力画像
# INPUT_IMAGE_PATH = "samples/sample_images/gemini-768x1344-woman.jpg"

# INPUT_IMAGE_PATH = "samples/sample_images/unsplash/tamara-bellis-JoKS3XweV50-unsplash.jpg"
INPUT_IMAGE_PATH = "samples/sample_images/gemini-768x1344-man.jpg"
# Geminiモデル
MODEL_ID = "gemini-3-pro-image-preview"

# アスペクト比
#   選択可能: "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
ASPECT_RATIO = "9:16"

# 解像度 (Gemini 3のみ有効)
#   選択可能: "1K", "2K", "4K"
RESOLUTION = "1K"

# 編集画像を保存するか
SAVE_EDITED_IMAGE = True

# ライティング調整を有効化
ENABLE_LIGHTING_ADJUSTMENT = True

# ============================================================


# アルファマット生成プロンプト
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
    "4. Solid accessories and frames of glasses. "
    # --- 眼鏡・透過物の汎用ルール ---
    "Rule for Transparent Objects: Transparent items (like glasses lenses) located ON TOP OF the skin or clothing must be rendered as solid white, because the object underneath is opaque. "
    # --- 半透明（グレー）の限定的な適用 ---
    "Only use shades of gray for accurately representing semi-transparency at the very edges: "
    "- Fine, individual hair strands transitioning to the background. "
    "- Edges of truly sheer fabrics (like lace) where they meet the background. "
    "The final output must be a clean segmentation mask where the main subject is a solid white silhouette, and gray is used only for edge refinement."
)


ALPHA_MATTE_PROMPT = ALPHA_MATTE_PROMPT_HUMAN_V2


# 出力設定
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def calculate_target_resolution(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    """アスペクト比と解像度から目標解像度を計算する。"""
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


def load_input_image(image_path: str) -> Image.Image:
    """入力画像を読み込んでRGB形式で変換する。"""
    img = Image.open(image_path)

    # RGBに変換（アルファチャンネルがあれば削除）
    if img.mode == "RGBA":
        # 白背景で合成
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def determine_edit_prompt(
    input_size: tuple[int, int],
    target_size: tuple[int, int],
    enable_lighting: bool = True,
) -> tuple[str, str]:
    """
    入力画像サイズと目標サイズを比較して、適切な編集プロンプトを生成する。

    Args:
        input_size: 入力画像サイズ (width, height)
        target_size: 目標サイズ (width, height)
        enable_lighting: ライティング調整を有効にするか

    Returns:
        (編集プロンプト, 編集タイプ)
    """
    input_w, input_h = input_size
    target_w, target_h = target_size

    # アスペクト比を計算（小数点4桁で比較）
    input_ratio = round(input_w / input_h, 4)
    target_ratio = round(target_w / target_h, 4)

    # 共通の前処理プロンプト
    preprocessing = "First, isolate the person by replacing the background with a clean neutral gray. "
    if enable_lighting:
        preprocessing += (
            "Apply flat, natural studio lighting to the person for even illumination. "
        )
    preprocessing += "Slightly enhance edge definition for clarity. "

    # サイズが完全に一致
    if input_w == target_w and input_h == target_h:
        prompt = (
            f"{preprocessing}"
            f"Then sharpen this image slightly to enhance overall clarity while maintaining natural appearance."
        )
        return prompt, "sharpen"

    # 縮小が必要（入力が目標より大きい）
    if input_w > target_w or input_h > target_h:
        prompt = (
            f"{preprocessing}"
            f"Then resize this image to exactly {target_w}x{target_h} pixels. "
            f"Maintain the subject's clarity and quality during resizing."
        )
        return prompt, "downscale"

    # 拡大が必要（入力が目標より小さく、比率が同じ）
    if abs(input_ratio - target_ratio) < 0.01:  # ほぼ同じ比率
        prompt = (
            f"{preprocessing}"
            f"Then upscale this image to exactly {target_w}x{target_h} pixels. "
            f"Enhance details and maintain quality during upscaling."
        )
        return prompt, "upscale"

    # 生成拡張が必要（入力が目標より小さく、比率が異なる）
    prompt = (
        f"{preprocessing}"
        f"Then expand this image to {target_w}x{target_h} pixels by generating natural extensions "
        f"around the original content. The original subject should remain centered and unchanged. "
        f"Generate realistic background that naturally extends from the existing image edges."
    )
    return prompt, "generative_expand"


def compose_alpha_channel(rgb_img: Image.Image, alpha_bytes: bytes) -> bytes:
    """
    RGB画像とアルファマット画像を合成してRGBA PNG画像を生成する。

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


def print_header(title: str) -> None:
    """ヘッダーを表示する"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_step(step_num: int, total_steps: int, message: str) -> None:
    """ステップ番号付きメッセージを表示する"""
    print(f"\n[{step_num}/{total_steps}] {message}")


def generate_output_filenames(timestamp: str) -> dict[str, Path]:
    """出力ファイル名を生成する"""
    base_name = f"nanobanana_edit_alpha_multi_turn_{timestamp}"
    return {
        "edited": OUTPUT_DIR / f"{base_name}_edited.png",
        "matte": OUTPUT_DIR / f"{base_name}_matte.png",
        "rgba": OUTPUT_DIR / f"{base_name}_rgba.png",
        "json": OUTPUT_DIR / f"{base_name}.json",
    }


def save_metadata(
    filepath: Path,
    timestamp: str,
    input_path: str,
    input_size: tuple[int, int],
    target_size: tuple[int, int],
    edit_type: str,
    edited_size: tuple[int, int],
    output_files: dict[str, Path],
    rgba_size: int,
    api_response_1: Any = None,
    api_response_2: Any = None,
) -> None:
    """メタデータをJSON形式で保存する"""
    input_w, input_h = input_size
    target_w, target_h = target_size
    edited_w, edited_h = edited_size

    # 1ターン目のレスポンスメタデータ
    response_metadata_1 = {}
    if api_response_1:
        usage = getattr(api_response_1, "usage_metadata", None)
        if usage:
            response_metadata_1["usage_metadata"] = {
                "prompt_token_count": getattr(usage, "prompt_token_count", None),
                "candidates_token_count": getattr(
                    usage, "candidates_token_count", None
                ),
                "total_token_count": getattr(usage, "total_token_count", None),
            }

    # 2ターン目のレスポンスメタデータ
    response_metadata_2 = {}
    if api_response_2:
        usage = getattr(api_response_2, "usage_metadata", None)
        if usage:
            response_metadata_2["usage_metadata"] = {
                "prompt_token_count": getattr(usage, "prompt_token_count", None),
                "candidates_token_count": getattr(
                    usage, "candidates_token_count", None
                ),
                "total_token_count": getattr(usage, "total_token_count", None),
            }

    metadata = {
        "timestamp": timestamp,
        "input_image": str(input_path),
        "input_resolution": {"width": input_w, "height": input_h},
        "target_resolution": {"width": target_w, "height": target_h},
        "edited_resolution": {"width": edited_w, "height": edited_h},
        "edit_type": edit_type,
        "generation": {
            "model": MODEL_ID,
            "aspect_ratio": ASPECT_RATIO,
            "resolution": RESOLUTION,
            "edited_image_path": str(output_files["edited"])
            if SAVE_EDITED_IMAGE
            else None,
            "alpha_matte_path": str(output_files["matte"]),
            "rgba_output_path": str(output_files["rgba"]),
            "rgba_size_bytes": rgba_size,
        },
        "api_response_turn1": response_metadata_1,
        "api_response_turn2": response_metadata_2,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    """メイン処理"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_files = generate_output_filenames(timestamp)

    # ヘッダー表示
    print_header("Nanobanana アルファマット生成サンプル (マルチターン版)")
    print(f"入力画像: {INPUT_IMAGE_PATH}")
    print(f"モデル: {MODEL_ID}")
    print(f"アスペクト比: {ASPECT_RATIO}")
    print(f"解像度: {RESOLUTION}")

    # Step 1: 入力画像を読み込み
    print_step(1, 5, "入力画像を読み込み中...")
    input_img = load_input_image(INPUT_IMAGE_PATH)
    input_w, input_h = input_img.size
    print(f"入力解像度: {input_w} x {input_h} pixels")

    # Step 2: 目標解像度を計算
    print_step(2, 5, "編集プロンプト生成中...")
    target_w, target_h = calculate_target_resolution(ASPECT_RATIO, RESOLUTION)
    print(f"目標解像度: {target_w} x {target_h} pixels")

    edit_prompt, edit_type = determine_edit_prompt(
        (input_w, input_h), (target_w, target_h), ENABLE_LIGHTING_ADJUSTMENT
    )
    print(f"編集タイプ: {edit_type}")
    print(f"ライティング調整: {'有効' if ENABLE_LIGHTING_ADJUSTMENT else '無効'}")
    print(f"プロンプト: {edit_prompt}")

    # 入力画像をバイナリ化
    img_buffer = BytesIO()
    input_img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    # Step 3: チャット作成と1ターン目（画像編集）
    print_step(3, 5, "Gemini APIで画像編集中...")

    client = genai.Client(api_key=API_KEY, vertexai=False)

    chat = client.chats.create(
        model=MODEL_ID,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=ASPECT_RATIO, image_size=RESOLUTION
            ),
        ),
    )

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
            print(f"APIレスポンス: {part.text}")
        elif hasattr(part, "inline_data") and part.inline_data:
            data_field = part.inline_data.data
            if isinstance(data_field, str):
                edited_img_data = base64.b64decode(data_field)
            else:
                edited_img_data = data_field

    if edited_img_data is None:
        print("エラー: 編集画像を取得できませんでした")
        return

    # 編集画像を保存
    edited_img = Image.open(BytesIO(edited_img_data))
    edited_w, edited_h = edited_img.size
    print(f"編集後解像度: {edited_w} x {edited_h} pixels")

    if SAVE_EDITED_IMAGE:
        with open(output_files["edited"], "wb") as f:
            f.write(edited_img_data)
        print(f"編集画像を保存: {output_files['edited']}")

    # Step 4: 2ターン目（アルファマット生成）
    print_step(4, 5, "Gemini APIでアルファマット生成中...")

    response_2 = chat.send_message(ALPHA_MATTE_PROMPT)

    # アルファマット画像を取得
    alpha_matte_data = None
    for part in response_2.parts:
        if part.text is not None:
            print(f"APIレスポンス: {part.text}")
        elif hasattr(part, "inline_data") and part.inline_data:
            data_field = part.inline_data.data
            if isinstance(data_field, str):
                alpha_matte_data = base64.b64decode(data_field)
            else:
                alpha_matte_data = data_field

    if alpha_matte_data is None:
        print("エラー: アルファマット画像を取得できませんでした")
        return

    # アルファマット画像を保存
    with open(output_files["matte"], "wb") as f:
        f.write(alpha_matte_data)
    print(f"アルファマット画像を保存: {output_files['matte']}")

    # Step 5: RGBA画像合成
    print_step(5, 5, "RGBA画像合成中...")
    rgba_bytes = compose_alpha_channel(edited_img, alpha_matte_data)

    with open(output_files["rgba"], "wb") as f:
        f.write(rgba_bytes)
    print(f"RGBA合成画像を保存: {output_files['rgba']}")
    print(f"ファイルサイズ: {len(rgba_bytes)} bytes")

    # メタデータを保存
    save_metadata(
        filepath=output_files["json"],
        timestamp=timestamp,
        input_path=INPUT_IMAGE_PATH,
        input_size=(input_w, input_h),
        target_size=(target_w, target_h),
        edit_type=edit_type,
        edited_size=(edited_w, edited_h),
        output_files=output_files,
        rgba_size=len(rgba_bytes),
        api_response_1=response_1,
        api_response_2=response_2,
    )
    print(f"\nメタデータを保存: {output_files['json']}")

    print_header("処理完了")


if __name__ == "__main__":
    main()
