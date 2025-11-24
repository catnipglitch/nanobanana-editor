"""
Nanobanana アルファマット生成サンプル

入力画像からプロフェッショナルなアルファマット（透過マスク）を生成します。

■ 機能
  - 入力: JPG/PNG画像（キャラクターが明確な画像推奨）
  - 処理: Gemini APIでグレースケール・アルファマット生成
  - 出力: RGB画像 + アルファマット → RGBA PNG

■ リサイズ・キャンバス拡張処理
  入力画像サイズに応じて自動的にリサイズ・キャンバス拡張を行います。

  【縮小処理】
  入力画像が目標解像度より大きい場合:
    - アスペクト比を維持して縮小
    - 警告メッセージを表示

  【キャンバス拡張処理】
  縮小後または元画像が目標解像度より小さい場合:
    - 白キャンバス中央に配置してキャンバス拡張

  例1: 640x480 → 1K(1024x1024) 指定の場合（拡張のみ）
      1. 640x480を1024x1024の白キャンバス中央に配置
      2. 処理済み画像を保存 (*_expanded.png)
      3. APIに送信してアルファマット生成
      4. 処理済み画像(RGB) + アルファマット(α) → RGBA PNG保存

  例2: 2048x3072 → 1K(1024x1024) 指定の場合（縮小→拡張）
      1. 2048x3072をアスペクト比維持で縮小 → 683x1024
      2. 683x1024を1024x1024の白キャンバス中央に配置
      3. 処理済み画像を保存 (*_expanded.png)
      4. 警告メッセージ表示
      5. APIに送信してアルファマット生成
      6. 処理済み画像(RGB) + アルファマット(α) → RGBA PNG保存

■ 設定
  - GOOGLE_API_KEY: .envファイルに設定
  - INPUT_IMAGE_PATH: 入力画像パス
  - ASPECT_RATIO: アスペクト比 (1:1, 16:9等)
  - RESOLUTION: 解像度 (1K, 2K, 4K)

■ 出力ファイル (output/ディレクトリ)
  - *_expanded.png: キャンバス拡張後の画像
  - *_matte.png: 生成されたアルファマット
  - *_rgba.png: 最終的なRGBA合成画像
  - *.json: メタデータ

■ 実行例
  python samples/gemini_api_sample/nanobanana_edit_alpha_sample.py
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
INPUT_IMAGE_PATH = "samples/sample_images/gemini-768x1344-woman.jpg"
# Geminiモデル
MODEL_ID = "gemini-3-pro-image-preview"

# アスペクト比
#   選択可能: "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
ASPECT_RATIO = "9:16"

# 解像度 (Gemini 3のみ有効)
#   選択可能: "1K", "2K", "4K"
RESOLUTION = "4K"

# キャンバス拡張画像を保存するか
SAVE_EXPANDED_IMAGE = True

# キャンバス拡張時の背景色 (RGB)
CANVAS_BG_COLOR = (255, 255, 255)  # 白

# ============================================================


# アルファマット生成プロンプト
ALPHA_MATTE_PROMPT = (
    "A professional high-precision grayscale alpha matte of the character from the input image. "
    "The background must be pure black (Hex #000000). "
    "The character's body, hair, and worn accessories must be rendered in white. "
    "Crucially, represent semi-transparency (e.g., fine hair strands, lace fabric, transparent plastic) "
    "using accurate shades of gray to indicate opacity levels. "
    "Strictly exclude and mask out any cast shadows, floor reflections, and detached objects placed on the floor "
    "(such as bags or hats not being worn). "
    "The output should be a clean, noise-free segmentation mask suitable for professional compositing."
)


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


def resize_and_expand_canvas(
    img: Image.Image,
    target_size: tuple[int, int],
    bg_color: tuple[int, int, int]
) -> tuple[Image.Image, bool, tuple[int, int] | None]:
    """
    画像をリサイズ・キャンバス拡張して目標サイズにする（中央配置）。

    処理フロー:
    1. 入力画像が目標サイズより大きい場合、アスペクト比を維持して縮小
    2. 縮小後または元のサイズが目標より小さい場合、キャンバス拡張

    Args:
        img: 入力画像
        target_size: 目標サイズ (width, height)
        bg_color: キャンバス背景色 (R, G, B)

    Returns:
        (処理済み画像, 縮小実行フラグ, 縮小後のサイズ or None)
    """
    target_width, target_height = target_size
    img_width, img_height = img.size
    was_resized = False
    resized_size = None

    # 1. 縮小が必要かチェック
    if img_width > target_width or img_height > target_height:
        # アスペクト比を維持して縮小（両辺が目標サイズ以下になるように）
        scale = min(target_width / img_width, target_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        was_resized = True
        resized_size = (new_width, new_height)
        img_width, img_height = img.size

    # 2. 既に目標サイズと一致する場合はそのまま返す
    if img_width == target_width and img_height == target_height:
        return img, was_resized, resized_size

    # 3. キャンバス拡張が必要な場合
    if img_width < target_width or img_height < target_height:
        canvas = Image.new("RGB", target_size, bg_color)

        # 中央に配置
        x_offset = (target_width - img_width) // 2
        y_offset = (target_height - img_height) // 2

        canvas.paste(img, (x_offset, y_offset))
        return canvas, was_resized, resized_size

    return img, was_resized, resized_size


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
    print("\n" + "="*60)
    print(title)
    print("="*60)


def print_step(step_num: int, total_steps: int, message: str) -> None:
    """ステップ番号付きメッセージを表示する"""
    print(f"\n[{step_num}/{total_steps}] {message}")


def generate_output_filenames(timestamp: str) -> dict[str, Path]:
    """出力ファイル名を生成する"""
    base_name = f"nanobanana_edit_alpha_sample_{timestamp}"
    return {
        "expanded": OUTPUT_DIR / f"{base_name}_expanded.png",
        "matte": OUTPUT_DIR / f"{base_name}_matte.png",
        "rgba": OUTPUT_DIR / f"{base_name}_rgba.png",
        "json": OUTPUT_DIR / f"{base_name}.json",
    }


def request_alpha_matte(
    image_bytes: bytes,
    prompt: str,
    model_id: str,
    aspect_ratio: str,
    resolution: str,
    api_key: str,
) -> tuple[bytes | None, Any]:
    """Gemini APIにアルファマット生成をリクエストする"""
    client = genai.Client(api_key=api_key, vertexai=False)

    # Gemini 3の場合は画像設定を追加
    config = None
    if model_id.startswith("gemini-3-"):
        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution,
            ),
        )

    # APIリクエスト
    response = client.models.generate_content(
        model=model_id,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
        config=config,
    )

    # レスポンスから画像データを抽出
    parts = getattr(response, "parts", None)
    if not parts:
        return None, response

    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            data_field = inline_data.data
            if isinstance(data_field, str):
                return base64.b64decode(data_field), response
            return data_field, response

    return None, response


def save_metadata(
    filepath: Path,
    timestamp: str,
    input_path: str,
    orig_size: tuple[int, int],
    target_size: tuple[int, int],
    expanded_size: tuple[int, int],
    output_files: dict[str, Path],
    rgba_size: int,
    resized_size: tuple[int, int] | None = None,
    api_response: Any = None,
) -> None:
    """メタデータをJSON形式で保存する"""
    orig_w, orig_h = orig_size
    target_w, target_h = target_size
    expanded_w, expanded_h = expanded_size

    # レスポンスメタデータの抽出
    response_metadata = {}
    if api_response:
        # Usage Metadata
        usage = getattr(api_response, "usage_metadata", None)
        if usage:
            response_metadata["usage_metadata"] = {
                "prompt_token_count": getattr(usage, "prompt_token_count", None),
                "candidates_token_count": getattr(usage, "candidates_token_count", None),
                "total_token_count": getattr(usage, "total_token_count", None),
            }
        
        # Safety & Finish Reason (from first candidate)
        candidates = getattr(api_response, "candidates", None)
        if candidates and len(candidates) > 0:
            cand = candidates[0]
            response_metadata["finish_reason"] = str(getattr(cand, "finish_reason", None))
            
            safety = getattr(cand, "safety_ratings", None)
            if safety:
                response_metadata["safety_ratings"] = []
                for rating in safety:
                    response_metadata["safety_ratings"].append({
                        "category": str(getattr(rating, "category", None)),
                        "probability": str(getattr(rating, "probability", None)),
                        "blocked": getattr(rating, "blocked", None)
                    })

    # 縮小情報を追加
    resized_info = None
    if resized_size:
        resize_w, resize_h = resized_size
        resized_info = {"width": resize_w, "height": resize_h}

    metadata = {
        "timestamp": timestamp,
        "input_image": str(input_path),
        "original_resolution": {"width": orig_w, "height": orig_h},
        "target_resolution": {"width": target_w, "height": target_h},
        "resized_resolution": resized_info,
        "expanded_resolution": {"width": expanded_w, "height": expanded_h},
        "prompt": ALPHA_MATTE_PROMPT,
        "generation": {
            "model": MODEL_ID,
            "aspect_ratio": ASPECT_RATIO,
            "resolution": RESOLUTION,
            "was_resized": resized_size is not None,
            "expanded_image_path": str(output_files["expanded"]) if SAVE_EXPANDED_IMAGE else None,
            "alpha_matte_path": str(output_files["matte"]),
            "rgba_output_path": str(output_files["rgba"]),
            "rgba_size_bytes": rgba_size,
        },
        "api_response_metadata": response_metadata,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)



def main() -> None:
    """メイン処理"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_files = generate_output_filenames(timestamp)

    # ヘッダー表示
    print_header("Nanobanana アルファマット生成サンプル")
    print(f"入力画像: {INPUT_IMAGE_PATH}")
    print(f"モデル: {MODEL_ID}")
    print(f"アスペクト比: {ASPECT_RATIO}")
    print(f"解像度: {RESOLUTION}")

    # Step 1: 入力画像を読み込み
    print_step(1, 5, "入力画像を読み込み中...")
    input_img = load_input_image(INPUT_IMAGE_PATH)
    orig_w, orig_h = input_img.size
    print(f"元の解像度: {orig_w} x {orig_h} pixels")

    # Step 2: 目標解像度を計算してリサイズ・キャンバス拡張
    print_step(2, 5, "リサイズ・キャンバス拡張処理中...")
    target_w, target_h = calculate_target_resolution(ASPECT_RATIO, RESOLUTION)
    print(f"目標解像度: {target_w} x {target_h} pixels")

    expanded_img, was_resized, resized_size = resize_and_expand_canvas(
        input_img, (target_w, target_h), CANVAS_BG_COLOR
    )
    expanded_w, expanded_h = expanded_img.size

    # 縮小が実行された場合は警告を表示
    if was_resized and resized_size:
        resize_w, resize_h = resized_size
        print("\n⚠️ 警告: 入力画像が目標サイズより大きいため、縮小しました")
        print(f"  元のサイズ: {orig_w} x {orig_h} pixels")
        print(f"  縮小後のサイズ: {resize_w} x {resize_h} pixels")

    print(f"最終解像度: {expanded_w} x {expanded_h} pixels")

    if SAVE_EXPANDED_IMAGE:
        expanded_img.save(output_files["expanded"])
        print(f"処理済み画像を保存: {output_files['expanded']}")

    # Step 3: Gemini APIでアルファマット生成
    print_step(3, 5, "Gemini APIでアルファマット生成中...")

    img_buffer = BytesIO()
    expanded_img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    alpha_matte_data, api_response = request_alpha_matte(
        image_bytes=img_bytes,
        prompt=ALPHA_MATTE_PROMPT,
        model_id=MODEL_ID,
        aspect_ratio=ASPECT_RATIO,
        resolution=RESOLUTION,
        api_key=API_KEY,
    )

    if alpha_matte_data is None:
        print("エラー: アルファマット画像を取得できませんでした")
        return

    # Step 4: アルファマット画像を保存
    print_step(4, 5, "アルファマット画像を保存中...")
    with open(output_files["matte"], "wb") as f:
        f.write(alpha_matte_data)
    print(f"保存完了: {output_files['matte']}")

    # Step 5: RGBA画像合成
    print_step(5, 5, "RGBA画像合成中...")
    rgba_bytes = compose_alpha_channel(expanded_img, alpha_matte_data)

    with open(output_files["rgba"], "wb") as f:
        f.write(rgba_bytes)
    print(f"RGBA合成画像を保存: {output_files['rgba']}")
    print(f"ファイルサイズ: {len(rgba_bytes)} bytes")

    # メタデータを保存
    save_metadata(
        filepath=output_files["json"],
        timestamp=timestamp,
        input_path=INPUT_IMAGE_PATH,
        orig_size=(orig_w, orig_h),
        target_size=(target_w, target_h),
        expanded_size=(expanded_w, expanded_h),
        output_files=output_files,
        rgba_size=len(rgba_bytes),
        resized_size=resized_size,
        api_response=api_response,
    )
    print(f"\nメタデータを保存: {output_files['json']}")

    print_header("処理完了")


if __name__ == "__main__":
    main()
