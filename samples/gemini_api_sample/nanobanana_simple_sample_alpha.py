"""
Nanobanana Text-to-Image サンプル

Google Generative AI Python クライアント (`google.genai`) を使用して
Gemini 画像モデルからテキストプロンプトで画像を生成するサンプルコードです。

機能:
- APIキー認証で Gemini 画像モデルを使用
- テキストプロンプトから画像を生成
- 画像フォーマット (PNG / JPEG) をコード内で切り替え
- アスペクト比 / 解像度 (Gemini 3 系のみ) をコード内で選択
- タイムスタンプ付きの自動出力ファイル管理
- Split Alpha 機能（左右分割マスクから RGBA PNG を生成して別ファイルとして保存）

設定項目:
- GOOGLE_API_KEY: Gemini API の API キー

出力:
- 生成された画像は 'output/' ディレクトリに保存されます
- ファイル名形式: nanobanana_simple_sample_YYYYMMDDHHMM.(png|jpg)

参考サイト
- Gemini Developer API ドキュメント
  https://developers.generativeai.google/products/gemini-developer-api/docs/get-started/overview
  https://ai.google.dev/gemini-api/docs/image-generation?hl=ja
  
"""

import os
import json
from datetime import datetime
from pathlib import Path
from io import BytesIO
import base64
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# 環境変数を読み込み
load_dotenv()

# .env から設定を読み込み
API_KEY = os.getenv("GOOGLE_API_KEY")

# API Keyの先頭の10文字をプリント
print(f"Using API Key: {API_KEY[:10]}...")

#MODEL_ID = "gemini-2.5-flash-image"
MODEL_ID = "gemini-3-pro-image-preview"
#GEMINI_IMAGE_PROMPT = "A cute cat on the sofa. For portable use."
# GEMINI_IMAGE_PROMPT = "A cute cat wearing a cape, flying in the sky. Character-style, gentle expression. Generate as a PNG with a fully transparent background and an alpha channel. No background elements at all. Clean, crisp outlines for the cat and cape."


GEMINI_IMAGE_PROMPT = (
    # --- 1. 被写体の描写にスカーフと風を追加 ---
    "A full-body shot of a young female model with long hair, **wearing a stylish scarf that is naturally blowing and flowing in a gentle breeze.** "
    "She is posing for high-end fashion photography in a professional studio setting against a seamless, completely shadowless pure white infinity background. "
    "The composition must be clean, strictly devoid of any text, logos, magazine overlays, or typography. "
    "The image captures her entire figure from head to toe. \n"
    
    "formatted as a side-by-side split-screen layout. The left half displays the subject in full color. "
    "The right half displays the generated grayscale alpha matte against a pure black background. "
    
    # --- 2. マスク対象の定義を更新 ---
    "The matte must represent the opacity of the entire subject. "
    # スカーフも含むように明記
    "Fully opaque areas (e.g., body, thick clothing, **the main part of the scarf**, shoes) must be represented as solid white. "
    # 半透明の例に、風になびくスカーフの端などを追加
    "Semi-transparent areas (e.g., sheer fabric, **blowing scarf edges**, lace, fine hair tips) must be represented by corresponding shades of gray, where lighter gray indicates higher opacity. "
    
    "Crucially, the matte must strictly exclude any cast shadows on the white floor or background. "
    
    # 細部の保持にスカーフの動きを追加
    "Within this matte, the edges and fine details like individual hair strands **and the flowing fabric of the scarf** must be strictly preserved without simplification. "
    
    "CRITICAL: The grayscale mask on the right must be spatially aligned perfectly with the color subject on the left. "
    "It must occupy the exact same relative position, scale, and pose within its half as the subject does in the left half. "
    "There should be zero horizontal or vertical offset between the subject and its generated matte."
)

# 出力画像形式（PNG/JPEG）をコード内で切り替え可能にする
#   高画質: "image/png"
#   軽量:   "image/jpeg"  ※デフォルト
#   ※ 現行の genai API では画像モデルに対して
#      MIME タイプを直接指定することはできないため、
#      この値は「保存ファイルの拡張子」と
#      「解析時に渡す MIME」のみに使用する。
#   ← 必要に応じてこの 1 行だけを書き換えて検証する
RESPONSE_MIME_TYPE = "image/jpeg"

# アスペクト比をコード内で選択
#   利用可能な値:
#     "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
ASPECT_RATIO = "9:16"

# 解像度をコード内で選択（Gemini 3 系のみ有効）
#   利用可能な値: "1K", "2K", "4K"
#   ※ Gemini 2.5 系モデルに渡すとエラーになるため、
#      実際に API に送るのは gemini-3-* モデルのときだけにする
RESOLUTION = "4K"

# Split Alpha機能
#   True: 画像の左半分をRGB、右半分をアルファチャンネルとして合成したPNGを生成
#   False: 通常の画像保存のみ
SPLIT_ALPHA = True

if RESPONSE_MIME_TYPE == "image/jpeg":
    OUTPUT_EXT = ".jpg"
else:
    OUTPUT_EXT = ".png"

#IMAGEN4_MODEL_ID= "imagen-4.0-generate-001  "
#IMAGEN4_PROMPT= "'Robot holding a red skateboard."

# MODEL_ID= IMAGEN4_MODEL_ID
# GEMINI_IMAGE_PROMPT= IMAGEN4_PROMPT



# 出力設定
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d%H%M")
output_file = OUTPUT_DIR / f"nanobanana_simple_sample_{timestamp}{OUTPUT_EXT}"

# Gemini Developer API クライアントを作成（vertexai=Falseを明示）
client = genai.Client(api_key=API_KEY, vertexai=False)

print(f"モデル: {MODEL_ID}")
print(f"プロンプト: {GEMINI_IMAGE_PROMPT}")
print(f"出力 MIME タイプ: {RESPONSE_MIME_TYPE}")
print(f"アスペクト比: {ASPECT_RATIO}")
print(f"解像度 (Gemini 3 のみ有効): {RESOLUTION}")
print(f"画像を生成中...")

# 画像生成（公式サンプルに近い形: contents=[prompt]）
# モデルごとに渡すパラメータを変える:
#   - gemini-3-* : aspect_ratio + resolution (image_size)
#   - gemini-2.5-* : aspect_ratio のみ
gen_config = None
if MODEL_ID.startswith("gemini-3-"):
    gen_config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=ASPECT_RATIO,
            image_size=RESOLUTION,
        ),
    )
elif MODEL_ID.startswith("gemini-2.5-"):
    gen_config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=ASPECT_RATIO,
        ),
    )

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[GEMINI_IMAGE_PROMPT],
    config=gen_config,
)

# レスポンス構造の簡易検証
print(f"response type: {type(response)}")
parts = getattr(response, "parts", None)
candidates = getattr(response, "candidates", None)
print(f"has parts: {bool(parts)}")
print(f"has candidates: {bool(candidates)}")
if candidates:
    print(f"candidates len: {len(candidates)}")


def split_alpha_channel(image_bytes: bytes) -> bytes:
    """
    画像の左半分をRGB、右半分をアルファチャンネルとして合成した
    RGBA PNG画像を生成する。

    Args:
        image_bytes: 入力画像のバイナリデータ

    Returns:
        RGBA PNG形式のバイナリデータ
    """
    # バイナリから画像を読み込み
    img = Image.open(BytesIO(image_bytes))

    # RGB画像に変換（アルファチャンネルがあれば削除）
    if img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    half_width = width // 2

    # 左半分をRGB画像として抽出
    left_half = img.crop((0, 0, half_width, height))

    # 右半分を抽出してグレースケールに変換（アルファチャンネル用）
    right_half = img.crop((half_width, 0, width, height))
    right_gray = right_half.convert("L")  # グレースケール変換

    # RGBA画像を作成（左半分のRGB + 右半分のアルファ）
    rgba_img = left_half.convert("RGBA")
    rgba_img.putalpha(right_gray)

    # PNG形式でバイナリ出力
    output_buffer = BytesIO()
    rgba_img.save(output_buffer, format="PNG")
    return output_buffer.getvalue()


# 画像データを取得して保存
image_saved = False
saved_image_size = None
alpha_output_file = None
alpha_image_size = None

# 1. 公式サンプルと同様に response.parts を優先的に見る
if parts:
    for part in parts:
        if getattr(part, "text", None):
            print("TEXT (from parts):", part.text[:200])
            continue

        inline_data = getattr(part, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            # レスポンスからMIMEタイプを取得（あれば）
            mime_from_response = getattr(inline_data, "mime_type", None)
            print(f"レスポンスのMIMEタイプ: {mime_from_response}")

            data_field = inline_data.data
            if isinstance(data_field, str):
                image_data = base64.b64decode(data_field)
            else:
                image_data = data_field

            # 画像データの先頭バイトでフォーマットを判定
            if image_data[:8] == b'\x89PNG\r\n\x1a\n':
                actual_format = 'PNG'
                print("実際の画像フォーマット: PNG")
            elif image_data[:2] == b'\xff\xd8':
                actual_format = 'JPEG'
                print("実際の画像フォーマット: JPEG")
            else:
                actual_format = 'UNKNOWN'
                print(f"実際の画像フォーマット: 不明 (先頭バイト: {image_data[:8].hex()})")

            with open(output_file, "wb") as f:
                f.write(image_data)

            print(f"\n画像を生成しました: {output_file}")
            print(f"ファイルサイズ: {len(image_data)} バイト")
            image_saved = True
            saved_image_size = len(image_data)

            # Split Alpha機能が有効な場合、アルファチャンネル合成処理を実行
            if SPLIT_ALPHA:
                try:
                    print("\nSplit Alpha処理を実行中...")
                    alpha_bytes = split_alpha_channel(image_data)
                    alpha_output_file = OUTPUT_DIR / f"nanobanana_simple_sample_{timestamp}_alpha.png"

                    with open(alpha_output_file, "wb") as f:
                        f.write(alpha_bytes)

                    alpha_image_size = len(alpha_bytes)
                    print(
                        f"アルファチャンネル合成画像を保存しました: {alpha_output_file} "
                        f"({len(alpha_bytes)} バイト)"
                    )
                except Exception as e:
                    print(f"Split Alpha処理中にエラーが発生しました: {e}")

            break

# 2. parts に何もない場合、candidates 経由で取得する（SDK 実装差異のフォールバック）
if not image_saved and candidates:
    print("fallback: scanning candidates for image parts...")
    for idx, candidate in enumerate(candidates):
        content = getattr(candidate, "content", None)
        c_parts = getattr(content, "parts", None)
        if not c_parts:
            continue

        print(f" candidate[{idx}] parts: {len(c_parts)}")
        for part in c_parts:
            if getattr(part, "text", None):
                print("TEXT (from candidates):", part.text[:200])
                continue

            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                # レスポンスからMIMEタイプを取得（あれば）
                mime_from_response = getattr(inline_data, "mime_type", None)
                print(f"レスポンスのMIMEタイプ: {mime_from_response}")

                data_field = inline_data.data
                if isinstance(data_field, str):
                    image_data = base64.b64decode(data_field)
                else:
                    image_data = data_field

                # 画像データの先頭バイトでフォーマットを判定
                if image_data[:8] == b'\x89PNG\r\n\x1a\n':
                    actual_format = 'PNG'
                    print("実際の画像フォーマット: PNG")
                elif image_data[:2] == b'\xff\xd8':
                    actual_format = 'JPEG'
                    print("実際の画像フォーマット: JPEG")
                else:
                    actual_format = 'UNKNOWN'
                    print(f"実際の画像フォーマット: 不明 (先頭バイト: {image_data[:8].hex()})")

                with open(output_file, "wb") as f:
                    f.write(image_data)

                print(f"\n画像を生成しました: {output_file}")
                print(f"ファイルサイズ: {len(image_data)} バイト")
                image_saved = True
                saved_image_size = len(image_data)

                # Split Alpha機能が有効な場合、アルファチャンネル合成処理を実行
                if SPLIT_ALPHA:
                    try:
                        print("\nSplit Alpha処理を実行中...")
                        alpha_bytes = split_alpha_channel(image_data)
                        alpha_output_file = OUTPUT_DIR / f"nanobanana_simple_sample_{timestamp}_alpha.png"

                        with open(alpha_output_file, "wb") as f:
                            f.write(alpha_bytes)

                        alpha_image_size = len(alpha_bytes)
                        print(
                            f"アルファチャンネル合成画像を保存しました: {alpha_output_file} "
                            f"({len(alpha_bytes)} バイト)"
                        )
                    except Exception as e:
                        print(f"Split Alpha処理中にエラーが発生しました: {e}")

                break

        if image_saved:
            break

if not image_saved:
    print("\nエラー: 画像データを取得できませんでした")


# 生成された画像があれば、その画像を別モデルで解析してみる
analysis_text: str | None = None
if image_saved:
    try:
        with open(output_file, "rb") as f:
            image_bytes = f.read()

        print("\n生成された画像を gemini-2.5-flash で解析中...")
        print(f"解析入力 MIME タイプ: {RESPONSE_MIME_TYPE}")

        analysis = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Describe this image in detail.",
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=RESPONSE_MIME_TYPE,
                ),
            ],
        )

        print("\n解析結果:")
        analysis_text = analysis.text
        print(analysis_text)
    except Exception as e:
        print(f"\n画像解析中にエラーが発生しました: {e}")


# パラメータと結果情報を JSON に保存
json_output_file = OUTPUT_DIR / f"nanobanana_simple_sample_{timestamp}.json"

# --- 詳細メタデータの抽出 ---
# リクエストパラメータの再構築
request_parameters = {
    "model": MODEL_ID,
    "prompt": GEMINI_IMAGE_PROMPT,
    "config": {
        "response_modalities": ["TEXT", "IMAGE"],
        "image_config": {
            "aspect_ratio": ASPECT_RATIO,
        }
    }
}
if MODEL_ID.startswith("gemini-3-"):
    request_parameters["config"]["image_config"]["image_size"] = RESOLUTION

# レスポンスメタデータの抽出
response_metadata = {}
if 'response' in locals() and response:
    # Usage Metadata
    usage = getattr(response, "usage_metadata", None)
    if usage:
        response_metadata["usage_metadata"] = {
            "prompt_token_count": getattr(usage, "prompt_token_count", None),
            "candidates_token_count": getattr(usage, "candidates_token_count", None),
            "total_token_count": getattr(usage, "total_token_count", None),
        }

    # Safety & Finish Reason (from first candidate)
    if candidates and len(candidates) > 0:
        cand = candidates[0]
        response_metadata["finish_reason"] = str(getattr(cand, "finish_reason", None))

        safety = getattr(cand, "safety_ratings", None)
        if safety:
            # 安全性評価をリスト化
            response_metadata["safety_ratings"] = []
            for rating in safety:
                response_metadata["safety_ratings"].append({
                    "category": str(getattr(rating, "category", None)),
                    "probability": str(getattr(rating, "probability", None)),
                    "blocked": getattr(rating, "blocked", None)
                })

result_payload = {
    "timestamp": timestamp,
    "request_parameters": request_parameters,
    "response_metadata": response_metadata,
    "prompt": GEMINI_IMAGE_PROMPT,
    "generation": {
        "model": MODEL_ID,
        "response_mime_type": RESPONSE_MIME_TYPE,
        "aspect_ratio": ASPECT_RATIO,
        "resolution": RESOLUTION,
        "split_alpha_enabled": SPLIT_ALPHA,
        "output_image_path": str(output_file) if image_saved else None,
        "image_saved": image_saved,
        "image_size_bytes": saved_image_size,
        "alpha_image_path": str(alpha_output_file) if alpha_output_file else None,
        "alpha_image_saved": alpha_output_file is not None,
        "alpha_image_size_bytes": alpha_image_size,
    },
    "analysis": {
        "model": "gemini-2.5-flash",
        "input_mime_type": RESPONSE_MIME_TYPE if image_saved else None,
        "text": analysis_text,
    },
}

with open(json_output_file, "w", encoding="utf-8") as f:
    json.dump(result_payload, f, ensure_ascii=False, indent=2)

print(f"\nパラメータと結果情報を JSON に保存しました: {json_output_file}")
