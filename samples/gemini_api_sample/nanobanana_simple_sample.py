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
from datetime import datetime
from pathlib import Path
import base64
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 環境変数を読み込み
load_dotenv()

# .env から設定を読み込み
API_KEY = os.getenv("GOOGLE_API_KEY")

# API Keyの先頭の10文字をプリント
print(f"Using API Key: {API_KEY[:10]}...")

#MODEL_ID = "gemini-2.5-flash-image"
MODEL_ID = "gemini-3-pro-image-preview"

GEMINI_IMAGE_PROMPT = (
    "A full-body shot of a young female model with long hair, "
    "wearing a stylish scarf that is naturally blowing and flowing in a gentle breeze. "
    "She is posing for high-end fashion photography in a professional studio setting "
    "against a seamless, completely shadowless pure white infinity background. "
    "The composition must be clean, strictly devoid of any text, logos, magazine overlays, or typography. "
    "The image captures her entire figure from head to toe."
)

# 出力画像形式（PNG/JPEG）をコード内で切り替え可能にする
#   高画質: "image/png"
#   軽量:   "image/jpeg"  ※デフォルト
#   ※ 現行の genai API では画像モデルに対して
#      MIME タイプを直接指定することはできないため、
#      この値は「保存ファイルの拡張子」と
#      ログ出力時の表記のみに使用する。
RESPONSE_MIME_TYPE = "image/jpeg"

# アスペクト比をコード内で選択
#   利用可能な値:
#     "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
ASPECT_RATIO = "16:9"

# 解像度をコード内で選択（Gemini 3 系のみ有効）
#   利用可能な値: "1K", "2K", "4K"
RESOLUTION = "1K"

if RESPONSE_MIME_TYPE == "image/jpeg":
    OUTPUT_EXT = ".jpg"
else:
    OUTPUT_EXT = ".png"

# 出力設定
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d%H%M")
output_file = OUTPUT_DIR / f"nanobanana_simple_sample_{timestamp}{OUTPUT_EXT}"

# Gemini Developer API クライアントを作成（vertexai=Falseを明示）
client = genai.Client(api_key=API_KEY, vertexai=False)

print(f"モデル: {MODEL_ID}")
print(f"プロンプト: {GEMINI_IMAGE_PROMPT}")
print(f"出力 MIME タイプ(想定): {RESPONSE_MIME_TYPE}")
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

print(f"response type: {type(response)}")


def extract_image_bytes(resp) -> bytes:
    """Gemini のレスポンスから最初の画像バイト列を取り出す。"""
    parts = getattr(resp, "parts", None)
    candidates = getattr(resp, "candidates", None)

    # 1. response.parts を優先
    if parts:
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                data_field = inline_data.data
                if isinstance(data_field, bytes):
                    return data_field
                if isinstance(data_field, str):
                    return base64.b64decode(data_field)

    # 2. candidates 経由のフォールバック
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            c_parts = getattr(content, "parts", None)
            if not c_parts:
                continue
            for part in c_parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    data_field = inline_data.data
                    if isinstance(data_field, bytes):
                        return data_field
                    if isinstance(data_field, str):
                        return base64.b64decode(data_field)

    raise RuntimeError("画像データをレスポンスから取得できませんでした")


# 画像データを取得して保存
try:
    image_data = extract_image_bytes(response)
except Exception as e:
    print(f"\nエラー: 画像データの取得に失敗しました: {e}")
else:
    # フォーマットの簡易判定（ログ用）
    if image_data[:8] == b'\x89PNG\r\n\x1a\n':
        actual_format = "PNG"
    elif image_data[:2] == b'\xff\xd8':
        actual_format = "JPEG"
    else:
        actual_format = "UNKNOWN"

    print(f"実際の画像フォーマット(先頭バイト判定): {actual_format}")

    with open(output_file, "wb") as f:
        f.write(image_data)

    print(f"\n画像を生成しました: {output_file}")
    print(f"ファイルサイズ: {len(image_data)} バイト")
