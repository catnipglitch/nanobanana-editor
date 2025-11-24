"""
Imagen 4.0 Text-to-Image サンプル

Google Gemini API の Imagen 4.0 モデルを使用して画像を生成するサンプルコードです。

機能:
- APIキー認証で Imagen 4.0 モデルを使用
- テキストプロンプトから画像を生成
- 環境変数 (.env ファイル) で設定可能
- タイムスタンプ付きの自動出力ファイル管理

設定項目:
- GOOGLE_API_KEY: Google API キー

出力:
- 生成された画像は 'output/' ディレクトリに保存されます
- ファイル名形式: imagen4_simple_sample_YYYYMMDDHHMM_N.png
"""

from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# 環境変数を読み込み
load_dotenv()

# .env から設定を読み込み
API_KEY = os.getenv("GOOGLE_API_KEY")

# Imagen 4.0 モデル設定
MODEL_ID = "imagen-4.0-generate-001"
PROMPT = "Robot holding a red skateboard"
NUMBER_OF_IMAGES = 1  # 生成する画像枚数
 
# 出力設定
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d%H%M")

print(f"Using API Key: {API_KEY[:10] if API_KEY else 'None'}...")
print(f"モデル: {MODEL_ID}")
print(f"プロンプト: {PROMPT}")
print(f"画像枚数: {NUMBER_OF_IMAGES}")
print("\n画像を生成中...")

# Gemini Developer API クライアントを作成（vertexai=Falseを明示）
client = genai.Client(api_key=API_KEY, vertexai=False)

# 画像生成
response = client.models.generate_images(
    model=MODEL_ID,
    prompt=PROMPT,
    config=types.GenerateImagesConfig(
        number_of_images=NUMBER_OF_IMAGES,
    )
)

# 画像を保存
for i, generated_image in enumerate(response.generated_images, 1):
    output_file = OUTPUT_DIR / f"imagen4_simple_sample_{timestamp}_{i}.png"

    # PIL Image を保存
    generated_image.image.save(output_file)

    # ファイルサイズを取得
    file_size = output_file.stat().st_size

    print(f"\n画像を生成しました: {output_file}")
    print(f"ファイルサイズ: {file_size:,} バイト")

print(f"\n✓ 合計 {len(response.generated_images)} 枚の画像を生成しました")
