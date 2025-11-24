"""
Imagen 4 シンプルサンプル

Google Cloud Vertex AI の Imagen モデルを使用して画像を生成するサンプルです。

機能:
- テキストプロンプトから画像を生成
- 複数の Imagen モデルバージョンに対応 (3.0, 4.0 など)
- 環境変数 (.env ファイル) で設定可能
- タイムスタンプ付きの自動出力ファイル管理
- サポートされているモデルの検証機能

設定項目:
- GCP_PROJECT_ID: Google Cloud プロジェクト ID
- GCP_LOCATION: GCP リージョン (デフォルト: us-central1)
- IMAGEN_MODEL: 使用するモデル (AVAILABLE_MODELS リストを参照)
- IMAGEN_PROMPT: 画像生成用のテキストプロンプト

出力:
- 生成された画像は 'output/' ディレクトリに保存されます
- ファイル名形式: imagen4_simple_sample_YYYYMMDDHHMM.png
"""

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# 環境変数を読み込み
load_dotenv()

# 利用可能な Imagen モデル
AVAILABLE_MODELS = [
    "imagen-4.0-generate-001",
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-3.0-generate-002",
    "imagen-3.0-generate-001",
    "imagen-3.0-fast-generate-001",
    "imagen-3.0-capability-001",
    "imagegeneration@006",
    "imagegeneration@005",
    "imagegeneration@002",
]

# .env から設定を読み込み
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
IMAGEN_MODEL = os.getenv("IMAGEN_MODEL", "imagen-3.0-generate-002")
IMAGEN_PROMPT = os.getenv("IMAGEN_PROMPT", "A beautiful sunset over mountains")

# モデル選択の検証
if IMAGEN_MODEL not in AVAILABLE_MODELS:
    print(f"警告: {IMAGEN_MODEL} は既知のモデルリストにありません。")
    print(f"利用可能なモデル: {', '.join(AVAILABLE_MODELS)}")

# 出力設定
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d%H%M")
output_file = OUTPUT_DIR / f"imagen4_simple_sample_{timestamp}.png"

vertexai.init(project=PROJECT_ID, location=LOCATION)

model = ImageGenerationModel.from_pretrained(IMAGEN_MODEL)

images = model.generate_images(
    prompt=IMAGEN_PROMPT,
    # オプションパラメータ
    number_of_images=1,
    language="en",
    # シード値とウォーターマークは同時に使用できません
    # add_watermark=False,
    # seed=100,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

images[0].save(location=output_file, include_generation_parameters=False)

# オプション: ノートブックで生成された画像を表示
# images[0].show()

print(f"{len(images[0]._image_bytes)} バイトで画像を生成しました")
# 出力例:
# 1234567 バイトで画像を生成しました