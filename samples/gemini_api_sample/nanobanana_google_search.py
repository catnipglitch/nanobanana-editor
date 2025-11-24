"""
Nanobanana Google Search grounding サンプル

Google Generative AI Python クライアント (`google.genai`) を使用して
Google 検索ツールでグラウンディングした Gemini 画像生成を行うサンプルコードです。

機能:
- `.env` から GOOGLE_API_KEY を読み込んで API キー認証
- Google 検索ツールを有効にした上で Gemini 3 Pro Image Preview で画像生成
- テキスト応答（解説）と画像を取得
- `output/` ディレクトリにタイムスタンプ付きファイル名で画像を書き出し

設定項目:
- GOOGLE_API_KEY: Gemini API の API キー

出力:
- 生成された画像は 'output/' ディレクトリに保存されます
- ファイル名形式: nanobanana_google_search_YYYYMMDDHHMM.(png|jpg)
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

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY が設定されていません。.env を確認してください。")

print(f"Using API Key: {API_KEY[:10]}...")

MODEL_ID = "gemini-3-pro-image-preview"

PROMPT_SF = (
    "Visualize the current weather forecast for the next 5 days in San Francisco "
    "as a clean, modern weather chart. Add a visual on what I should wear each day."
)

PROMPT_MIHOYO = (
    "(Subject) cute anime girl, vtuber, weather reporter, pointing at holographic screen, "
    "(Style) art style of Hoyoverse, Genshin Impact style, Honkai Star Rail style, cel shading, vibrant colors, clean lines,"
    " (Details) intricate details, beautiful face, detailed eyes, shiny hair, (Outfit) stylish modern fashion, trendy outfit, "
    "(Background) Tokyo Shibuya street background, depth of field, bokeh, floating weather icons, sunny and rainy icons, "
    "(Quality) masterpiece, best quality, 8k, ray tracing, cinematic lighting, dynamic angle, highly detailed --ar 3:4, "
    "bold, sans-serif font for text elements"
)

PROMPT_WEATHER_REPORT = """
# 前提条件
地域・場所: 東京 渋谷スクランブル交差点

# 指示
上記「前提条件」で指定された場所にて、テレビの天気予報番組が生中継を行っている画像を生成してください。
Google検索を使用し、実在の最新情報を反映させること。

## STEP 1: Google Search (Data Acquisition)
指定された場所について以下の検索を行い、情報を取得してください。
1. 指定された場所の向こう3日間の天気予報 (日付、曜日、天気、最高/最低気温)
2. 指定された場所の現在の天気と適した服装
3. 現在時刻 (現地時間)

## STEP 2: Variable Definition
検索結果に基づき、以下の要素を確定させてください。
- {CURRENT_TIME}: 画像生成時点の時刻 (例: 14:35)。タイムゾーンは指定された場所に合わせること
- {CURRENT_WEATHER}: 現在の現地の天候、空の明るさ、ライティング
- {OUTFIT_STYLE}: 気温と天気に適した、清潔感のあるキャスター風のモダンな服装
- {FORECAST_DATA}: 明日から3日間の天気情報

## STEP 3: Image Generation Description
以下の要件で画像を描画してください。

### 1. テーマと構図
- テーマ: 天気予報番組の生中継 (Live Broadcast)
- 構図: 現地からレポートするお天気キャスター。バストアップからウェストアップからのアングルで、そこにいるかのように視聴者に伝わるバランス
- UIオーバーレイ (画面表示):
  1. 左上: 時刻 {CURRENT_TIME} と番組ロゴ "NHTV" (白文字、ドロップシャドウ、サンセリフ体)
  2. 右上: 赤い "LIVE" アイコン と 「指定された場所の地名」 (白文字)

### 2. メインキャラクター
- キャラクター: アニメ調のかわいいお天気お姉さん（Live2D/Vtuberスタイル）
- 演技: カメラ（視聴者）に向かってプロらしく、天気予報説明パネルを丁寧に両手で自然に持ち、親しみやすく視聴者に向けて語りかけている。
- 服装: {OUTFIT_STYLE} (季節感とトレンドを意識したスタイリング)

### 3. アイテム：天気予報パネル
- アイテム: 情報は整理された日本語の天気予報説明パネル
- 表示内容:
  1. 注記: 明日、明後日などの文字は入れず、日付と曜日のみを表示する
  2. レイアウト: 横並びの3つの枠
  3. 内容: 最新の予報データを 日付(曜日) | 天気アイコン | 最高気温/最低気温 の形式で3日分記述、下部には手書きで服装や傘の有無のアドバイスを追加
  4. フォント: "Noto Sans JP" (Bold) で視認性を高く
  5. 大きさはお天気キャスターが両手で持てる程度に調整する

### 4. 背景
- 場所: 前提条件で指定された場所の風景
- 状況: 人（通行人）は存在する
- 配置: キャスターの周囲（至近距離）には人がおらず、クリアな空間が確保されている。通行人は遠景や中景に配置する
- 被写界深度: 背景の街並みと通行人は、カメラの絞りを開放したように深くぼけている（ボケ味）。キャスターのみにピントが合っている
- ライティング: {CURRENT_WEATHER} と {CURRENT_TIME} を反映したリアルな環境
- キャラクターにもリアルなライティングとPBRを適用し、背景と調和させる
"""
PROMPT = PROMPT_WEATHER_REPORT

# アスペクト比
ASPECT_RATIO = "9:16"  # "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"

# 解像度 (Gemini 3 系のみ有効)
RESOLUTION = "1K"

# 保存形式は JPEG を想定（軽量）
RESPONSE_MIME_TYPE = "image/jpeg"
OUTPUT_EXT = ".jpg"

# 出力設定
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d%H%M")
output_file = OUTPUT_DIR / f"nanobanana_google_search_{timestamp}{OUTPUT_EXT}"

# Gemini Developer API クライアント作成
client = genai.Client(api_key=API_KEY, vertexai=False)

print(f"モデル: {MODEL_ID}")
print(f"プロンプト: {PROMPT}")
print(f"アスペクト比: {ASPECT_RATIO}")
print(f"解像度 (Gemini 3 のみ有効): {RESOLUTION}")
print("Google 検索ツールを使って画像を生成中...")

gen_config = types.GenerateContentConfig(
    response_modalities=["TEXT", "IMAGE"],
    image_config=types.ImageConfig(
        aspect_ratio=ASPECT_RATIO,
        image_size=RESOLUTION,
    ),
    tools=[{"google_search": {}}],
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[PROMPT],
    config=gen_config,
)


def extract_image_bytes(resp) -> bytes:
    """Gemini のレスポンスから最初の画像バイト列を取り出す。"""
    parts = getattr(resp, "parts", None)
    candidates = getattr(resp, "candidates", None)

    # response.parts を優先
    if parts:
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                data_field = inline_data.data
                if isinstance(data_field, bytes):
                    return data_field
                if isinstance(data_field, str):
                    return base64.b64decode(data_field)

    # candidates 経由のフォールバック
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


def extract_text_parts(resp) -> str:
    """Gemini レスポンスからテキスト応答を連結して取り出す。"""
    texts = []
    parts = getattr(resp, "parts", None)
    candidates = getattr(resp, "candidates", None)

    if parts:
        for part in parts:
            if getattr(part, "text", None):
                texts.append(part.text)

    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            c_parts = getattr(content, "parts", None)
            if not c_parts:
                continue
            for part in c_parts:
                if getattr(part, "text", None):
                    texts.append(part.text)

    return "\n".join(texts)


# テキスト応答を表示
text_response = extract_text_parts(response)
print("\n=== Text Response ===")
print(text_response)

# 画像データを取得して保存
try:
    image_data = extract_image_bytes(response)
except Exception as e:
    print(f"\nエラー: 画像データの取得に失敗しました: {e}")
else:
    # フォーマットの簡易判定（ログ用）
    if image_data[:8] == b"\x89PNG\r\n\x1a\n":
        actual_format = "PNG"
    elif image_data[:2] == b"\xff\xd8":
        actual_format = "JPEG"
    else:
        actual_format = "UNKNOWN"

    print(f"実際の画像フォーマット(先頭バイト判定): {actual_format}")

    with open(output_file, "wb") as f:
        f.write(image_data)

    print(f"\n画像を生成しました: {output_file}")
    print(f"ファイルサイズ: {len(image_data)} バイト")
