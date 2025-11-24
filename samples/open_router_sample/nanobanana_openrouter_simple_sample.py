"""
Nanobanana Text-to-Image サンプル (OpenRouter 版)

OpenRouter 経由で Google Gemini 3 Pro Image Preview
(`google/gemini-3-pro-image-preview`) を呼び出して、
テキストプロンプトから画像を生成するサンプルコードです。

特徴:
- .env の OPENROUTER_API_KEY を使用
- OpenRouter Chat Completions API を利用して画像生成
- 生成結果の Base64 Data URL をパースして画像ファイルとして保存
- パラメータと結果サマリを JSON で保存

事前準備:
- .env に以下の環境変数を設定してください
    OPENROUTER_API_KEY=あなたの_api_key

実行例:
- python samples/open_router_sample/nanobanana_openrouter_simple_sample.py

参考:
- https://openrouter.ai/google/gemini-3-pro-image-preview/api
"""

import os
import json
import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image

# .env の読み込み
load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "google/gemini-3-pro-image-preview"
OUTPUT_DIR = Path("output")

# ====================
# 画像生成設定 (ここで切り替え)
# ====================
# アスペクト比をコード内で選択
#   利用可能な値:
#     "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
ASPECT_RATIO = "16:9"

# 解像度をコード内で選択（Gemini 3 系のみ有効）
#   利用可能な値: "1K", "2K", "4K"
RESOLUTION = "4K"

# Split Alpha機能
#   True: 画像の左半分をRGB、右半分をアルファチャンネルとして合成したPNGを生成
#   False: 通常の画像保存のみ
SPLIT_ALPHA = True
# ====================

# 元 nanobanana サンプルと同様のイメージプロンプト
DEFAULT_PROMPT = (
    "A full-body shot of a young female model with long hair, posing for high-end fashion photography in a professional studio. "
    "The composition must be clean, strictly devoid of any text, logos, magazine overlays, or typography. "
    "The image captures her entire figure from head to toe. \n"
    "formatted as a side-by-side split-screen layout. The left half displays the subject in full color. "
    "The right half displays the generated binary alpha matte against a pure black background. "
    "The matte must be a solid white silhouette covering the entire figure of the subject, "
    "completely including her body, clothing, hair, shoes, and any held objects. "
    "Within this complete figure silhouette, the edges must strictly preserve high-frequency details "
    "such as individual hair strands without simplification or blobbing. "
    "Ensure pixel-perfect alignment between the color subject's complete area and the white silhouette mask for compositing."
)

def get_api_key() -> str:
    """OPENROUTER_API_KEY を環境変数から取得し、未設定なら例外を投げる。"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "環境変数 OPENROUTER_API_KEY が設定されていません。.env を確認してください。"
        )
    return api_key


def build_payload(prompt: str) -> Dict[str, Any]:
    """OpenRouter Chat Completions API 用のリクエストペイロードを構築する。"""
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        # 画像とテキスト両方の出力を有効にする
        "modalities": ["image", "text"],
    }

    # Gemini 3 系のモデルの場合、アスペクト比と解像度を追加
    # ※ OpenRouter の正しいパラメータ構造を検証中
    # 以下の3つのアプローチを試行可能（コメントアウトで切り替え）

    if MODEL_ID.startswith("google/gemini-3-"):
        # アプローチA: トップレベルに image_config を配置（1376x768で失敗）
        # payload["image_config"] = {
        #     "aspect_ratio": ASPECT_RATIO,
        #     "image_size": RESOLUTION,
        # }

        # アプローチB: generation_config でネスト（試行中）
        payload["generation_config"] = {
            "image_config": {
                "aspect_ratio": ASPECT_RATIO,
                "image_size": RESOLUTION,
            }
        }

        # アプローチC: config でネスト（試行用、コメントアウト解除して試す）
        # payload["config"] = {
        #     "image_config": {
        #         "aspect_ratio": ASPECT_RATIO,
        #         "image_size": RESOLUTION,
        #     }
        # }

    return payload


def call_openrouter(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """OpenRouter API を呼び出し、JSON レスポンスを辞書として返す。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"\n{'='*60}")
    print("OpenRouter API にリクエストを送信します")
    print(f"{'='*60}")
    print(f"エンドポイント: {OPENROUTER_API_URL}")
    print(f"\n送信ペイロード:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"{'='*60}\n")

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)

    print(f"\nレスポンスステータス: {response.status_code}")

    if response.status_code != 200:
        print(f"エラーレスポンス: {response.text}")
        raise RuntimeError(
            f"OpenRouter API エラー: status={response.status_code}, body={response.text}"
        )

    response_json = response.json()
    print(f"レスポンス構造キー: {list(response_json.keys())}")

    # レスポンスの詳細情報を出力
    if "usage" in response_json:
        print(f"使用トークン情報: {response_json['usage']}")
    if "model" in response_json:
        print(f"使用されたモデル: {response_json['model']}")

    return response_json


def extract_image_data_url(result: Dict[str, Any]) -> Optional[str]:
    """
    OpenRouter レスポンスから最初の画像の Data URL を抽出する。

    期待する構造:
      result["choices"][0]["message"]["images"][0]["image_url"]["url"]
    """
    choices = result.get("choices")
    if not choices:
        print("レスポンスに choices が含まれていません。")
        return None

    message = choices[0].get("message") or {}
    images = message.get("images")
    if not images:
        print("message.images が空、または存在しません。")
        return None

    first_image = images[0]
    image_url = (first_image.get("image_url") or {}).get("url")
    if not image_url:
        print("image_url.url が見つかりません。レスポンス構造を確認してください。")
        return None

    if not isinstance(image_url, str):
        print("image_url.url が文字列ではありません。")
        return None

    return image_url


def parse_data_url(data_url: str) -> Tuple[str, bytes]:
    """
    data:[mime];base64,[payload] 形式の Data URL から
    MIME タイプとデコード済みバイナリを返す。
    """
    if not data_url.startswith("data:"):
        raise ValueError("Data URL 形式ではありません。")

    try:
        header, b64_data = data_url.split(",", 1)
    except ValueError as exc:
        raise ValueError("Data URL の区切り (,) を解釈できません。") from exc

    # 例: data:image/png;base64
    if ";base64" not in header:
        raise ValueError("base64 エンコードされた Data URL ではありません。")

    mime_type = header[len("data:") : header.index(";base64")]
    try:
        binary = base64.b64decode(b64_data)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Data URL の base64 部分をデコードできません。") from exc

    return mime_type, binary


def guess_extension(mime_type: str) -> str:
    """MIME タイプからファイル拡張子を推定する。"""
    if mime_type == "image/png":
        return ".png"
    if mime_type in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    # 想定外の MIME はいったん JPG 扱いにする
    return ".jpg"


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


def save_image(
    image_bytes: bytes,
    mime_type: str,
    timestamp: str,
) -> Path:
    """画像バイナリを output ディレクトリに保存し、パスを返す。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    ext = guess_extension(mime_type)
    output_file = OUTPUT_DIR / f"nanobanana_openrouter_simple_sample_{timestamp}{ext}"

    with open(output_file, "wb") as f:
        f.write(image_bytes)

    print(f"画像を保存しました: {output_file} ({len(image_bytes)} bytes)")
    return output_file


def save_metadata(
    timestamp: str,
    prompt: str,
    model: str,
    api_endpoint: str,
    output_image_path: Optional[Path],
    image_size_bytes: Optional[int],
    alpha_image_path: Optional[Path],
    alpha_image_size_bytes: Optional[int],
    raw_response: Dict[str, Any],
    request_payload: Dict[str, Any] = {},
) -> Path:
    """生成パラメータと結果サマリを JSON として保存する。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    json_output_file = OUTPUT_DIR / f"nanobanana_openrouter_simple_sample_{timestamp}.json"

    # raw_response をサニタイズ（Base64 画像データが大きすぎる場合）
    safe_raw_response = raw_response.copy()
    if "choices" in safe_raw_response:
        for choice in safe_raw_response["choices"]:
            if "message" in choice and "images" in choice["message"]:
                # images リストをコピーして変更
                choice["message"]["images"] = [img.copy() for img in choice["message"]["images"]]
                for img in choice["message"]["images"]:
                    if "image_url" in img and "url" in img["image_url"]:
                         url = img["image_url"]["url"]
                         if isinstance(url, str) and url.startswith("data:") and len(url) > 1000:
                             img["image_url"]["url"] = "<base64_data_truncated_for_log>"

    payload: Dict[str, Any] = {
        "timestamp": timestamp,
        "prompt": prompt,
        "generation": {
            "model": model,
            "api_endpoint": api_endpoint,
            "aspect_ratio": ASPECT_RATIO,
            "resolution": RESOLUTION,
            "split_alpha_enabled": SPLIT_ALPHA,
            "output_image_path": str(output_image_path) if output_image_path else None,
            "image_saved": output_image_path is not None,
            "image_size_bytes": image_size_bytes,
            "alpha_image_path": str(alpha_image_path) if alpha_image_path else None,
            "alpha_image_saved": alpha_image_path is not None,
            "alpha_image_size_bytes": alpha_image_size_bytes,
        },
        "request_payload": request_payload,
        "raw_response": safe_raw_response,
    }

    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"メタデータを保存しました: {json_output_file}")
    return json_output_file


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    prompt = DEFAULT_PROMPT

    print(f"モデル: {MODEL_ID}")
    print(f"プロンプト: {prompt}")
    print(f"アスペクト比: {ASPECT_RATIO}")
    print(f"解像度: {RESOLUTION}")
    print("画像を生成中...")

    api_key = get_api_key()
    payload = build_payload(prompt)
    result = call_openrouter(api_key, payload)

    data_url = extract_image_data_url(result)
    image_path: Optional[Path] = None
    image_size: Optional[int] = None
    alpha_image_path: Optional[Path] = None
    alpha_image_size: Optional[int] = None

    if data_url is None:
        print("画像 Data URL を取得できませんでした。")
    else:
        try:
            mime_type, image_bytes = parse_data_url(data_url)

            # 画像の実際の解像度を確認
            img = Image.open(BytesIO(image_bytes))
            actual_width, actual_height = img.size
            print(f"\n{'='*60}")
            print("生成された画像の解像度情報")
            print(f"{'='*60}")
            print(f"実際の解像度: {actual_width} x {actual_height} pixels")
            print(f"設定値: アスペクト比={ASPECT_RATIO}, 解像度={RESOLUTION}")

            # 4K相当の解像度チェック（16:9の場合、4K = 3840x2160）
            if RESOLUTION == "4K" and ASPECT_RATIO == "16:9":
                expected_width, expected_height = 3840, 2160
                if actual_width == expected_width and actual_height == expected_height:
                    print("✓ 4K解像度が正しく適用されています")
                else:
                    print(f"✗ 警告: 期待値 {expected_width}x{expected_height} と異なります")
            print(f"{'='*60}\n")

            image_path = save_image(image_bytes, mime_type, timestamp)
            image_size = len(image_bytes)

            # Split Alpha機能が有効な場合、アルファチャンネル合成処理を実行
            if SPLIT_ALPHA:
                print("\nSplit Alpha処理を実行中...")
                alpha_bytes = split_alpha_channel(image_bytes)
                alpha_output_file = OUTPUT_DIR / f"nanobanana_openrouter_simple_sample_{timestamp}_alpha.png"

                with open(alpha_output_file, "wb") as f:
                    f.write(alpha_bytes)

                alpha_image_path = alpha_output_file
                alpha_image_size = len(alpha_bytes)
                print(f"アルファチャンネル合成画像を保存しました: {alpha_output_file} ({len(alpha_bytes)} bytes)")

        except Exception as exc:  # noqa: BLE001
            print(f"画像データの処理中にエラーが発生しました: {exc}")

    save_metadata(
        timestamp=timestamp,
        prompt=prompt,
        model=MODEL_ID,
        api_endpoint=OPENROUTER_API_URL,
        output_image_path=image_path,
        image_size_bytes=image_size,
        alpha_image_path=alpha_image_path,
        alpha_image_size_bytes=alpha_image_size,
        raw_response=result,
        request_payload=payload,
    )


if __name__ == "__main__":
    main()
