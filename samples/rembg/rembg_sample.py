"""
rembg 背景削除サンプル

rembgライブラリを使用して、画像の背景を削除するサンプルコードです。

特徴:
- 入力画像: JPG, PNG 対応
- 出力画像: PNG で背景透過画像を保存
- パラメーターはコード上部の変数で変更可能

事前準備:
- rembgライブラリのインストール: pip install rembg
- 必要に応じて依存関係をインストール: pip install pillow

実行例:
- python samples/rembg/rembg_sample.py

参考:
- rembg GitHub: https://github.com/danielgatis/rembg
"""

from pathlib import Path
from datetime import datetime
from PIL import Image
from rembg import remove

# ====================
# 設定 (ここで切り替え)
# ====================
# 入力画像パス（ワークスペースルートからの相対パス）
INPUT_IMAGE = "samples/sample_images/unsplash/nanobanana_edit_alpha_multi_turn_202511240011_edited.png"

# 出力ディレクトリ（Noneの場合は"output"フォルダ）
OUTPUT_DIR = None  # 例: "output" or None

# モデル選択
#   "u2net" - 汎用的な背景削除（デフォルト、高精度）
#   "u2netp" - 軽量版（高速だが精度は若干低い）
#   "u2net_human_seg" - 人物セグメンテーション特化
#   "u2net_cloth_seg" - 衣服セグメンテーション特化
#   "silueta" - 人物シルエット抽出特化
MODEL_NAME = "u2net"

# アルファマット処理
#   True: エッジを滑らかにする（高品質）
#   False: エッジ処理なし（高速）
ALPHA_MATTING = False

# アルファマット処理の詳細設定（ALPHA_MATTING=Trueの場合のみ有効）
ALPHA_MATTING_FOREGROUND_THRESHOLD = 240  # 前景閾値 (0-255)
ALPHA_MATTING_BACKGROUND_THRESHOLD = 10  # 背景閾値 (0-255)
ALPHA_MATTING_ERODE_SIZE = 10  # エロージョンサイズ (ピクセル)

# 出力ファイル名のプレフィックス
OUTPUT_PREFIX = "rembg_output"
# ====================


def remove_background(
    input_path: Path,
    output_path: Path,
    model: str = "u2net",
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
) -> None:
    """
    画像から背景を削除してPNG形式で保存する。

    Args:
        input_path: 入力画像のパス
        output_path: 出力画像のパス
        model: 使用するモデル名
        alpha_matting: アルファマット処理を有効にするか
        alpha_matting_foreground_threshold: 前景閾値
        alpha_matting_background_threshold: 背景閾値
        alpha_matting_erode_size: エロージョンサイズ
    """
    print(f"入力画像: {input_path}")
    print(f"モデル: {model}")
    print(f"アルファマット処理: {'有効' if alpha_matting else '無効'}")

    # 画像を読み込み
    with open(input_path, "rb") as input_file:
        input_data = input_file.read()

    # 背景削除処理
    print("背景削除処理中...")
    output_data = remove(
        input_data,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=alpha_matting_background_threshold,
        alpha_matting_erode_size=alpha_matting_erode_size,
    )

    # 結果を保存
    with open(output_path, "wb") as output_file:
        output_file.write(output_data)

    # 画像サイズ情報を表示
    img = Image.open(output_path)
    width, height = img.size
    print(f"\n出力画像: {output_path}")
    print(f"解像度: {width} x {height} pixels")
    print(f"ファイルサイズ: {len(output_data)} bytes")


def main() -> None:
    # ワークスペースルートからの相対パスで入力画像を指定
    input_path = Path(INPUT_IMAGE)

    # 入力ファイルの存在確認
    if not input_path.exists():
        raise FileNotFoundError(
            f"入力画像が見つかりません: {input_path}\n"
            f"ワークスペースルートからの相対パスで {INPUT_IMAGE} を配置してください。"
        )

    # 出力ディレクトリの設定
    if OUTPUT_DIR is None:
        output_dir = Path("output")
    else:
        output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # 出力ファイル名の生成（タイムスタンプ付き）
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"{OUTPUT_PREFIX}_{timestamp}.png"
    output_path = output_dir / output_filename

    print("=" * 60)
    print("rembg 背景削除サンプル")
    print("=" * 60)

    # 背景削除処理を実行
    remove_background(
        input_path=input_path,
        output_path=output_path,
        model=MODEL_NAME,
        alpha_matting=ALPHA_MATTING,
        alpha_matting_foreground_threshold=ALPHA_MATTING_FOREGROUND_THRESHOLD,
        alpha_matting_background_threshold=ALPHA_MATTING_BACKGROUND_THRESHOLD,
        alpha_matting_erode_size=ALPHA_MATTING_ERODE_SIZE,
    )

    print("=" * 60)
    print("処理完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
