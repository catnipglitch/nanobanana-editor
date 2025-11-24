"""
Image Generation CLI

画像生成のコマンドラインインターフェース。
ImagenとGemini 2.5 Flash Imageの両方に対応。
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.generators import GenerationConfig, ModelType, GeminiImageGenerator, ImagenImageGenerator, TestImageGenerator
from src.core.model_specs import ModelRegistry
from src.core.output_manager import OutputManager


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="画像生成CLI - ImagenとGemini 2.5 Flash Imageに対応"
    )

    # 必須パラメータ
    parser.add_argument(
        "prompt",
        type=str,
        help="画像生成用のテキストプロンプト"
    )

    # モデル選択
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemini-2.5-flash-image",
        help="使用するモデル名 (デフォルト: gemini-2.5-flash-image)"
    )

    # 出力設定
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output",
        help="出力ディレクトリ (デフォルト: output)"
    )

    parser.add_argument(
        "-p", "--prefix",
        type=str,
        default="generated",
        help="出力ファイル名のプレフィックス (デフォルト: generated)"
    )

    # 画像生成パラメータ
    parser.add_argument(
        "-n", "--number-of-images",
        type=int,
        default=1,
        help="生成する画像の枚数 (Imagen/Test用、デフォルト: 1、Imagenの上限: 4)"
    )

    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="1:1",
        choices=["1:1", "9:16", "16:9", "4:3", "3:4"],
        help="アスペクト比 (デフォルト: 1:1)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="再現性のための乱数シード値 (Imagen/Test用、オプション)"
    )

    # 非推奨パラメータ（後方互換性のため保持）
    parser.add_argument(
        "--safety-filter",
        type=str,
        default="block_only_high",
        choices=["block_low_and_above", "block_medium_and_above", "block_only_high", "block_none",
                 "block_some", "block_few", "block_most"],  # 旧パラメータも後方互換性のため保持
        help="[非推奨] 安全フィルタレベル (現在は無視されます)"
    )

    parser.add_argument(
        "--person-generation",
        type=str,
        default="allow_adult",
        choices=["allow_adult", "allow_all", "block_all"],
        help="[非推奨] 人物生成の許可レベル (現在は無視されます)"
    )

    # 認証情報の上書き（オプション）
    parser.add_argument(
        "--use-vertexai",
        action="store_true",
        help="APIキー認証を使用（デフォルト: 環境変数から読み込み）"
    )

    parser.add_argument(
        "--use-adc",
        action="store_true",
        help="ADC認証を使用（デフォルト: 環境変数から読み込み）"
    )

    parser.add_argument(
        "--gcp-project-id",
        type=str,
        help="Google Cloud プロジェクトID (ADC認証用、環境変数を上書き)"
    )

    parser.add_argument(
        "--gcp-location",
        type=str,
        help="GCPリージョン (環境変数を上書き)"
    )

    parser.add_argument(
        "--google-api-key",
        type=str,
        help="Google APIキー (APIキー認証用、環境変数を上書き)"
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    # 環境変数を読み込み
    load_dotenv()

    # コマンドライン引数をパース
    args = parse_args()

    print("=" * 60)
    print("画像生成CLI")
    print("=" * 60)

    # 認証方式を決定（コマンドライン引数 > 環境変数）
    if args.use_vertexai:
        use_vertexai = True
    elif args.use_adc:
        use_vertexai = False
    else:
        # 環境変数から読み込み
        use_vertexai_str = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower()
        use_vertexai = use_vertexai_str in ("true", "1", "yes")

    # 認証情報を取得（コマンドライン引数 > 環境変数）
    gcp_project_id = args.gcp_project_id or os.getenv("GCP_PROJECT_ID")
    gcp_location = args.gcp_location or os.getenv("GCP_LOCATION", "us-central1")
    google_api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY")

    # モデルタイプを判定
    try:
        model_spec = ModelRegistry.get_model_spec(args.model)
        if not model_spec:
            raise ValueError(f"Unknown model: {args.model}")
        model_type = model_spec.model_type
    except ValueError as e:
        print(f"エラー: {e}")
        sys.exit(1)

    # 認証情報のチェック
    if use_vertexai:
        if not google_api_key:
            print("エラー: APIキー認証モードではGOOGLE_API_KEYが必要です")
            print("  --google-api-key オプションを指定するか、.envファイルに設定してください")
            print("\n認証方式の設定:")
            print("  APIキー認証: GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_API_KEY")
            print("  ADC認証: GOOGLE_GENAI_USE_VERTEXAI=false + GCP_PROJECT_ID")
            sys.exit(1)
    else:
        if not gcp_project_id:
            print("エラー: ADC認証モードではGCP_PROJECT_IDが必要です")
            print("  --gcp-project-id オプションを指定するか、.envファイルに設定してください")
            print("\n認証方式の設定:")
            print("  APIキー認証: GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_API_KEY")
            print("  ADC認証: GOOGLE_GENAI_USE_VERTEXAI=false + GCP_PROJECT_ID")
            sys.exit(1)

    # 画像生成器とアウトプットマネージャーを初期化
    try:
        # モデルタイプに応じてジェネレーターを選択
        if model_type == ModelType.GEMINI:
            generator = GeminiImageGenerator(google_api_key=google_api_key)
        elif model_type == ModelType.IMAGEN:
            generator = ImagenImageGenerator(google_api_key=google_api_key)
        elif model_type == ModelType.TEST:
            generator = TestImageGenerator(google_api_key="dummy_key_for_test")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except ValueError as e:
        print(f"\nエラー: {e}")
        print("\n認証方式の設定:")
        print("  APIキー認証: GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_API_KEY")
        print("  ADC認証: GOOGLE_GENAI_USE_VERTEXAI=false + GCP_PROJECT_ID")
        sys.exit(1)
    output_manager = OutputManager(output_dir=args.output_dir)

    # 生成設定を作成
    config = GenerationConfig(
        model_type=model_type,
        model_name=args.model,
        prompt=args.prompt,
        number_of_images=args.number_of_images,
        aspect_ratio=args.aspect_ratio,
        seed=args.seed,
        safety_filter_level=args.safety_filter,
        person_generation=args.person_generation,
    )

    # 設定を表示
    print(f"\n認証方式: {'APIキー認証' if use_vertexai else 'ADC認証'}")
    print(f"モデル: {config.model_name}")
    print(f"プロンプト: {config.prompt}")
    print(f"画像枚数: {config.number_of_images}")
    print(f"アスペクト比: {config.aspect_ratio}")
    if config.seed is not None:
        print(f"Seed: {config.seed}")

    # 画像を生成
    print("\n画像を生成中...")
    try:
        image_data_list, metadata = generator.generate(config)
    except Exception as e:
        print(f"\nエラー: 画像生成に失敗しました")
        print(f"  {e}")
        sys.exit(1)

    # 画像とメタデータを保存
    print("画像とメタデータを保存中...")
    try:
        # Gemini の画像生成結果は JPEG なので、拡張子を .jpg にする
        # Imagen やテストモデルは従来どおり PNG を使用
        extension = "jpg" if config.model_type == ModelType.GEMINI else "png"

        image_paths, metadata_path = output_manager.save_images_with_metadata(
            image_data_list=image_data_list,
            metadata=metadata,
            prefix=args.prefix,
            extension=extension,
        )
    except Exception as e:
        print(f"\nエラー: ファイル保存に失敗しました")
        print(f"  {e}")
        sys.exit(1)

    # 結果を表示
    print("\n" + "=" * 60)
    print("✓ 画像生成が完了しました!")
    print("=" * 60)
    print(f"生成枚数: {len(image_paths)}")
    for i, image_path in enumerate(image_paths, 1):
        print(f"画像ファイル {i}: {image_path}")
    print(f"メタデータ: {metadata_path}")
    total_size = sum(len(data) for data in image_data_list)
    print(f"合計サイズ: {total_size:,} バイト")
    print("=" * 60)


if __name__ == "__main__":
    main()
