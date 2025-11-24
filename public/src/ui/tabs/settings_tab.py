"""
Settings Tab

APIキー管理とアプリケーション設定を行うタブ。
"""

import gradio as gr
import logging
from .base_tab import BaseTab
from ...core.tab_specs import TAB_SETTINGS

logger = logging.getLogger(__name__)


class SettingsTab(BaseTab):
    """Settings タブ（APIキー管理）"""

    def create_ui(self) -> None:
        """Settings タブのUIを構築"""
        with gr.Tab(TAB_SETTINGS.display_name, id=TAB_SETTINGS.key, elem_id=TAB_SETTINGS.elem_id):
            gr.Markdown("""
            ### APIキー設定

            Google Gen AI のAPIキーを設定します。

            - `.env` ファイルから自動読み込みされますが、ここで上書きできます
            - 入力したAPIキーはブラウザのセッションストレージに保存されます（タブを閉じると消えます）
            - セキュリティのため、APIキーはコード内に保持されません
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    settings_api_key_input = gr.Textbox(
                        label="Google API Key",
                        placeholder="AIza...",
                        type="password",
                        lines=1,
                        info="Google Gen AI APIキー（マスク表示）"
                    )

                    # APIキー適用ボタン（上）
                    settings_apply_button = gr.Button("APIキーを適用", variant="primary", size="lg")

                    gr.Markdown("""
                    #### 接続テスト

                    **1️⃣ トークンカウントテスト（無料）**
                    - APIキーの有効性のみ確認
                    - 料金は発生しません
                    - Free Tier でも利用可能

                    **2️⃣ 画像生成テスト（💰有料）**
                    - 実際に画像を1枚生成して完全動作確認
                    - 約$0.134の料金が発生します
                    - **Paid Tier のAPIキーが必要です**

                    ⚠️ **重要**: トークンカウントで成功しても、Free Tier の場合は画像生成APIにアクセスできません。Paid Tier のAPIキーを取得してください。
                    """)

                    # 接続テストボタン（下）
                    with gr.Row():
                        settings_token_test_button = gr.Button("1️⃣ トークンカウントテスト（無料）", variant="secondary")
                        settings_image_test_button = gr.Button("2️⃣ 画像生成テスト（有料）", variant="secondary")

                    settings_status = gr.Markdown("", visible=True)

                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 現在の設定

                    **APIキーステータス**:
                    """)
                    if self.app.google_api_key:
                        gr.Markdown(f"✅ APIキーが設定されています（末尾: ...{self.app.google_api_key[-4:]}）")
                    else:
                        gr.Markdown("❌ APIキーが設定されていません")

                    gr.Markdown("""
                    ### APIキーの取得方法

                    1. [Google AI Studio](https://aistudio.google.com/app/apikey) にアクセス
                    2. 「Get API Key」をクリック
                    3. プロジェクトを選択またはCreate
                    4. APIキーをコピー
                    5. 上記のフィールドに貼り付けて「接続テスト」

                    ### セキュリティに関する注意

                    - APIキーは厳重に管理してください
                    - 公開リポジトリにコミットしないでください
                    - `.env` ファイルは `.gitignore` に追加されています
                    """)

            # トークンカウントテストボタン（無料）
            settings_token_test_button.click(
                fn=self.test_api_token_count,
                inputs=[settings_api_key_input],
                outputs=[settings_status]
            )

            # 画像生成テストボタン（有料）
            settings_image_test_button.click(
                fn=self.test_api_image_generation,
                inputs=[settings_api_key_input],
                outputs=[settings_status]
            )

            # APIキー適用ボタン
            settings_apply_button.click(
                fn=self.update_api_key,
                inputs=[settings_api_key_input],
                outputs=[settings_status]
            )

    def test_api_token_count(self, api_key: str):
        """
        トークンカウントでAPIキーをテストする（無料）

        Args:
            api_key: テストするAPIキー

        Returns:
            status_message: ステータスメッセージ
        """
        if not api_key or api_key.strip() == "":
            return "⚠ APIキーを入力してください"

        logger.info("Testing API connection with token count...")

        try:
            import google.generativeai as genai

            # APIキーを設定
            genai.configure(api_key=api_key)

            # トークンカウントでテスト（無料）
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            test_text = "API key connection test"
            response = model.count_tokens(test_text)

            logger.info(f"Token count test successful: {response.total_tokens} tokens")
            return f"""✅ トークンカウントテスト成功！

**結果**: APIキーは有効です（テストテキスト: {response.total_tokens} トークン）

⚠️ **注意**: このテストは無料ですが、APIキーの有効性のみを確認します。
Free Tier の場合、画像生成APIにはアクセスできません。
完全な動作確認には「画像生成テスト」を実行してください（Paid Tier が必要、有料）。
"""

        except Exception as e:
            logger.error(f"Token count test failed: {e}", exc_info=True)
            return f"""❌ トークンカウントテスト失敗

**エラー**: {str(e)}

**考えられる原因**:
- APIキーが無効
- ネットワーク接続の問題
- Gemini APIへのアクセス権限がない
"""

    def test_api_image_generation(self, api_key: str):
        """
        実際の画像生成でAPIキーをテストする（有料）

        Args:
            api_key: テストするAPIキー

        Returns:
            status_message: ステータスメッセージ
        """
        if not api_key or api_key.strip() == "":
            return "⚠ APIキーを入力してください"

        logger.info("Testing API connection with image generation...")

        try:
            from google import genai
            from google.genai import types

            # Gemini Developer API クライアントを作成
            client = genai.Client(api_key=api_key, vertexai=False)

            # 画像生成設定
            model_name = "gemini-3-pro-image-preview"
            prompt = "A simple test image: a red circle"

            gen_config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="1:1",
                    image_size="1K",
                ),
            )

            # 実際に画像生成（有料）
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=gen_config,
            )

            # 画像データを取得
            image_data = None
            if hasattr(response, 'parts') and response.parts:
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        if hasattr(part.inline_data, 'data'):
                            data_field = part.inline_data.data
                            if isinstance(data_field, bytes):
                                image_data = data_field
                                break
                            elif isinstance(data_field, str):
                                import base64
                                image_data = base64.b64decode(data_field)
                                break

            image_data_list = [image_data] if image_data else None

            if image_data_list and len(image_data_list) > 0:
                logger.info("Image generation test successful")
                return f"""✅ 画像生成テスト成功！

**結果**: APIキーは完全に動作しています。
**生成サイズ**: {len(image_data_list[0]):,} バイト

💰 **料金**: このテストで約$0.134の料金が発生しました（1024x1024画像生成）。
✅ **確認**: 画像生成APIが正常に動作することを確認しました。
"""
            else:
                logger.error("Image generation test failed: No image data returned")
                return "❌ 画像生成テスト失敗: 画像データが取得できませんでした"

        except Exception as e:
            logger.error(f"Image generation test failed: {e}", exc_info=True)
            error_msg = str(e)

            # Free Tier の可能性を判定
            if "403" in error_msg or "permission" in error_msg.lower() or "quota" in error_msg.lower():
                return f"""❌ 画像生成テスト失敗（権限エラー）

**エラー**: {error_msg}

**考えられる原因**:
- **Free Tier** のAPIキーを使用している可能性があります
- Free Tier ではGemini Chat APIには使えますが、画像生成APIには使えません
- **Paid Tier** にアップグレードする必要があります

**解決方法**:
1. [Google AI Studio](https://aistudio.google.com/app/apikey) にアクセス
2. Billing（課金）を有効化してPaid Tier にアップグレード
3. 新しいAPIキーを取得（または既存のキーが自動的にPaid Tier になります）
4. 画像生成APIへのアクセスが可能になります
"""
            else:
                return f"""❌ 画像生成テスト失敗

**エラー**: {error_msg}

**考えられる原因**:
- ネットワーク接続の問題
- APIの一時的な障害
- 課金設定が有効になっていない
"""

    def update_api_key(self, api_key: str):
        """
        APIキーを更新してジェネレーターを再初期化する

        Args:
            api_key: 新しいAPIキー

        Returns:
            status_message: ステータスメッセージ
        """
        if not api_key or api_key.strip() == "":
            return "⚠ APIキーを入力してください"

        logger.info("Updating API key and reinitializing generators...")

        try:
            from ...core.generators import GeminiImageGenerator, ImagenImageGenerator, TestImageGenerator

            # 新しいAPIキーでジェネレーターを再初期化
            self.app.google_api_key = api_key
            self.app.gemini_generator = GeminiImageGenerator(google_api_key=api_key)
            self.app.imagen_generator = ImagenImageGenerator(google_api_key=api_key)
            self.app.test_generator = TestImageGenerator(google_api_key="dummy_key_for_test")
            self.app.api_key_missing = False

            logger.info("API key updated and generators reinitialized successfully")
            return "✅ APIキーを更新しました。画像生成が可能です。"

        except Exception as e:
            logger.error(f"Failed to update API key: {e}", exc_info=True)
            self.app.api_key_missing = True
            return f"❌ APIキーの更新に失敗しました: {str(e)}"
