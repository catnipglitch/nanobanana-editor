"""
Vertex AI Generative Models に ADC（Application Default Credentials）で
アクセスするシンプルなサンプルコード。

API キーは使用せず、GCP の認証情報（ADC）を利用して
Gemini モデルにテキストを 1 回送信する動作を確認できます。

事前準備:
- GCP_PROJECT_ID, LOCATION, GEMINI_DEFAULT_MODEL を .env に設定
- ADC（gcloud auth application-default login など）を有効化

実行例:
- python samples/gcp_samples/vertexai_adc_login.py
"""

from vertexai.generative_models import GenerativeModel
import vertexai
import os
from dotenv import load_dotenv

def main() -> None:
    """ADC を用いて Vertex AI Gemini モデルに接続し、1 文生成する。"""
    load_dotenv()

    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("LOCATION", "us-central1")
    gemini_model = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.0-flash")

    if not project_id:
        raise RuntimeError(
            "環境変数 GCP_PROJECT_ID が設定されていません。.env を確認してください。"
        )

    vertexai.init(project=project_id, location=location)

    model = GenerativeModel(gemini_model)
    resp = model.generate_content("こんにちは、世界")
    print(resp.text)


if __name__ == "__main__":
    main()
