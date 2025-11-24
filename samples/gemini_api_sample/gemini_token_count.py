# トークンカウントサンプル

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# 1. APIキーを設定
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=API_KEY)

def get_token_count(text_input):
    """
    テキストを受け取り、正確なトークン数を返す関数
    """
    # 計測に使うモデルを指定（Proモデル系ならトークン計算方法はほぼ同じです）
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # APIに問い合わせてカウント（この問い合わせ自体は無料です）
    response = model.count_tokens(text_input)
    return response.total_tokens

# --- 実行テスト ---
my_prompt = "Geminiの料金体系について学習中です。トークン計算は重要です。"

# 関数を呼び出す
count = get_token_count(my_prompt)

print(f"テキスト: {my_prompt}")
print(f"トークン数: {count}")