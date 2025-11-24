# https://ai.google.dev/gemini-api/docs/image-generation?hl=ja&batch=file#python_8

# マルチターンの画像編集
# 会話形式で画像の生成と編集を続けます。画像に対して反復処理を行うには、チャットまたはマルチターンの会話をおすすめします。次の例は、光合成に関するインフォグラフィックを生成するプロンプトを示しています。

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# API key setup
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

# Output directory setup
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Create client with API key
client = genai.Client(api_key=API_KEY, vertexai=False)

chat = client.chats.create(
    model="gemini-3-pro-image-preview",
    config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE'],
        tools=[{"google_search": {}}]
    )
)

message = "Create a vibrant infographic that explains photosynthesis as if it were a recipe for a plant's favorite food. Show the \"ingredients\" (sunlight, water, CO2) and the \"finished dish\" (sugar/energy). The style should be like a page from a colorful kids' cookbook, suitable for a 4th grader."

response = chat.send_message(message)

timestamp = datetime.now().strftime("%Y%m%d%H%M")
output_file_1 = OUTPUT_DIR / f"photosynthesis_{timestamp}.png"

for part in response.parts:
    if part.text is not None:
        print(part.text)
    elif image:= part.as_image():
        image.save(output_file_1)
        print(f"\n画像を保存しました: {output_file_1}")

# 同じチャットを使用して、グラフィックの言語をスペイン語に変更できます

message = "Update this infographic to be in Spanish. Do not change any other elements of the image."
aspect_ratio = "16:9" # "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
resolution = "2K" # "1K", "2K", "4K"

response = chat.send_message(message,
    config=types.GenerateContentConfig(
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=resolution
        ),
    ))

timestamp_2 = datetime.now().strftime("%Y%m%d%H%M")
output_file_2 = OUTPUT_DIR / f"photosynthesis_spanish_{timestamp_2}.png"

for part in response.parts:
    if part.text is not None:
        print(part.text)
    elif image:= part.as_image():
        image.save(output_file_2)
        print(f"\n画像を保存しました: {output_file_2}")