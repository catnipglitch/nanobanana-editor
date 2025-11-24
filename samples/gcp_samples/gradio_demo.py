"""
Gradio を用いたシンプルなデモアプリケーション。

テキストと画像ファイルを入力として受け取り、
テキストを反転し、画像を左右反転して表示するサンプルです。

実行例:
- python samples/gcp_samples/gradio_demo.py
"""

import gradio as gr
from PIL import Image


def process_inputs(text: str, image: Image.Image):
    """
    テキストを反転し、画像を左右反転する

    Args:
        text: 入力テキスト
        image: 入力画像

    Returns:
        reversed_text: 反転されたテキスト
        flipped_image: 左右反転された画像
    """
    # テキストを反転
    reversed_text = text[::-1] if text else ""

    # 画像を左右反転
    flipped_image = None
    if image is not None:
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    return reversed_text, flipped_image


# Gradioインターフェースの作成
demo = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Textbox(label="テキスト入力", placeholder="テキストを入力してください"),
        gr.Image(label="画像入力", type="pil")
    ],
    outputs=[
        gr.Textbox(label="反転されたテキスト"),
        gr.Image(label="左右反転された画像")
    ],
    title="シンプルなGradioデモ",
    description="テキストを反転し、画像を左右反転するシンプルなアプリケーション"
)


if __name__ == "__main__":
    # ローカルで起動
    demo.launch(server_name="0.0.0.0", server_port=7860)
