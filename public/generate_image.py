#!/usr/bin/env python3
"""
画像生成CLIのエントリーポイント

使用例:
    python generate_image.py "A beautiful sunset over mountains"
    python generate_image.py "A cute cat on a sofa" -m imagen-3.0-generate-002
    python generate_image.py "A futuristic city" --aspect-ratio 16:9 --prefix city
"""

from src.cli.image_gen_cli import main

if __name__ == "__main__":
    main()
