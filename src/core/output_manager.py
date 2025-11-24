"""
Output Manager

ファイル出力とメタデータ管理を担当するモジュール。
- yyyymmddhhmmss + ユニークナンバーのファイル名生成
- 画像とプロンプト・パラメータのJSON保存
- 出力ファイルの重複防止
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib


class OutputManager:
    """出力ファイルとメタデータの管理を行うクラス"""

    def __init__(self, output_dir: str = "output"):
        """
        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_filename(
        self,
        prefix: str = "output",
        extension: str = "png",
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[Path, Path]:
        """
        ユニークなファイル名を生成する

        形式: {prefix}_{yyyymmddhhmmss}_{unique_number}.{extension}

        Args:
            prefix: ファイル名のプレフィックス
            extension: ファイル拡張子
            metadata: メタデータ（ユニーク番号の生成に使用）

        Returns:
            (image_path, metadata_path): 画像ファイルパスとメタデータJSONファイルパス
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # メタデータからユニーク番号を生成（ハッシュの最初の6文字）
        unique_number = self._generate_unique_number(timestamp, metadata)

        # ファイル名を生成
        base_filename = f"{prefix}_{timestamp}_{unique_number}"
        image_path = self.output_dir / f"{base_filename}.{extension}"
        metadata_path = self.output_dir / f"{base_filename}.json"

        # 万が一の重複を避けるため、ファイルが存在する場合はカウンタを追加
        counter = 1
        while image_path.exists() or metadata_path.exists():
            base_filename = f"{prefix}_{timestamp}_{unique_number}_{counter}"
            image_path = self.output_dir / f"{base_filename}.{extension}"
            metadata_path = self.output_dir / f"{base_filename}.json"
            counter += 1

        return image_path, metadata_path

    def _generate_unique_number(
        self,
        timestamp: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """
        タイムスタンプとメタデータからユニーク番号を生成

        Args:
            timestamp: タイムスタンプ文字列
            metadata: メタデータ辞書

        Returns:
            6桁の16進数文字列
        """
        # タイムスタンプとメタデータを結合してハッシュ化
        hash_input = timestamp
        if metadata:
            # メタデータをJSON文字列に変換してハッシュに含める
            hash_input += json.dumps(metadata, sort_keys=True)

        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        return hash_value[:6]

    def save_image_with_metadata(
        self,
        image_data: bytes,
        metadata: Dict[str, Any],
        prefix: str = "output",
        extension: str = "png"
    ) -> tuple[Path, Path]:
        """
        画像データとメタデータを保存する

        Args:
            image_data: 保存する画像のバイトデータ
            metadata: 保存するメタデータ（プロンプト、パラメータなど）
            prefix: ファイル名のプレフィックス
            extension: 画像ファイルの拡張子

        Returns:
            (image_path, metadata_path): 保存した画像パスとメタデータパス
        """
        # ファイル名を生成
        image_path, metadata_path = self.generate_filename(
            prefix=prefix,
            extension=extension,
            metadata=metadata
        )

        # 画像を保存
        with open(image_path, "wb") as f:
            f.write(image_data)

        # メタデータを保存（タイムスタンプも追加）
        metadata_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "image_file": image_path.name,
            **metadata
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_with_timestamp, f, ensure_ascii=False, indent=2)

        return image_path, metadata_path

    def save_images_with_metadata(
        self,
        image_data_list: list[bytes],
        metadata: Dict[str, Any],
        prefix: str = "output",
        extension: str = "png"
    ) -> tuple[list[Path], Path]:
        """
        複数の画像データとメタデータを保存する

        Args:
            image_data_list: 保存する画像のバイトデータのリスト
            metadata: 保存するメタデータ（プロンプト、パラメータなど）
            prefix: ファイル名のプレフィックス
            extension: 画像ファイルの拡張子

        Returns:
            (image_paths, metadata_path): 保存した画像パスのリストとメタデータパス
        """
        if not image_data_list:
            raise ValueError("image_data_list is empty")

        # 単一画像の場合は既存メソッドを使用
        if len(image_data_list) == 1:
            image_path, metadata_path = self.save_image_with_metadata(
                image_data_list[0], metadata, prefix, extension
            )
            return [image_path], metadata_path

        # 複数画像の場合
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_number = self._generate_unique_number(timestamp, metadata)

        image_paths = []
        image_filenames = []

        for idx, image_data in enumerate(image_data_list):
            # ファイル名を生成（インデックス付き）
            base_filename = f"{prefix}_{timestamp}_{unique_number}_{idx}"
            image_path = self.output_dir / f"{base_filename}.{extension}"

            # 万が一の重複を避けるため、ファイルが存在する場合はカウンタを追加
            counter = 1
            while image_path.exists():
                base_filename = f"{prefix}_{timestamp}_{unique_number}_{idx}_{counter}"
                image_path = self.output_dir / f"{base_filename}.{extension}"
                counter += 1

            # 画像を保存
            with open(image_path, "wb") as f:
                f.write(image_data)

            image_paths.append(image_path)
            image_filenames.append(image_path.name)

        # メタデータを保存
        metadata_filename = f"{prefix}_{timestamp}_{unique_number}.json"
        metadata_path = self.output_dir / metadata_filename

        metadata_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "image_files": image_filenames,
            **metadata
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_with_timestamp, f, ensure_ascii=False, indent=2)

        return image_paths, metadata_path

    def load_metadata(self, metadata_path: Path) -> Dict[str, Any]:
        """
        メタデータJSONファイルを読み込む

        Args:
            metadata_path: メタデータファイルのパス

        Returns:
            メタデータ辞書
        """
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_outputs(self, pattern: str = "*.json") -> list[Path]:
        """
        出力ディレクトリ内のメタデータファイルを一覧表示

        Args:
            pattern: 検索パターン（デフォルト: "*.json"）

        Returns:
            メタデータファイルパスのリスト（新しい順）
        """
        files = list(self.output_dir.glob(pattern))
        # 更新日時でソート（新しい順）
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files
