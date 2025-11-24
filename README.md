---
title: nanobanana-editor
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---


# nanobanana-editor

Google Gen AI SDK for Python を使用した画像生成ツール

## 概要

GeminiとImagenモデルによるAI画像生成ツールです。ブラウザベースの直感的なUIで、テキストや参照画像から画像を生成できます。

## インストール

### 必要要件

- Python 3.13+
- Google API キー（Gemini Developer API）

### セットアップ手順

#### 1. uvのインストール

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. 依存関係のインストール

```bash
uv sync
```

#### 3. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成：

```bash
GOOGLE_API_KEY=your-api-key
```

詳細は `.env.example` を参照してください。

## 起動方法

### Gradio UI（推奨）

```bash
# 仮想環境を有効化
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 起動
python app.py

# または、テストモード（APIキー不要）
python app.py --test
```

起動後、ブラウザで `http://localhost:7860` にアクセスします。

## 機能一覧

### Tab 1: 画像生成（Gemini）
Geminiモデルによる高速な画像生成。プロンプトから1枚ずつ生成します。

### Tab 2: 画像生成（Imagen）
Imagenモデルによる高品質な画像生成。複数枚の一括生成や詳細なパラメータ調整が可能です。

### Tab 3: 参照画像ベース生成
最大14枚の参照画像とプロンプトから新しい画像を生成。Gemini 3 Pro Image Previewを使用します。

### Tab 4: Settings（設定）
APIキーの管理と接続テスト。アプリ起動時にAPIキー未設定の場合は自動的にこのタブが表示されます。

## 対応モデル

### Gemini
- `gemini-2.5-flash-image` - 高速生成
- `gemini-3-pro-image-preview` - 参照画像ベース生成

### Imagen
- `imagen-4.0-generate-001` - 最新高品質
- `imagen-4.0-fast-generate-001` - 最新高速
- `imagen-3.0-generate-002` - 推奨
- `imagen-3.0-generate-001`
- `imagen-3.0-fast-generate-001` - 高速

### Test Model
- `test-model` - APIキー不要のテストモデル（開発・デバッグ用）

## 出力ファイル

生成されたファイルは `output/` ディレクトリに保存されます。

### ファイル命名規則

**単一画像:**
```
{prefix}_{YYYYMMDDHHMMSS}_{unique_id}.png
{prefix}_{YYYYMMDDHHMMSS}_{unique_id}.json
```

**複数画像:**
```
{prefix}_{YYYYMMDDHHMMSS}_{unique_id}_0.png
{prefix}_{YYYYMMDDHHMMSS}_{unique_id}_1.png
{prefix}_{YYYYMMDDHHMMSS}_{unique_id}.json
```

### メタデータ

各画像に対応するJSONファイルには以下の情報が保存されます:
- タイムスタンプ
- 画像ファイル名
- 使用モデル
- プロンプト
- 生成パラメータ
- 認証方式

## トラブルシューティング

### APIキーエラー

```
❌ エラー: GOOGLE_API_KEYが設定されていません
```

→ `.env`ファイルに`GOOGLE_API_KEY`を設定してください

```bash
GOOGLE_API_KEY=your-api-key
```

### Test Modelを使用したい

APIキー不要で動作確認が可能です:

```bash
# CLI
python generate_image.py "Test prompt" -m test-model

# Gradio UI
# Tab 1またはTab 2からモデル選択で "Test Model" を選択
```

## 技術スタック

- **Python**: 3.13+
- **Google Gen AI SDK for Python**: 統合APIクライアント
- **Gradio**: WebUIフレームワーク
- **PIL (Pillow)**: 画像処理

## ライセンス

このプロジェクトは個人利用を想定しています。
