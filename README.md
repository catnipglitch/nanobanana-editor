---
title: nanobanana-editor
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: 3.11
pinned: false
---


# nanobanana-editor

Google Gen AI SDK for Python を使用した画像生成ツール

## 概要

GeminiとImagenモデルによるAI画像生成ツールです。ブラウザベースの直感的なUIで、テキストや参照画像から画像を生成できます。

## インストール

### 必要要件

- Python 3.11以上（ローカル開発では3.13推奨、Hugging Face Spacesでは3.11）
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

## 機能一覧（11タブ構成）

### Tab 1: 画像生成（Gemini）✅
Geminiモデルによる高速な画像生成。プロンプトから1枚ずつ生成。Google検索ツール統合済み。

### Tab 2: ベーシック編集 🚧
基本的な画像編集機能。回転・反転・リサイズ・アルファチャンネル抽出など。

### Tab 3: 参照画像ベース生成 ✅
最大14枚の参照画像とプロンプトから新しい画像を生成。Gemini 3 Pro Image Previewを使用。

### Tab 4: マルチターン編集 🚧
対話形式での段階的な画像編集。編集履歴の保持と任意のステップへの復帰が可能。

### Tab 5: レイアウト編集 🚧
キャラクターシートなど定型レイアウトでの画像配置。

### Tab 6: 衣装チェンジ 🚧
キャラクター着替え専用機能。マスク処理による部分的な画像変更。

### Tab 7: 高度な編集 🚧
アルファチャンネル処理・高度なマスク編集など。

### Tab 8: 画像解析 🚧
画像説明生成・ポーズ抽出・オブジェクト検出。Gemini Visionを使用。

### Tab 9: Chat / エージェント ✅
対話型エージェントによる編集支援。

### Tab 10: 画像生成（Imagen）✅
Imagenモデルによる高品質な画像生成。複数枚の一括生成や詳細なパラメータ調整が可能。

### Tab 11: Settings ✅
APIキーの管理と接続テスト。アプリ起動時にAPIキー未設定の場合は自動的にこのタブが表示されます。

> **凡例**: ✅ 実装済み / 🚧 Coming Soon

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

### ローカル環境

生成されたファイルは `output/` ディレクトリに保存されます。

**ファイル命名規則（単一画像）:**
```
{prefix}_{YYYYMMDDHHMMSS}_{unique_id}.png
{prefix}_{YYYYMMDDHHMMSS}_{unique_id}.json
```

**ファイル命名規則（複数画像）:**
```
{prefix}_{YYYYMMDDHHMMSS}_{unique_id}_0.png
{prefix}_{YYYYMMDDHHMMSS}_{unique_id}_1.png
{prefix}_{YYYYMMDDHHMMSS}_{unique_id}.json
```

**メタデータ:**

各画像に対応するJSONファイルには以下の情報が保存されます:
- タイムスタンプ
- 画像ファイル名
- 使用モデル
- プロンプト
- 生成パラメータ
- 認証方式

### Hugging Face Spaces

クラウド環境では、ストレージがephemeral（一時的）なため、ファイル保存は自動的に無効化されます。

- 画像はGradioの一時ファイル機構で管理
- UIに「保存無効」メッセージを表示
- メタデータはJSON表示コンポーネントで確認可能
- ユーザーは表示された画像を個別にダウンロード可能

**環境検出:** `SPACE_ID` 環境変数の存在により自動判定されます。

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

- **Python**: 3.11以上（ローカル: 3.13推奨、HF Spaces: 3.11）
- **Google Gen AI SDK for Python**: 統合APIクライアント
- **Gradio**: WebUIフレームワーク
- **PIL (Pillow)**: 画像処理

## デプロイ

### Hugging Face Spaces

このアプリは Hugging Face Spaces へのデプロイに対応しています。

**デプロイ手順:**

1. Hugging Face にログイン
2. 新しい Space を作成（SDK: Gradio）
3. リポジトリをプッシュ
4. Secrets に `GOOGLE_API_KEY` を設定

**自動対応:**
- ファイル保存の無効化（`SPACE_ID` 環境変数を検出）
- Gradioの一時ファイル機構を使用
- メタデータはUIに直接表示

**手動制御:**

環境変数 `DISABLE_FILE_SAVE=true` を設定することで、任意の環境でファイル保存を無効化できます。

詳細は `.private/docs/TECH_SPEC.md` の「ファイル出力管理とデプロイ対応」セクションを参照してください。

## ライセンス

このプロジェクトは個人利用を想定しています。
