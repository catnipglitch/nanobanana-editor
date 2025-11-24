
# Sample code and API for Nano Banana Pro (Gemini 3 Pro Image Preview)

https://openrouter.ai/google/gemini-3-pro-image-preview/api

```python
import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "google/gemini-3-pro-image-preview",
    "messages": [
        {
          "role": "user",
          "content": "Generate a beautiful sunset over mountains"
        }
      ],
    "modalities": ["image", "text"]
  })
)

result = response.json()

# The generated image will be in the assistant message
if result.get("choices"):
  message = result["choices"][0]["message"]
  if message.get("images"):
    for image in message["images"]:
      image_url = image["image_url"]["url"]  # Base64 data URL
      print(f"Generated image: {image_url[:50]}...")
```

---

## 既知の問題: 4K解像度が出力されない

### 問題の詳細

**日時**: 2025-11-23
**モデル**: `google/gemini-3-pro-image-preview`
**症状**: `image_config`パラメータで`"image_size": "4K"`を指定しても、実際には低解像度（1376x768）の画像が生成される

### 検証結果

#### 試したアプローチ

1. **アプローチA: トップレベルに`image_config`を配置**
   ```json
   {
     "model": "google/gemini-3-pro-image-preview",
     "messages": [...],
     "modalities": ["image", "text"],
     "image_config": {
       "aspect_ratio": "16:9",
       "image_size": "4K"
     }
   }
   ```
   **結果**: ✗ 失敗（1376x768で生成される）

2. **アプローチB: `generation_config`でネスト**
   ```json
   {
     "model": "google/gemini-3-pro-image-preview",
     "messages": [...],
     "modalities": ["image", "text"],
     "generation_config": {
       "image_config": {
         "aspect_ratio": "16:9",
         "image_size": "4K"
       }
     }
   }
   ```
   **結果**: （検証中）

### 原因の分析

1. **OpenRouterの公式サンプルコードに`image_config`パラメータが含まれていない**
   - 公式サンプルは`modalities`のみ指定
   - 解像度やアスペクト比の指定方法が文書化されていない

2. **OpenRouterのプロキシ層の問題の可能性**
   - Google公式APIでは`types.ImageConfig(aspect_ratio, image_size)`で正常動作
   - OpenRouter経由では同じパラメータ構造が機能しない可能性

3. **ドキュメントと実装のギャップ**
   - モデルページには「2K/4K outputs and flexible aspect ratios」と記載
   - 実際のAPI実装が追いついていない可能性

### 回避策

現時点では**回避策なし**。以下の対応が考えられる：

1. OpenRouterのサポートに問い合わせ
2. Google公式Gemini API（直接）を使用（こちらは4K出力確認済み）
3. OpenRouterのドキュメント更新を待つ

### 参考情報

- **Google公式API**: `genai.Client()`経由で4K出力成功
  - ファイル: `samples/gemini_api_sample/nanobanana_simple_sample.py`
  - パラメータ: `types.ImageConfig(aspect_ratio="16:9", image_size="4K")`
  - 結果: 正常に4K解像度（3840x2160相当）で生成

- **OpenRouter API**: 4K出力失敗
  - ファイル: `samples/open_router_sample/nanobanana_openrouter_simple_sample.py`
  - 試行したパラメータ: 上記アプローチA/B参照
  - 実際の出力: 1376x768（約1K相当）

### 次のステップ

- [ ] アプローチBの検証結果を確認
- [ ] アプローチCを試行
- [ ] OpenRouterのサポートに問い合わせ
- [ ] OpenRouterのドキュメント更新を監視
