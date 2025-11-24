# CLI (Maintenance Mode)

This CLI is in **maintenance mode**. No new features will be added.

## Status

- **Maintenance only**: Critical bug fixes only
- **No new features**: All new development should focus on the Gradio UI (`src/ui/`)
- **Stable**: The CLI works as intended and requires no changes

## For AI Agents

AI agents should **NOT** modify files in this directory unless:
- Fixing a critical security vulnerability
- Fixing a bug that prevents the CLI from functioning
- Updating imports due to changes in `src/core/`

For all new image generation features, use the Gradio UI instead.

## Usage

See the main README.md for CLI usage examples. The CLI remains fully functional:

```bash
# Gemini model
python generate_image.py "A cute cat" -m gemini-2.5-flash-image

# Imagen model
python generate_image.py "A robot" -m imagen-4.0-fast-generate-001

# Test model (no API call)
python generate_image.py "Test prompt" -m test-model
```
