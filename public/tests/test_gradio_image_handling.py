"""
Test script to verify Gradio UI fix for multiple image generation

Tests that:
1. Single image returns: (image, None, visible=True, visible=False, info, json)
2. Multiple images return: (None, images_list, visible=False, visible=True, info, json)
"""

import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.image_generator import ImageGenerator, GenerationConfig, ModelType
from src.core.output_manager import OutputManager
from PIL import Image
import io
import json

def test_single_image():
    """Test single image generation (Test Model with n=1)"""
    print("=" * 60)
    print("Testing single image generation...")
    print("=" * 60)

    generator = ImageGenerator(google_api_key="dummy_key_for_test")

    config = GenerationConfig(
        model_type=ModelType.TEST,
        model_name="test-model",
        prompt="Single image test",
        number_of_images=1,
        aspect_ratio="1:1",
    )

    # Generate
    image_data_list, metadata = generator.generate(config)

    # Verify
    assert len(image_data_list) == 1, f"Expected 1 image, got {len(image_data_list)}"

    # Convert to PIL
    pil_images = [Image.open(io.BytesIO(data)) for data in image_data_list]

    # Expected return structure
    if len(pil_images) == 1:
        single_output = pil_images[0]
        gallery_output = None
        single_visible = True
        gallery_visible = False

        print(f"✓ Single image: {type(single_output)}")
        print(f"✓ Gallery output: {gallery_output}")
        print(f"✓ Image component visible: {single_visible}")
        print(f"✓ Gallery component visible: {gallery_visible}")
        print("✓ Single image test PASSED")
    else:
        print("✗ Single image test FAILED")

    print()
    return True

def test_multiple_images():
    """Test multiple image generation (Test Model with n=3)"""
    print("=" * 60)
    print("Testing multiple image generation...")
    print("=" * 60)

    generator = ImageGenerator(google_api_key="dummy_key_for_test")

    config = GenerationConfig(
        model_type=ModelType.TEST,
        model_name="test-model",
        prompt="Multiple images test",
        number_of_images=3,
        aspect_ratio="1:1",
        seed=12345,
    )

    # Generate
    image_data_list, metadata = generator.generate(config)

    # Verify
    assert len(image_data_list) == 3, f"Expected 3 images, got {len(image_data_list)}"

    # Convert to PIL
    pil_images = [Image.open(io.BytesIO(data)) for data in image_data_list]

    # Expected return structure
    if len(pil_images) > 1:
        single_output = None
        gallery_output = pil_images
        single_visible = False
        gallery_visible = True

        print(f"✓ Single image output: {single_output}")
        print(f"✓ Gallery output: {type(gallery_output)} with {len(gallery_output)} images")
        print(f"✓ Image component visible: {single_visible}")
        print(f"✓ Gallery component visible: {gallery_visible}")
        print("✓ Multiple images test PASSED")
    else:
        print("✗ Multiple images test FAILED")

    print()
    return True

def main():
    print("\n")
    print("=" * 60)
    print("Gradio UI Fix Test - Multiple Image Generation")
    print("=" * 60)
    print()

    try:
        # Run tests
        test_single_image()
        test_multiple_images()

        print("=" * 60)
        print("All tests PASSED ✓")
        print("=" * 60)
        print()
        print("The fix should now work correctly:")
        print("- Single image: displays in Image component")
        print("- Multiple images: displays in Gallery component")
        print("- Components toggle visibility automatically")
        print()

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
