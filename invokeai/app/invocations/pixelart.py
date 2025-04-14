# 2024 skunkworxdark (https://github.com/skunkworxdark)
# Updated 2025-04-13 (v2) to use OpenCV instead of scikit-learn for K-Means, with refined parameters

from typing import Optional

import numpy as np
from PIL import Image

# Removed: from sklearn.cluster import KMeans
import cv2  # Added: Import OpenCV

from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
)


# --- apply_dithering remains the same ---
def apply_dithering(image: Image.Image, palette: np.ndarray) -> Image.Image:
    """Applies Floyd-Steinberg dithering to an image using a given palette."""
    # Ensure image is RGB for consistent processing
    image_rgb = image.convert("RGB")
    pixels = np.array(image_rgb, dtype=np.float32)  # Work with float for error diffusion
    # Ensure palette is float32 for distance calculation and error diffusion
    palette_float32 = np.array(palette, dtype=np.float32)
    width, height = image_rgb.size

    # Helper to find the nearest color in the palette
    def get_closest_color(color: np.ndarray) -> np.ndarray:
        # Calculate distances against the float32 palette
        distances = np.sum((palette_float32 - color) ** 2, axis=1)
        return palette_float32[np.argmin(distances)]

    for y in range(height):
        for x in range(width):
            old_color = pixels[y, x].copy()  # Use copy to avoid modifying original before error calculation
            new_color = get_closest_color(old_color)
            pixels[y, x] = new_color
            error = old_color - new_color

            # Floyd-Steinberg error diffusion
            if x < width - 1:
                pixels[y, x + 1] = np.clip(pixels[y, x + 1] + error * 7 / 16, 0, 255)
            if y < height - 1:
                if x > 0:
                    pixels[y + 1, x - 1] = np.clip(pixels[y + 1, x - 1] + error * 3 / 16, 0, 255)
                pixels[y + 1, x] = np.clip(pixels[y + 1, x] + error * 5 / 16, 0, 255)
                if x < width - 1:
                    pixels[y + 1, x + 1] = np.clip(pixels[y + 1, x + 1] + error * 1 / 16, 0, 255)

    # Convert back to uint8 for PIL image
    return Image.fromarray(pixels.astype("uint8"), "RGB")


def optimize_palette(image: Image.Image, num_colors: int, iterations: int = 5) -> Optional[np.ndarray]:
    """
    Finds an optimized color palette using K-Means clustering (OpenCV)
    and selects the one minimizing dithering error.
    """
    image_rgb = image.convert("RGB")
    # Keep the original 3D pixel array for error calculation later
    original_pixels_3d = np.array(image_rgb)  # Shape (height, width, 3)

    # Reshape pixels specifically for K-Means input
    pixels_reshaped = original_pixels_3d.reshape((-1, 3))  # Shape (height * width, 3)
    pixels_float32 = pixels_reshaped.astype(np.float32)

    best_palette = None
    best_error = float("inf")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    for i in range(iterations):
        compactness, labels, centers = cv2.kmeans(
            data=pixels_float32,
            K=num_colors,
            bestLabels=None,
            criteria=criteria,
            attempts=1,
            flags=cv2.KMEANS_RANDOM_CENTERS,
        )

        # Handle case where kmeans might return fewer centers than requested (rare)
        if centers is None or centers.shape[0] < num_colors:
            print(
                f"Warning: K-Means returned only {centers.shape[0] if centers is not None else 0} centers for iteration {i + 1}."
            )
            if centers is None or centers.shape[0] == 0:
                continue

        palette = np.round(centers).astype(np.uint8)

        # Apply dithering with the generated palette
        dithered_img = apply_dithering(image_rgb, palette)
        dithered_pixels = np.array(dithered_img)  # Shape (height, width, 3)

        # Calculate error comparing the original 3D array with the dithered 3D array
        error = np.sum((original_pixels_3d.astype(np.float64) - dithered_pixels.astype(np.float64)) ** 2)

        if error < best_error:
            best_error = error
            best_palette = palette
        elif best_palette is None:
            best_palette = palette

    # Final check if a palette was ever assigned
    if best_palette is None:
        print("Warning: optimize_palette failed to find any valid palette. Running a basic K-Means as fallback.")
        # Run one basic K-Means attempt as a last resort
        compactness, labels, centers = cv2.kmeans(pixels_float32, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        best_palette = np.round(centers).astype(np.uint8)

    return best_palette


# --- reduce_colors_with_dithering remains mostly the same, ensures palette type consistency ---
def reduce_colors_with_dithering(
    image: Image.Image,
    num_colors: Optional[int] = None,
    dither: bool = True,
    custom_palette: Optional[np.ndarray] = None,
) -> Image.Image:
    """Reduces colors in an image, optionally applying dithering."""
    image_rgb = image.convert("RGB")

    final_palette: np.ndarray
    if custom_palette is not None:
        # Ensure palette is uint8 NumPy array
        final_palette = np.array(custom_palette, dtype=np.uint8)
    elif num_colors is not None:
        generated_palette = optimize_palette(image_rgb, num_colors)
        if generated_palette is None:
            # Fallback: if optimize_palette fails, generate a simple palette
            # This is a basic fallback, might need more robust handling
            print("Warning: Optimized palette generation failed. Using basic K-Means fallback.")
            pixels_float32 = np.array(image_rgb).reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            compactness, labels, centers = cv2.kmeans(
                pixels_float32, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
            generated_palette = np.round(centers).astype(np.uint8)

        final_palette = generated_palette
    else:
        raise ValueError("Either num_colors or custom_palette must be provided")

    if final_palette.size == 0 or final_palette.shape[1] != 3:
        raise ValueError(f"Invalid palette shape received: {final_palette.shape}")

    if dither:
        # apply_dithering handles conversion internally if needed
        return apply_dithering(image_rgb, final_palette)
    else:
        # No dithering: map each pixel to the closest palette color
        pixels = np.array(image_rgb)
        palette_float32 = final_palette.astype(np.float32)  # Use float for distance calculation

        # Reshape for broadcasting
        distances = np.sum(
            (pixels[:, :, np.newaxis, :].astype(np.float32) - palette_float32[np.newaxis, np.newaxis, :, :]) ** 2,
            axis=3,
        )
        indices = np.argmin(distances, axis=2)
        # Create new image using the uint8 palette
        new_pixels = final_palette[indices]
        return Image.fromarray(new_pixels, "RGB")  # Already uint8


@invocation(
    "pixel-art",
    title="Pixel Art",
    tags=["image", "pixelart", "dithering"],
    category="image",
    version="1.1.0",
)
class PixelArtInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Convert an image to pixelart"""

    # Inputs
    image: ImageField = InputField(description="The image to convert to pixelart")
    palette_image: Optional[ImageField] = InputField(
        default=None, description="Optional - Image with the palette to use"
    )
    pixel_size: int = InputField(
        default=4,
        ge=2,
        le=256,
        description="Size of the large 'pixels' in the output",
    )
    palette_size: int = InputField(
        default=16,
        ge=2,
        le=256,
        description="No. of colours in Palette (used if palette_image is not provided or to extract from it)",
    )
    dither: bool = InputField(
        default=False,
        description="Apply Floyd-Steinberg dithering",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        img = context.images.get_pil(self.image.image_name)
        img_rgb = img.convert("RGB")  # Work with RGB

        sw = max(1, img.width // self.pixel_size)
        sh = max(1, img.height // self.pixel_size)
        simg = img_rgb.resize((sw, sh), resample=Image.Resampling.BILINEAR)

        custom_palette = None
        if self.palette_image:
            pimg = context.images.get_pil(self.palette_image.image_name)
            pimg_rgb = pimg.convert("RGB")
            pixels = np.array(pimg_rgb).reshape((-1, 3))
            pixels_float32 = pixels.astype(np.float32)

            # Use stricter criteria and RANDOM centers for palette extraction from image
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  # Changed: stricter criteria
            compactness, labels, centers = cv2.kmeans(
                data=pixels_float32,
                K=self.palette_size,
                bestLabels=None,
                criteria=criteria,
                attempts=10,
                flags=cv2.KMEANS_RANDOM_CENTERS,  # Changed: Use RANDOM to mimic n_init=10
            )
            # Round before casting to uint8
            custom_palette = np.round(centers).astype(np.uint8)  # Changed: Added np.round()

        # Reduce colors of the small image
        img_processed = reduce_colors_with_dithering(
            image=simg,
            num_colors=self.palette_size if custom_palette is None else None,
            dither=self.dither,
            custom_palette=custom_palette,
        )

        # Resize back to original size using nearest neighbor
        result = img_processed.resize(img.size, Image.Resampling.NEAREST)

        # Handle alpha if original image had it
        original_mode = img.mode
        if original_mode in ["RGBA", "LA"]:
            try:
                alpha = img.getchannel("A")
                # Ensure result is compatible before putting alpha
                if result.mode == "RGB" and original_mode == "RGBA":
                    result = result.convert("RGBA")
                    result.putalpha(alpha)
                elif result.mode == "L" and original_mode == "LA":
                    result = result.convert("LA")
                    result.putalpha(alpha)
                elif result.mode == "RGB" and original_mode == "LA":  # Handle L->RGB->LA case? Less common.
                    result = result.convert("L").convert("LA")
                    # Re-applying alpha might be tricky if intermediate was RGB
                    # Maybe resize alpha channel separately and merge
                    alpha_resized = alpha.resize(result.size, Image.Resampling.NEAREST)
                    result.putalpha(alpha_resized)
                elif result.mode == original_mode:  # Already correct mode (RGBA or LA)
                    result.putalpha(alpha)

            except ValueError:
                print("Warning: Could not get or apply alpha channel.")  # Handle cases where alpha is broken

        image_dto = context.images.save(result)
        return ImageOutput.build(image_dto)
