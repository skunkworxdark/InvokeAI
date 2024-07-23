# 2024 skunkworxdark (https://github.com/skunkworxdark)

from typing import Optional

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

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


def apply_dithering(image, palette):
    width, height = image.size
    pixels = np.array(image)
    palette = np.array(palette)

    def get_closest_color(color):
        distances = np.sum((palette - color) ** 2, axis=1)
        return palette[np.argmin(distances)]

    for y in range(height):
        for x in range(width):
            old_color = pixels[y, x].astype(float)
            new_color = get_closest_color(old_color)
            pixels[y, x] = new_color
            error = old_color - new_color

            if x < width - 1:
                pixels[y, x + 1] = np.clip(pixels[y, x + 1] + error * 7 / 16, 0, 255)
            if y < height - 1:
                if x > 0:
                    pixels[y + 1, x - 1] = np.clip(pixels[y + 1, x - 1] + error * 3 / 16, 0, 255)
                pixels[y + 1, x] = np.clip(pixels[y + 1, x] + error * 5 / 16, 0, 255)
                if x < width - 1:
                    pixels[y + 1, x + 1] = np.clip(pixels[y + 1, x + 1] + error * 1 / 16, 0, 255)

    return Image.fromarray(pixels.astype("uint8"))


def optimize_palette(image, num_colors, iterations=5):
    pixels = np.array(image).reshape((-1, 3))
    best_palette = None
    best_error = float("inf")

    for _ in range(iterations):
        kmeans = KMeans(n_clusters=num_colors, n_init=1, random_state=np.random.randint(0, 1000))
        kmeans.fit(pixels)
        palette = kmeans.cluster_centers_.astype(int)

        dithered = apply_dithering(image, palette)
        error = np.sum((np.array(image) - np.array(dithered)) ** 2)

        if error < best_error:
            best_error = error
            best_palette = palette

    return best_palette


def reduce_colors_with_dithering(image, num_colors=None, dither=True, custom_palette=None):
    if custom_palette is not None:
        palette = np.array(custom_palette)
    elif num_colors is not None:
        palette = optimize_palette(image, num_colors)
    else:
        raise ValueError("Either num_colors or custom_palette must be provided")

    if dither:
        return apply_dithering(image, palette)
    else:
        return Image.fromarray(
            palette[
                np.argmin(np.sum((np.array(image)[:, :, None] - palette[None, None, :]) ** 2, axis=3), axis=2)
            ].astype("uint8")
        )


@invocation(
    "pixel-art",
    title="Pixel Art",
    tags=["image"],
    category="image",
    version="1.0.0",
)
class PixelArtInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Convert an image to pixelart"""

    # Inputs
    image: ImageField = InputField(description="The image to convert to pixelart")
    palette_image: Optional[ImageField] = InputField(description="Optional - Image with the palette to use")
    pixel_size: int = InputField(
        default=4,
        ge=2,
        le=256,
        description="size of pixels",
    )
    palette_size: int = InputField(
        default=16,
        ge=2,
        le=256,
        description="No. of colours in Palette",
    )
    dither: bool = InputField(
        default=False,
        description="Dither",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        img = context.images.get_pil(self.image.image_name, "RGB")
        sw = img.width // self.pixel_size
        sh = img.height // self.pixel_size
        simg = img.resize((sw, sh), resample=Image.Resampling.BILINEAR)

        if self.palette_image:
            pimg = context.images.get_pil(self.palette_image.image_name, "RGB")

            pixels = np.array(pimg).reshape((-1, 3))
            kmeans = KMeans(n_clusters=self.palette_size, n_init=10, random_state=42)
            kmeans.fit(pixels)
            custom_palette = kmeans.cluster_centers_.astype(int)
        else:
            custom_palette = None

        img_processed = reduce_colors_with_dithering(simg, self.palette_size, self.dither, custom_palette)

        result = img_processed.resize(img.size, Image.Resampling.NEAREST)

        image_dto = context.images.save(result)
        return ImageOutput.build(image_dto)
