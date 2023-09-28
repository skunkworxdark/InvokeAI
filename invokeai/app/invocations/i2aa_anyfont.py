from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.models.image import ImageCategory, ResourceOrigin

import string


COMPARISON_TYPES = Literal[
    "SAD",
    "MSE",
    "SSIM",
]

COMPARISON_TYPE_LABELS = dict(
    SAD="Sum of Absolute Differences (SAD)",
    MSE="Mean Squared Error (MSE)",
    SSIM="Structural Similarity (SSIM)",
)

CHAR_RANGES = Literal[
    "All",
    "Low",
    "High",
    "Ascii",
    "Numbers",
    "Letters",
    "Lowercase",
    "Uppercase",
    "Hex",
    "Punctuation",
    "Printable",
]

CHAR_RANGE_LABELS = dict(
    All="ALL chars 0-255",
    Low="Low chars 32-127",
    High="High chars 128-255",
    Ascii="ASCII chars 32-255",
    Numbers="Numbers: 0-9 + .",
    Letters="Letters: a-z + A-Z + space",
    Lowercase="Lowercase: a-z + space",
    Uppercase="Uppercase: A-Z + space",
    Hex="Hex: 0-9 + a-f + A-F",
    Punctuation="All Punctuation",
    Printable="Letters + Numbers + Punctuation + space",
)

CHAR_SETS = {
    "All": [chr(i) for i in range(0, 255)],
    "Low": [chr(i) for i in range(32, 127)],
    "High": [chr(i) for i in range(128, 255)],
    "Ascii": [chr(i) for i in range(32, 255)],
    "Numbers": string.digits + ".",
    "Letters": string.ascii_letters + " ",
    "Lowercase": string.ascii_lowercase + " ",
    "Uppercase": string.ascii_uppercase + " ",
    "HEX": string.hexdigits,
    "Punctuation": string.punctuation,
    "Printable": string.digits + string.ascii_letters + string.punctuation + " "
}

@invocation(
    "I2AA_AnyFont",
    title="Image to ASCII Art AnyFont",
    tags=["image", "ascii art"],
    category="image",
    version="0.1.0",
)
class ImageToAAInvocation(BaseInvocation):
    """Convert an Image to Ascii Art Image using any font or size
    https://github.com/dernyn/256/tree/master this is a great font to use"""

    input_image: ImageField = InputField(description="Image to convert to ASCII art")
    font_path: str = InputField(default="cour.ttf", description="Name of the font to use")
    font_size: int = InputField(default=6, description="Font size for the ASCII art characters")
    character_range: CHAR_RANGES = InputField(
        default="Ascii", description="The character range to use",
        ui_choice_labels=CHAR_RANGE_LABELS,

    )
    comparison_type: COMPARISON_TYPES = InputField(
        default="MSE",
        description="Choose the comparison type (Sum of Absolute Differences (SAD), Mean Squared Error (MSE), Structural Similarity Index (SSIM))",
        ui_choice_labels=COMPARISON_TYPE_LABELS,
    )
    color_mode: bool = InputField(default=False, description="Enable color mode (default: grayscale)")

    def get_font_chars(self, font_path, font_size, char_range):
        font = ImageFont.truetype(font_path, font_size)
        chars = CHAR_SETS.get(char_range, [])
        char_images = {c: Image.new("L", (font_size, font_size)) for c in chars}
        for c, img in char_images.items():
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), c, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((font_size - w) / 2, (font_size - h) / 2), c, font=font, fill=255)
        return {c: np.array(img) for c, img in char_images.items()}

    def sad(self, img1, img2):
        return np.sum(np.abs(img1 - img2))

    def mse(self, img1, img2):
        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        err /= float(img1.shape[0] * img1.shape[1])
        return err

    def ssim(self, img1, img2):
        mu_img1 = np.mean(img1)
        mu_img2 = np.mean(img2)
        sigma_img1 = np.var(img1)
        sigma_img2 = np.var(img2)
        sigma_img12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
        k1, k2, L = 0.01, 0.03, 255
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        ssim = ((2 * mu_img1 * mu_img2 + C1) * (2 * sigma_img12 + C2)) / (
            (mu_img1**2 + mu_img2**2 + C1) * (sigma_img1 + sigma_img2 + C2)
        )
        return ssim

    def convert_image_to_mosaic_weighted(
        self,
        input_image: Image.Image,
        font_path: str,
        font_size: int,
        color_mode: bool,
        comparison_method="mse",
        char_range: str = "32-255",
    ):
        l_image = input_image.convert("L")  # grayscale for comparison
        c_image = input_image.convert("RGB")  # full color for average check

        char_images = self.get_font_chars(font_path, font_size, char_range)  # get the char images for comparison
        mosaic_img = Image.new("RGB" if color_mode else "L", input_image.size)  # create a color or grayscale output

        draw = ImageDraw.Draw(mosaic_img)
        for i in range(0, l_image.width, font_size):
            for j in range(0, l_image.height, font_size):
                box = (i, j, i + font_size, j + font_size)
                l_region = l_image.crop(box)
                l_region_array = np.array(l_region.resize((font_size, font_size)))

                # Calculate which char is the closest matching using selected method
                if comparison_method == "SAD":  # Sum of Absolute Differences (SAD).
                    comparisons = {c: self.sad(l_region_array, char_img) for c, char_img in char_images.items()}
                elif comparison_method == "MSE":  # Mean Squared Error (MSE)
                    comparisons = {c: self.mse(l_region_array, char_img) for c, char_img in char_images.items()}
                elif comparison_method == "SSIM":  # Structural Similarity (SSIM)
                    comparisons = {c: self.ssim(l_region_array, char_img) for c, char_img in char_images.items()}

                # Pick the char with the smallest difference.
                best_char = min(comparisons, key=comparisons.get)
                if color_mode:
                    c_region = c_image.crop(box)
                    c_region_array = np.array(c_region.resize((font_size, font_size)))
                    avg_color = tuple(np.mean(c_region_array, axis=(0, 1)).astype(int))
                else:
                    avg_color = 255

                # Draw the character image on the mosaic image.
                draw.text((i, j), best_char, font=ImageFont.truetype(font_path, font_size), fill=avg_color)
        # Save the mosaic image.
        return mosaic_img

    def invoke(self, context: InvocationContext) -> ImageOutput:
        input_image = context.services.images.get_pil_image(self.input_image.image_name)

        detailed_ascii_art_image = self.convert_image_to_mosaic_weighted(
            input_image,
            self.font_path,
            self.font_size,
            self.color_mode,
            self.comparison_type,
            self.character_range,
        )
        image_dto = context.services.images.create(
            image=detailed_ascii_art_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
