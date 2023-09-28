import os
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.models.image import ImageCategory, ResourceOrigin


@invocation(
    "I2AA_Image",
    title="I2AA Image",
    tags=["image", "ascii art"],
    category="image",
    version="0.1.0",
)
class ImageToAAInvocation(BaseInvocation):
    """Convert an Image to Ascii Art Image using any font or size"""

    input_image: ImageField = InputField(description="Image to convert to ASCII art")
    font_name: str = InputField(default="courier.ttf", description="Name of the font to use")
    font_size: int = InputField(default=6, description="Font size for the ASCII art characters")
    comparison_type: Literal["SAD", "MSE", "SSIM"] = InputField(
        default="MSE",
        description="Choose the comparison type (Sum of Absolute Differences (SAD), Mean Squared Error (MSE), Structural Similarity Index (SSIM))",
    )
    color_mode: bool = InputField(default=False, description="Enable color mode (default: grayscale)")

    def get_font_chars(self, font_path, font_size):
        chars = [
            chr(i) for i in range(0, 256)
        ]  # This includes High ASCII characters (maybe this should be a parameter passed from the node)
        font = ImageFont.truetype(font_path, font_size)
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
        self, input_image: Image.Image, font_path: str, font_size: int, color_mode: bool, comparison_method="mse",
    ):
        l_image = input_image.convert("L")  # grayscale for comparison
        c_image = input_image.convert("RGB")  # full color for average check
        char_images = self.get_font_chars(font_path, font_size)  # get the char images for comparison
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


@invocation(
    "Image_to_ASCII_Art_Image",
    title="Image to ASCII Art Image",
    tags=["image", "ascii art"],
    category="image",
    version="0.6.0",
)
class ImageToDetailedASCIIArtInvocation(BaseInvocation):
    """Convert an Image to Ascii Art Image"""

    input_image: ImageField = InputField(description="Input image to convert to ASCII art")
    font_spacing: int = InputField(default=6, description="Font size for the ASCII art characters")
    ascii_set: Literal["High Detail", "Medium Detail", "Low Detail", "Other", "Blocks", "Binary"] = InputField(
        default="Medium Detail", description="Choose the desired ASCII character set"
    )
    color_mode: bool = InputField(default=False, description="Enable color mode (default: grayscale)")
    output_to_file: bool = InputField(default=False, description="Output ASCII art to a text file")

    def get_ascii_chars(self):
        sets = {
            "High Detail": "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\\^`'. ",
            "Medium Detail": "@%#*+=-:. ",
            "Low Detail": "@#=-. ",
            "Other": " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"[
                ::-1
            ],
            "Blocks": "[]|-",
            "Binary": "01",
        }
        return sets[self.ascii_set]

    def image_to_detailed_ascii_art(self, input_image: Image.Image, font_spacing: int, color_mode: bool) -> Image.Image:
        ascii_chars = self.get_ascii_chars()

        if color_mode:
            ascii_art_image = Image.new("RGB", input_image.size, (255, 255, 255))
        else:
            ascii_art_image = Image.new("L", input_image.size, 255)

        draw = ImageDraw.Draw(ascii_art_image)

        num_cols = input_image.width // font_spacing
        num_rows = input_image.height // font_spacing

        for y in range(num_rows):
            for x in range(num_cols):
                pixel_value = input_image.getpixel((x * font_spacing, y * font_spacing))
                if isinstance(pixel_value, tuple):
                    pixel_value = pixel_value[0]

                pixel_value = max(0, min(pixel_value, 255))

                if self.ascii_set == "Binary":
                    ascii_index = 0 if pixel_value < 127.5 else 1
                else:
                    ascii_index = int(pixel_value * (len(ascii_chars) - 1) / 255)

                ascii_char = ascii_chars[ascii_index]

                if color_mode:
                    color = input_image.getpixel((x * font_spacing, y * font_spacing))
                    draw.text((x * font_spacing, y * font_spacing), ascii_char, fill=color)
                else:
                    draw.text((x * font_spacing, y * font_spacing), ascii_char, fill=0)

        return ascii_art_image

    def image_to_ascii_string(self, input_image: Image.Image, font_spacing: int) -> str:
        ascii_chars = self.get_ascii_chars()

        ascii_str = ""
        font_aspect_ratio = 2

        num_cols = input_image.width // font_spacing
        num_rows = input_image.height // (font_spacing * font_aspect_ratio)

        for y in range(num_rows):
            for x in range(num_cols):
                pixel_value = input_image.getpixel((x * font_spacing, y * font_spacing * font_aspect_ratio))
                if isinstance(pixel_value, tuple):
                    pixel_value = pixel_value[0]
                pixel_value = max(0, min(pixel_value, 255))
                ascii_index = int(pixel_value * (len(ascii_chars) - 1) / 255)
                ascii_char = ascii_chars[ascii_index]
                ascii_str += ascii_char
            ascii_str += "\n"

        return ascii_str

    def get_next_filename(self, base_filename="output.txt"):
        counter = 0
        new_filename = base_filename

        while os.path.exists(os.path.join("asciiart_output", new_filename)):
            counter += 1
            new_filename = f"output_{counter}.txt"

        return new_filename

    def ensure_directory_exists(self, directory_name="asciiart_output"):
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        input_image = context.services.images.get_pil_image(self.input_image.image_name)

        if self.output_to_file:
            ascii_str = self.image_to_ascii_string(input_image, self.font_spacing)

            self.ensure_directory_exists()
            filename = os.path.join("asciiart_output", self.get_next_filename())

            with open(filename, "w") as f:
                f.write(ascii_str)

        detailed_ascii_art_image = self.image_to_detailed_ascii_art(input_image, self.font_spacing, self.color_mode)
        mask_dto = context.services.images.create(
            image=detailed_ascii_art_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=mask_dto.image_name),
            width=mask_dto.width,
            height=mask_dto.height,
        )
