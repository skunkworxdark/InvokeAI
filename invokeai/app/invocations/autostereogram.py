# 2024 skunkworxdark (https://github.com/skunkworxdark)

from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.fields import InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import (
    ImageField,
    ImageOutput,
)


@invocation(
    "autostereogram",
    title="AutoStereogram",
    tags=["image"],
    category="image",
    version="1.1.0",
)
class AutostereogramInvocation(BaseInvocation, WithMetadata, WithBoard):
    """create an autostereogram from a depth map"""

    # Inputs
    depth_map: ImageField = InputField(description="The depth map to create the autostereogram from")
    pattern: Optional[ImageField] = InputField(
        default=None,
        description="The pattern image, if not provided then random dots will be used",
    )
    pattern_divisions: int = InputField(
        default=8,
        gt=1,
        description="How many pattern repeats in output 5-10 is in general a good range. lower = more depth but harder to see",
    )
    invert_depth_map: bool = InputField(
        default=False,
        description="Invert the depth map (difference between crossing and uncrossing eyes)",
    )
    grayscale: bool = InputField(
        default=False,
        description="Color or Grayscale output",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Load the depth map and ensure grayscale
        depth_map = context.images.get_pil(self.depth_map.image_name, "L")

        # Calculate the pattern width
        pattern_width = int(depth_map.width / self.pattern_divisions)

        # extend the left of the depthmap by the pattern width
        depth_map = ImageOps.expand(depth_map, border=(pattern_width, 0, 0, 0), fill=0)
        depth_width, depth_height = depth_map.size

        # Convert the depth map to a numpy array (invert if requested)
        depth_np = np.array(depth_map) ^ 0xFF if self.invert_depth_map else np.array(depth_map)

        if self.pattern:
            # Load the pattern
            pattern = context.images.get_pil(self.pattern.image_name, "RGB")

            # Resize pattern width
            pattern_height = int(pattern_width * (pattern.height / pattern.width))
            pattern = pattern.resize((pattern_width, pattern_height))
            pattern_np = np.array(pattern)
        else:
            pattern_height = depth_height
            pattern_np = np.random.randint(0, 256, (depth_height, pattern_width, 3), dtype=np.uint8)

        # Create an output image
        output_image = Image.new("RGB", (depth_width, depth_height))
        pixels = output_image.load()

        # Generate the autostereogram
        for y in range(depth_height):
            for x in range(depth_width):
                if x < pattern_width:
                    # unaltered pattern
                    r, g, b = pattern_np[y % pattern_height, x % pattern_width]
                else:
                    # shifted pattern based upon depth
                    shift = depth_np[y, x] / 255 * pattern_width * 0.4
                    r, g, b = pixels[int(x - pattern_width + shift), y]

                pixels[x, y] = (int(r), int(g), int(b))

        # Cut off the left hand pattern section
        output_image = output_image.crop((pattern_width, 0, depth_width, depth_height))

        if self.grayscale:
            output_image = output_image.convert("L")

        # Save the image
        image_dto = context.images.save(image=output_image)

        return ImageOutput.build(image_dto)


@invocation(
    "adv_autostereogram",
    title="Adv AutoStereogram",
    tags=["image"],
    category="image",
    version="1.1.0",
)
class AdvAutostereogramInvocation(BaseInvocation, WithMetadata, WithBoard):
    """create an advanced autostereogram from a depth map"""

    # Inputs
    depth_map: ImageField = InputField(description="The depth map to create the image from")
    pattern: Optional[ImageField] = InputField(
        default=None,
        description="The pattern image to use as the background, if not provided then random dots will be used",
    )
    pattern_width: int = InputField(
        default=100,
        gt=20,
        description="The pattern width pixels",
    )
    depth_steps: int = InputField(
        default=50,
        gt=10,
        description="The number of depth steps, 30-127 is a good range but should be less than the pattern width",
    )
    invert_depth_map: bool = InputField(
        default=False,
        description="Invert the depth map (difference between crossing and uncrossing eyes)",
    )
    grayscale: bool = InputField(
        default=False,
        description="Color or Grayscale output",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Load the depth map
        depth_map = context.images.get_pil(self.depth_map.image_name, "L")

        # extend the left of the depthmap by the separation value
        depth_map = ImageOps.expand(depth_map, border=(self.pattern_width, 0, 0, 0), fill=0)
        depth_width, depth_height = depth_map.size

        # Convert the depth map to a numpy array (invert if requested)
        depth_np = np.array(depth_map) ^ 0xFF if self.invert_depth_map else np.array(depth_map)

        pattern_width = self.pattern_width
        if self.pattern:
            # Load the pattern
            pattern = context.images.get_pil(self.pattern.image_name, "RGB")

            # Resize pattern width to separation size
            pattern_height = int(pattern_width * (pattern.height / pattern.width))
            pattern = pattern.resize((pattern_width, pattern_height))
            pattern_np = np.array(pattern)
        else:
            pattern_height = depth_height
            pattern_np = np.random.randint(0, 256, (depth_height, pattern_width, 3), dtype=np.uint8)

        # Create an output image
        output_image = Image.new("RGB", (depth_width, depth_height))
        pixels = output_image.load()

        # Generate the autostereogram
        for y in range(depth_height):
            for x in range(depth_width):
                if x < self.pattern_width:
                    # unaltered pattern
                    r, g, b = pattern_np[y % pattern_height, x % pattern_width]
                else:
                    # shifted pattern based upon depth
                    shift = depth_np[y, x] / 255.0 * self.depth_steps
                    r, g, b = pixels[int(x - self.pattern_width + shift), y]

                pixels[x, y] = (int(r), int(g), int(b))

        # Cut off the left hand pattern section
        output_image = output_image.crop((pattern_width, 0, depth_width, depth_height))

        if self.grayscale:
            output_image = output_image.convert("L")

        # Save the image
        image_dto = context.images.save(image=output_image)

        return ImageOutput.build(image_dto)
