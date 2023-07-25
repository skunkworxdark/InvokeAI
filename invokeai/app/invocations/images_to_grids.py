# Copyright (c) 2023 skunkworxdark (https://github.com/skunkworxdark)

from typing import Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field

from ..models.image import ImageCategory, ImageField, ResourceOrigin, ColorField
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext, InvocationConfig
from .image import PILInvocationConfig, PIL_RESAMPLING_MODES ,PIL_RESAMPLING_MAP

class ImagesToGridsOutput(BaseInvocationOutput):
    """Base class for invocations that output nothing"""

    # fmt: off
    type: Literal["image_grid_output"] = "image_grid_output"

    # Outputs
    collection: list[ImageField] = Field(default=[], description="The output images")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "collection"]}


class ImagesToGridsInvocation(BaseInvocation, PILInvocationConfig):
    """Load a collection of images and provide it as output."""

    # fmt: off
    type: Literal["image_grid"] = "image_grid"

    # Inputs
    images: list[ImageField] = Field(default=[], description="The image collection to turn into grids")
    columns: int = Field(default=1, ge=1, description="The number of columns in each grid")
    rows: int = Field(default=1, ge=1, description="The nuber of rows to have in each grid")
    space: int = Field(default=1, ge=0, description="The space to be added between images")
    scale_factor: Optional[float] = Field(default=1.0, gt=0, description="The factor by which to scale the images")
    resample_mode:  PIL_RESAMPLING_MODES = Field(default="bicubic", description="The resampling mode")
    background_color: ColorField = Field(
        default=ColorField(r=0, g=0, b=0, a=255),
        description="The color to use as the background",
    )
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "type_hints": {
                    "title": "Images To Grids",
                    "images": "image_collection",
                }
            },
        }

    def invoke(self, context: InvocationContext) -> ImagesToGridsOutput:
        """Convert an image list into a grids of images"""
        images = [context.services.images.get_pil_image(image.image_name) for image in self.images]
        width_max = int(max([image.width for image in images]) * self.scale_factor)
        height_max = int(max([image.height for image in images]) * self.scale_factor)
        background_width = width_max * self.columns + (self.space * (self.columns - 1))
        background_height = height_max * self.rows + (self.space * (self.rows - 1))
        resample_mode = PIL_RESAMPLING_MAP[self.resample_mode]

        column = 0
        row = 0
        x_offset = 0
        y_offset = 0
        background = Image.new('RGBA', (background_width, background_height), self.background_color.tuple())
        grid_images = []

        for image in images:
            if not self.scale_factor == 1.0:
                image = image.resize(
                    (int(image.width * self.scale_factor), int(image.width * self.scale_factor)),
                    resample=resample_mode,
                )

            background.paste(image, (x_offset, y_offset))

            column += 1
            x_offset += width_max + self.space
            if column >= self.columns:
                column = 0
                x_offset = 0
                y_offset += height_max + self.space
                row += 1
            
            if row >= self.rows:
                row = 0
                y_offset = 0
                image_dto = context.services.images.create(
                    image=background,
                    image_origin=ResourceOrigin.INTERNAL,
                    image_category=ImageCategory.GENERAL,
                    node_id=self.id,
                    session_id=context.graph_execution_state_id,
                    is_intermediate=self.is_intermediate,
                )
                grid_images.append(ImageField(image_name=image_dto.image_name))
                background = Image.new('RGBA', (background_width, background_height), self.background_color.tuple())
        
        if column > 0 or row > 0:
            image_dto = context.services.images.create(
                image=background,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
            )
            grid_images.append(ImageField(image_name=image_dto.image_name))

        return ImagesToGridsOutput(collection=grid_images)
