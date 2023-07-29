# Copyright (c) 2023 skunkworxdark (https://github.com/skunkworxdark)

from typing import Literal, Optional, Union, List, Any

from PIL import Image
from pydantic import BaseModel, Field
from itertools import product
import json

from ..models.image import ImageCategory, ImageField, ResourceOrigin, ColorField
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext, InvocationConfig
from .image import PILInvocationConfig, PIL_RESAMPLING_MODES ,PIL_RESAMPLING_MAP


class FloatsToStringsOutput(BaseInvocationOutput):
    """class for FloatToStringOutput"""
    type: Literal["float_to_string_output"] = "float_to_string_output"

    float_string: list[str] = Field(default=[], description="collection of strings")

    class Config:
        schema_extra = {"required": ["type", "float_string"]}
    
class FloastToStringsInvocation(BaseInvocation):
    """class for FloatToString converts a float or collections of floats to a collection of strings"""
    type: Literal["float_to_string"] = "float_to_string"

    floats: Union[float, list[float], None] = Field(default=None, description="float or collection of floats")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Floats To Strings",
                "type_hints": {
                    "floats": "float",
                }
            },
        }

    def invoke(self, context: InvocationContext) -> FloatsToStringsOutput:
        """Invoke with provided services and return outputs."""
        if self.floats is None:
            return None
        if isinstance(self.floats, list):
            return FloatsToStringsOutput(float_string=[str(x) for x in self.floats])
        else:
            return FloatsToStringsOutput(float_string=[str(self.floats)])


class StringToFloatOutput(BaseInvocationOutput):
    """class for StringToFloatOutput"""
    type: Literal["string_to_float_output"] = "string_to_float_output"

    floats: float = Field(default=1.0, description="float")

    class Config:
        schema_extra = {"required": ["type", "floats"]}
    
class StringToFloatInvocation(BaseInvocation):
    """class for StringToFloat converts a string to a float"""
    type: Literal["string_to_float"] = "string_to_float"

    float_string: str = Field(default='', description="string")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "String To Float",
                "type_hints": {
                    "floats": "float",
                }
            },
        }

    def invoke(self, context: InvocationContext) -> StringToFloatOutput:
        """Invoke with provided services and return outputs."""
        return StringToFloatOutput(floats=float(self.float_string))


class IntsToStringsOutput(BaseInvocationOutput):
    """class for IntToStringOutput"""
    type: Literal["int_to_string_output"] = "int_to_string_output"

    int_string: list[str] = Field(default=[], description="collection of strings")

    class Config:
        schema_extra = {"required": ["type", "int_string"]}
    
class IntsToStringsInvocation(BaseInvocation):
    """class for IntToString converts an int or collection of ints to a collection of strings"""
    type: Literal["int_to_string"] = "int_to_string"

    ints: Union[int, list[int], None] = Field(default=None, description="int or collection of ints")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Ints To Strings",
                "type_hints": {
                    "ints": "integer",
                }
            },
        }

    def invoke(self, context: InvocationContext) -> IntsToStringsOutput:
        """Invoke with provided services and return outputs."""
        if self.ints is None:
            return None
        if isinstance(self.ints, list):
            return IntsToStringsOutput(int_string=[str(x) for x in self.ints])
        else:
            return IntsToStringsOutput(int_string=[str(self.ints)])


class StringToIntOutput(BaseInvocationOutput):
    """class for StringToIntOutput"""
    type: Literal["string_to_int_output"] = "string_to_int_output"

    ints: int = Field(default=1, description="int")

    class Config:
        schema_extra = {"required": ["type", "ints"]}
    
class StringToIntInvocation(BaseInvocation):
    """class for StringToInt converts a string to an int"""
    type: Literal["string_to_int"] = "string_to_int"

    int_string: str = Field(default='', description="string")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "String To Int",
                "type_hints": {
                    "ints": "integer",
                }
            },
        }

    def invoke(self, context: InvocationContext) -> StringToIntOutput:
        """Invoke with provided services and return outputs."""
        return StringToIntOutput(ints=int(self.int_string))


class XYCollectOutput(BaseInvocationOutput):
    """class for XYCollectionExpand a collection that contains every combination of the input collections"""
    type: Literal["xy_collect_output"] = "xy_collect_output"

    xy_collection: list[list[str]] = Field(description="The x y product collection")

    class Config:
        schema_extra = {"required": ["type", "xy_collection"]}

class XYCollectInvocation(BaseInvocation):
    """class for XYCollectionExpand a collection that contains every combination of the input collections"""

    type: Literal["xy_collect"] = "xy_collect"

    x_collection: list[str] = Field(default=[], description="The X collection")
    y_collection: list[str] = Field(default=[], description="The Y collection")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "XY Collect",
                "type_hints": {
                }
            },
        }

    def invoke(self, context: InvocationContext) -> XYCollectOutput:
        """Invoke with provided services and return outputs."""
        return XYCollectOutput(xy_collection=list(product(self.x_collection, self.y_collection)))

class XYExpandOutput(BaseInvocationOutput):
    """class for XYCollectionExpand a collection that contains every combination of the input collections"""
    type: Literal["xy_expand_output"] = "xy_expand_output"

    x_item: str = Field(description="The X item")
    y_item: str = Field(description="The y item")

    schema_extra = {
        'required': [
            'type',
            'x_item',
            'y_item',
        ],
        "ui": {
            "type_hints": {
                "x_item": "number",
                "y_item": "number",
                }
        }
    }  

class XYExpandInvocation(BaseInvocation):
    """takes an XY Collection and expands it to an x item and y item"""
    type: Literal["xy_expand"] = "xy_expand"

    xy_collection: list[str] = Field(default=[], description="The XY collection item")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "XY Expand",
                "type_hints": {
                }
            },
        }

    def invoke(self, context: InvocationContext) -> XYExpandOutput:
        """Invoke with provided services and return outputs"""
        return XYExpandOutput(x_item=self.xy_collection[0], y_item=self.xy_collection[1])


class XYImageCollectOutput(BaseInvocationOutput):
    """class for XYCollectionExpand a collection that contains every combination of the input collections"""
    type: Literal["xyimage_collect_output"] = "xyimage_collect_output"

    xyimage: str = Field(description="The XY Image ")

    class Config:
        schema_extra = {
            'required': [
                'type',
                'xyimage',
            ]
        }

class XYImageCollectInvocation(BaseInvocation):
    """class for XYCollectionExpand a collection that contains every combination of the input collections"""

    type: Literal["XYCollectionExpand"] = "XYCollectionExpand"

    x_item: str = Field(default='', description="The X item")
    y_item: str = Field(default='', description="The Y item")
    image: ImageField = Field(default=None, description="The image collection to turn into grids")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "XYImage Collect",
                "type_hints": {
                    }
                }
            }

    def invoke(self, context: InvocationContext) -> XYImageCollectOutput:
        """Invoke with provided services and return outputs."""
        return XYImageCollectOutput(xyimage = json.dumps([self.x_item, self.y_item, self.image.image_name]))


class XYImageToGridOutput(BaseInvocationOutput):
    """class for XYImageToGridOutput"""

    # fmt: off
    type: Literal["xyimage_grid_output"] = "xyimage_grid_output"

    # Outputs
    collection: list[ImageField] = Field(default=[], description="The output images")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "collection"]}


class XYImagesToGridInvocation(BaseInvocation):#, PILInvocationConfig):
    """Load a collection of xyimage types and create a gridimage of them"""

    # fmt: off
    type: Literal["xyimage_grid"] = "xyimage_grid"

    # Inputs
    xyimages: list[str] = Field(default=[], description="The xyImage Collection")
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
                "title": "XYImage To Grid",
                "type_hints": {
                }
            },
        }

    def invoke(self, context: InvocationContext) -> XYImageToGridOutput:
        """Convert an image list into a grids of images"""
        new_array = [json.loads(s) for s in self.xyimages]
        sorted_array = sorted(new_array, key=lambda x: (x[0], x[1]))
        images = [context.services.images.get_pil_image(subarray[2]) for subarray in sorted_array]
        columns = len(set([item[0] for item in sorted_array]))
        rows = len(set([item[1] for item in sorted_array]))
        width_max = int(max([image.width for image in images]) * self.scale_factor)
        height_max = int(max([image.height for image in images]) * self.scale_factor)
        background_width = width_max * columns + (self.space * (columns - 1))
        background_height = height_max * rows + (self.space * (rows - 1))
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
            if column >= columns:
                column = 0
                x_offset = 0
                y_offset += height_max + self.space
                row += 1
            
            if row >= rows:
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

        return XYImageToGridOutput(collection=grid_images)


class ImagesToGridsOutput(BaseInvocationOutput):
    """Base class for ImagesToGridsOutput that output nothing"""

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
                "title": "Images To Grids",
                "type_hints": {
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
