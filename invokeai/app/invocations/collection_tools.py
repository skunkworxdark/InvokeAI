# 2024 skunkworxdark (https://github.com/skunkworxdark)

import json
import random
from typing import Any, Optional

from pydantic import BaseModel

from invokeai.app.invocations.model import LoRAField
from invokeai.app.invocations.primitives import (
    BooleanCollectionInvocation,
    BooleanCollectionOutput,
    ConditioningCollectionInvocation,
    ConditioningCollectionOutput,
    FloatCollectionInvocation,
    FloatCollectionOutput,
    ImageCollectionInvocation,
    ImageCollectionOutput,
    IntegerCollectionInvocation,
    IntegerCollectionOutput,
    LatentsCollectionInvocation,
    LatentsCollectionOutput,
    StringCollectionInvocation,
    StringCollectionOutput,
)
from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    BooleanOutput,
    ConditioningField,
    FieldDescriptions,
    FloatOutput,
    ImageField,
    ImageOutput,
    Input,
    InputField,
    IntegerOutput,
    InvocationContext,
    LatentsField,
    OutputField,
    StringOutput,
    UIComponent,
    UIType,
    invocation,
    invocation_output,
)


@invocation(
    "boolean_collection_linked",
    title="Boolean Collection Primitive Linked",
    tags=["primitives", "boolean", "collection"],
    category="primitives",
    version="1.0..0",
)
class BooleanCollectionLinkedInvocation(BooleanCollectionInvocation):
    """A collection of boolean primitive values"""

    value: bool = InputField(default=False, description="The boolean value")

    def invoke(self, context: InvocationContext) -> BooleanCollectionOutput:
        if self.value:
            self.collection.append(self.value)

        obj = super().invoke(context)

        params = obj.__dict__.copy()
        del params["type"]
        return BooleanCollectionOutput(collection=self.collection)


@invocation(
    "conditioning_collection_linked",
    title="Conditioning Collection Primitive Linked",
    tags=["primitives", "conditioning", "collection"],
    category="primitives",
    version="1.0.0",
)
class ConditioningCollectionLinkedInvocation(ConditioningCollectionInvocation):
    """A collection of conditioning tensor primitive values"""

    conditioning: ConditioningField = InputField(description=FieldDescriptions.cond, input=Input.Connection)

    def invoke(self, context: InvocationContext) -> ConditioningCollectionOutput:
        if self.conditioning:
            self.collection.append(self.conditioning)

        obj = super().invoke(context)

        params = obj.__dict__.copy()
        del params["type"]
        return ConditioningCollectionOutput(collection=self.collection)


@invocation(
    "float_collection_linked",
    title="Float Collection Primitive linked",
    tags=["primitives", "float", "collection"],
    category="primitives",
    version="1.0.0",
)
class FloatCollectionLinkedInvocation(FloatCollectionInvocation):
    """A collection of float primitive values"""

    value: float = InputField(default=0.0, description="The float value")

    def invoke(self, context: InvocationContext) -> FloatCollectionOutput:
        if self.value:
            self.collection.append(self.value)

        obj = super().invoke(context)

        params = obj.__dict__.copy()
        del params["type"]
        return FloatCollectionOutput(collection=self.collection)


@invocation(
    "image_collection_linked",
    title="Image Collection Primitive linked",
    tags=["primitives", "image", "collection"],
    category="primitives",
    version="1.0.0",
)
class ImageCollectionLinkedInvocation(ImageCollectionInvocation):
    """A collection of image primitive values"""

    image: ImageField = InputField(description="The image to load")

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        if self.image:
            self.collection.append(self.image)

        obj = super().invoke(context)

        params = obj.__dict__.copy()
        del params["type"]

        return ImageCollectionOutput(**params)


@invocation(
    "integer_collection_linked",
    title="Integer Collection Primitive Linked",
    tags=["primitives", "integer", "collection"],
    category="primitives",
    version="1.0.0",
)
class IntegerCollectionLinkedInvocation(IntegerCollectionInvocation):
    """A collection of integer primitive values"""

    value: int = InputField(default=0, description="The integer value")

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        if self.value:
            self.collection.append(self.value)

        obj = super().invoke(context)

        params = obj.__dict__.copy()
        del params["type"]

        return IntegerCollectionOutput(**params)


@invocation(
    "latents_collection_linked",
    title="Latents Collection Primitive Linked",
    tags=["primitives", "latents", "collection"],
    category="primitives",
    version="1.0.0",
)
class LatentsCollectionLinkedInvocation(LatentsCollectionInvocation):
    """A collection of latents tensor primitive values"""

    latents: Optional[LatentsField] = InputField(default=None, description="The latents tensor", input=Input.Connection)

    def invoke(self, context: InvocationContext) -> LatentsCollectionOutput:
        if self.latents:
            self.collection.append(self.latents)

        obj = super().invoke(context)

        params = obj.__dict__.copy()
        del params["type"]

        return LatentsCollectionOutput(**params)


@invocation(
    "string_collection_linked",
    title="String Collection Primitive Linked",
    tags=["primitives", "string", "collection"],
    category="primitives",
    version="1.0.0",
)
class StringCollectionLinkedInvocation(StringCollectionInvocation):
    """Allows creation of collection and optionally add a collection"""

    value: Optional[str] = InputField(default=None, description="The string value", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        if self.value:
            self.collection.append(self.value)

        obj = super().invoke(context)

        params = obj.__dict__.copy()
        del params["type"]

        return StringCollectionOutput(**params)


@invocation_output("lora_collection_output")
class LoRACollectionOutput(BaseInvocationOutput):
    collection: list[LoRAField] = OutputField(description="The collection of input items", title="Collection")


@invocation(
    "lora_collection",
    title="LoRA Collection Primitive",
    tags=["primitives", "lora", "collection"],
    category="primitives",
    version="1.0.0",
)
class LoRACollectionInvocation(BaseInvocation):
    """A collection of LoRA primitive values"""

    collection: list[LoRAField] = InputField(default=[], description="The collection of LoRA values")

    def invoke(self, context: InvocationContext) -> LoRACollectionOutput:
        return LoRACollectionOutput(collection=self.collection)


@invocation_output("collection_sort_output")
class CollectionSortOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(
        description="The collection of input items", title="Collection", ui_type=UIType._Collection
    )


@invocation(
    "collection_sort",
    title="Collection Sort",
    tags=["collection", "sort"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class CollectionSortInvocation(BaseInvocation):
    """CollectionSort Sorts a collection"""

    collection: list[Any] = InputField(
        description="collection",
        default=[],
        ui_type=UIType._Collection,
    )
    reverse: bool = InputField(
        default=False,
        description="Reverse Sort",
    )

    def sort_list(self, items: list[Any], reverse: bool = False) -> list[Any]:
        if all(isinstance(item, (int, float, str, bool)) for item in items):
            # If all items are of a simple sortable type, use the built-in sort
            sorted_items = sorted(items, reverse=reverse)
        elif all(isinstance(item, BaseModel) for item in items):
            # If all items are Pydantic models, sort based on their JSON representations
            sorted_items = sorted(items, key=lambda x: json.dumps(x.model_dump(), sort_keys=True), reverse=reverse)
        else:
            # Sort based on JSON string representation of items
            sorted_items = sorted(items, key=lambda x: json.dumps(x, sort_keys=True), reverse=reverse)

        return sorted_items

    def invoke(self, context: InvocationContext) -> CollectionSortOutput:
        return CollectionSortOutput(collection=self.sort_list(self.collection, self.reverse))


@invocation_output("collection_index_output")
class CollectionIndexOutput(BaseInvocationOutput):
    """Used to connect iteration outputs. Will be expanded to a specific output."""

    item: Any = OutputField(
        description="The item being iterated over", title="Collection Item", ui_type=UIType._CollectionItem
    )
    index: int = OutputField(description="The index of the item", title="Index")
    total: int = OutputField(description="The total number of items", title="Total")


@invocation(
    "collection_index",
    title="Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class CollectionIndexInvocation(BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[Any] = InputField(
        description="collection",
        ui_type=UIType._Collection,
    )
    random: bool = InputField(
        default=True,
        description="Random Index?",
    )
    index: int = InputField(
        default=0,
        ge=0,
        description="zero based index into collection (note index will wrap around if out of bounds)",
    )

    def invoke(self, context: InvocationContext) -> CollectionIndexOutput:
        total = len(self.collection)
        index = random.randrange(total) if self.random else self.index % total
        item = self.collection[index]

        return CollectionIndexOutput(item=item, index=index, total=total)


@invocation(
    "image_collection_index",
    title="Image Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class ImageCollectionIndexInvocation(BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[ImageField] = InputField(
        description="image collection",
    )
    random: bool = InputField(
        default=True,
        description="Random Index?",
    )
    index: int = InputField(
        default=0,
        ge=0,
        description="zero based index into collection (note index will wrap around if out of bounds)",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        total = len(self.collection)
        index = random.randrange(total) if self.random else self.index % total

        return ImageOutput.build(context.images.get_dto(self.collection[index].image_name))


@invocation(
    "string_collection_index",
    title="String Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class StringCollectionIndexInvocation(BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[str] = InputField(
        description="image collection",
    )
    random: bool = InputField(
        default=True,
        description="Random Index?",
    )
    index: int = InputField(
        default=0,
        ge=0,
        description="zero based index into collection (note index will wrap around if out of bounds)",
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        total = len(self.collection)
        index = random.randrange(total) if self.random else self.index % total

        return StringOutput(value=self.collection[index])


@invocation(
    "integer_collection_index",
    title="Integer Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class IntegerCollectionIndexInvocation(BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[int] = InputField(
        description="image collection",
    )
    random: bool = InputField(
        default=True,
        description="Random Index?",
    )
    index: int = InputField(
        default=0,
        ge=0,
        description="zero based index into collection (note index will wrap around if out of bounds)",
    )

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        total = len(self.collection)
        index = random.randrange(total) if self.random else self.index % total

        return IntegerOutput(value=self.collection[index])


@invocation(
    "float_collection_index",
    title="Float Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.1.0",
    use_cache=False,
)
class FloatCollectionIndexInvocation(BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[float] = InputField(
        description="image collection",
    )
    random: bool = InputField(
        default=True,
        description="Random Index?",
    )
    index: int = InputField(
        default=0,
        ge=0,
        description="zero based index into collection (note index will wrap around if out of bounds)",
    )

    def invoke(self, context: InvocationContext) -> FloatOutput:
        total = len(self.collection)
        index = random.randrange(total) if self.random else self.index % total

        return FloatOutput(value=self.collection[index])


@invocation(
    "bool_collection_index",
    title="Bool Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class BoolCollectionIndexInvocation(BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[bool] = InputField(
        description="image collection",
    )
    random: bool = InputField(
        default=True,
        description="Random Index?",
    )
    index: int = InputField(
        default=0,
        ge=0,
        description="zero based index into collection (note index will wrap around if out of bounds)",
    )

    def invoke(self, context: InvocationContext) -> BooleanOutput:
        total = len(self.collection)
        index = random.randrange(total) if self.random else self.index % total

        return BooleanOutput(value=self.collection[index])
