# 2024 skunkworxdark (https://github.com/skunkworxdark)

import json
import random
from typing import Any

from pydantic import BaseModel

from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    BooleanOutput,
    FloatOutput,
    ImageField,
    ImageOutput,
    InputField,
    IntegerOutput,
    InvocationContext,
    OutputField,
    StringOutput,
    UIType,
    invocation,
    invocation_output,
)


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
    version="1.0.0",
    use_cache=False,
)
class FloatCollectionIndexInvocation(BaseInvocation):
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
