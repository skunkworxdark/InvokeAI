# 2024 skunkworxdark (https://github.com/skunkworxdark)

import json
import random
from typing import Any, Optional, TypeVar, Union, cast

from pydantic import BaseModel

from invokeai.app.invocations.fields import FluxReduxConditioningField
from invokeai.app.invocations.flux_controlnet import FluxControlNetField, FluxControlNetOutput
from invokeai.app.invocations.flux_redux import FluxReduxOutput
from invokeai.app.invocations.model import LoRAField
from invokeai.app.invocations.primitives import (
    BooleanCollectionInvocation,
    BooleanCollectionOutput,
    ConditioningCollectionInvocation,
    ConditioningCollectionOutput,
    FloatCollectionInvocation,
    FloatCollectionOutput,
    FluxConditioningOutput,
    ImageCollectionInvocation,
    ImageCollectionOutput,
    IntegerCollectionInvocation,
    IntegerCollectionOutput,
    LatentsCollectionInvocation,
    LatentsCollectionOutput,
    LatentsOutput,
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
    FluxConditioningCollectionOutput,
    FluxConditioningField,
    ImageField,
    ImageOutput,
    Input,
    InputField,
    IntegerOutput,
    InvocationContext,
    LatentsField,
    ModelIdentifierField,
    OutputField,
    StringOutput,
    UIComponent,
    UIType,
    invocation,
    invocation_output,
)

T = TypeVar("T")


def _to_list(item_cls: type[T], value: Union[T, list[T], None], name: str) -> list[T]:
    """Converts a single item or a list of items to a list, ensuring type consistency."""

    if value is None:
        return []
    if isinstance(value, list):
        if not all(isinstance(i, item_cls) for i in value):
            raise ValueError(f"Invalid items type in '{name}': {value}, expected list of {item_cls.__name__}")
        return cast(list[T], value)
    if isinstance(value, item_cls):
        return [value]
    raise ValueError(
        f"Invalid type for '{name}': {type(value)}, expected {item_cls.__name__} or list of {item_cls.__name__} or None"
    )


def join_collections(item_cls: type[T], a: Union[T, list[T], None], b: Union[T, list[T], None] = None) -> list[T]:
    """joins any number of items or lists into a single list, ensuring consistency in type."""

    processed_a = _to_list(item_cls, a, "a")
    processed_b = _to_list(item_cls, b, "b")
    return processed_a + processed_b


def append_item_to_list(item_cls: type[T], new_item: Optional[T], items: Union[T, list[T], None] = None) -> list[T]:
    """appends an item to a list of that item, ensuring consistency in type."""

    existing_items = _to_list(item_cls, items, "items")
    if new_item is not None:
        if not isinstance(new_item, item_cls):
            raise ValueError(f"Invalid new_item type in: {new_item},  expected {item_cls}")
        existing_items.append(new_item)
    return existing_items


class IndexCollectionMixin(BaseInvocation):
    """Mixin for invocations that index a specific type of collection."""

    random: bool = InputField(default=True, description="Random Index?")
    index: int = InputField(
        default=0, ge=0, description="zero based index into collection (note index will wrap around if out of bounds)"
    )

    def _get_selected_item(self) -> Any:
        """Retrieves an item from the 'collection' based on index or randomness."""
        current_collection = getattr(self, "collection")  # Assumes 'collection' field exists
        if not current_collection or len(current_collection) == 0:
            raise ValueError("Input collection is empty.")
        total = len(current_collection)
        selected_index = random.randrange(total) if self.random else self.index % total
        return current_collection[selected_index]

    def _get_selected_item_with_info(self) -> tuple[Any, int, int]:
        """Retrieves an item from the 'collection' based on index or randomness, along with its index and the total count."""
        current_collection = getattr(self, "collection")  # Assumes 'collection' field exists
        if not current_collection or len(current_collection) == 0:
            raise ValueError("Input collection is empty.")
        total = len(current_collection)
        selected_index = random.randrange(total) if self.random else self.index % total
        return current_collection[selected_index], selected_index, total


@invocation(
    "boolean_collection_linked",
    title="Boolean Collection Primitive Linked",
    tags=["primitives", "boolean", "collection"],
    category="primitives",
    version="1.0.1",
)
class BooleanCollectionLinkedInvocation(BooleanCollectionInvocation):
    """A collection of boolean primitive values"""

    value: bool = InputField(default=False, description="The boolean value")

    def invoke(self, context: InvocationContext) -> BooleanCollectionOutput:
        self.collection = append_item_to_list(bool, self.value, self.collection)
        return super().invoke(context)


@invocation(
    "conditioning_collection_linked",
    title="Conditioning Collection Primitive Linked",
    tags=["primitives", "conditioning", "collection"],
    category="primitives",
    version="1.0.1",
)
class ConditioningCollectionLinkedInvocation(ConditioningCollectionInvocation):
    """A collection of conditioning tensor primitive values"""

    conditioning: ConditioningField = InputField(description=FieldDescriptions.cond, input=Input.Connection)

    def invoke(self, context: InvocationContext) -> ConditioningCollectionOutput:
        self.collection = append_item_to_list(ConditioningField, self.conditioning, self.collection)
        return super().invoke(context)


@invocation(
    "float_collection_linked",
    title="Float Collection Primitive linked",
    tags=["primitives", "float", "collection"],
    category="primitives",
    version="1.0.1",
)
class FloatCollectionLinkedInvocation(FloatCollectionInvocation):
    """A collection of float primitive values"""

    value: float = InputField(default=0.0, description="The float value")

    def invoke(self, context: InvocationContext) -> FloatCollectionOutput:
        self.collection = append_item_to_list(float, self.value, self.collection)
        return super().invoke(context)


@invocation(
    "image_collection_linked",
    title="Image Collection Primitive linked",
    tags=["primitives", "image", "collection"],
    category="primitives",
    version="1.0.1",
)
class ImageCollectionLinkedInvocation(ImageCollectionInvocation):
    """A collection of image primitive values"""

    image: ImageField = InputField(description="The image to load")

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        self.collection = append_item_to_list(ImageField, self.image, self.collection)
        return super().invoke(context)


@invocation(
    "integer_collection_linked",
    title="Integer Collection Primitive Linked",
    tags=["primitives", "integer", "collection"],
    category="primitives",
    version="1.0.1",
)
class IntegerCollectionLinkedInvocation(IntegerCollectionInvocation):
    """A collection of integer primitive values"""

    value: int = InputField(default=0, description="The integer value")

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        self.collection = append_item_to_list(int, self.value, self.collection)
        return super().invoke(context)


@invocation(
    "latents_collection_linked",
    title="Latents Collection Primitive Linked",
    tags=["primitives", "latents", "collection"],
    category="primitives",
    version="1.0.1",
)
class LatentsCollectionLinkedInvocation(LatentsCollectionInvocation):
    """A collection of latents tensor primitive values"""

    latents: Optional[LatentsField] = InputField(default=None, description="The latents tensor", input=Input.Connection)

    def invoke(self, context: InvocationContext) -> LatentsCollectionOutput:
        self.collection = append_item_to_list(LatentsField, self.latents, self.collection)
        return super().invoke(context)


@invocation(
    "string_collection_linked",
    title="String Collection Primitive Linked",
    tags=["primitives", "string", "collection"],
    category="primitives",
    version="1.0.1",
)
class StringCollectionLinkedInvocation(StringCollectionInvocation):
    """Allows creation of collection and optionally add a collection"""

    value: Optional[str] = InputField(default=None, description="The string value", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        self.collection = append_item_to_list(str, self.value, self.collection)
        return super().invoke(context)


@invocation_output("lora_collection_output")
class LoRACollectionOutput(BaseInvocationOutput):
    collection: list[LoRAField] = OutputField(description="The collection of input items", title="LoRAs")


@invocation(
    "lora_collection",
    title="LoRA Collection Primitive",
    tags=["primitives", "lora", "collection"],
    category="primitives",
    version="1.0.0",
)
class LoRACollectionInvocation(BaseInvocation):
    """A collection of LoRA primitive values"""

    collection: list[LoRAField] = InputField(default=[], description="The collection of LoRA values", title="LoRAs")

    def invoke(self, context: InvocationContext) -> LoRACollectionOutput:
        return LoRACollectionOutput(collection=self.collection)


@invocation(
    "lora_collection_linked",
    title="LoRA Collection Primitive Linked",
    tags=["primitives", "lora", "collection"],
    category="primitives",
    version="1.0.0",
)
class LoRACollectionLinkedInvocation(LoRACollectionInvocation):
    """Selects a LoRA model and weight."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)

    def invoke(self, context: InvocationContext) -> LoRACollectionOutput:
        new_lora_field = LoRAField(lora=self.lora, weight=self.weight)

        lora_keys_in_collection = {lora.lora.key for lora in self.collection}

        if new_lora_field.lora.key not in lora_keys_in_collection:
            self.collection = join_collections(LoRAField, self.collection, new_lora_field)

        return super().invoke(context)


@invocation(
    "flux_conditioning_collection",
    title="Flux Conditioning Collection Primitive",
    tags=["flux", "text_encoder", "conditioning", "primitives", "collection"],
    category="primitives",
    version="1.0.0",
)
class FluxConditioningCollectionInvocation(BaseInvocation):
    """A collection of flux conditioning tensor primitive values"""

    conditioning: list[FluxConditioningField] = InputField(
        default=[],
        description=FieldDescriptions.cond,
        title="FLUX Conditionings",
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningCollectionOutput:
        return FluxConditioningCollectionOutput(collection=self.conditioning)


@invocation(
    "flux_conditioning_collection_join",
    title="Flux Conditioning Collection join",
    tags=["flux", "text_encoder", "conditioning", "collection", "join"],
    category="util",
    version="1.0.0",
)
class FluxConditioningCollectionJoinInvocation(BaseInvocation):
    """Join a flux conditioning tensor or collections into a single collection of flux conditioning tensors"""

    conditionings_a: Optional[Union[FluxConditioningField, list[FluxConditioningField]]] = InputField(
        description=FieldDescriptions.cond,
        title="FLUX Text Encoder Conditioning or Collection",
        default=None,
        input=Input.Connection,
    )
    conditionings_b: Optional[Union[FluxConditioningField, list[FluxConditioningField]]] = InputField(
        description=FieldDescriptions.cond,
        title="FLUX Text Encoder Conditioning or Collection",
        default=None,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningCollectionOutput:
        conditionings = join_collections(FluxConditioningField, self.conditionings_a, self.conditionings_b)
        return FluxConditioningCollectionOutput(collection=conditionings)


@invocation_output("flux_controlnet_collection_output")
class FluxControlNetCollectionOutput(BaseInvocationOutput):
    collection: list[FluxControlNetField] = OutputField(
        description=FieldDescriptions.control, title="FLUX ControlNet List"
    )


# --FLUX ControlNet Collection
@invocation(
    "flux_controlnet_collection",
    title="FLUX ControlNet Collection Primitive",
    tags=["flux", "controlnet", "primitives", "collection"],
    category="primitives",
    version="1.0.0",
)
class FluxControlNetCollectionInvocation(BaseInvocation):
    """A collection of flux controlnet primitive values"""

    collection: list[FluxControlNetField] = InputField(
        description="FLUX ControlNets",
        title="FLUX ControlNet Collection",
    )

    def invoke(self, context: InvocationContext) -> FluxControlNetCollectionOutput:
        return FluxControlNetCollectionOutput(collection=self.collection)


@invocation(
    "flux_controlnet_collection_join",
    title="FLUX ControlNet Collection join",
    tags=["flux", "controlnet", "collection", "join"],
    category="util",
    version="1.0.0",
)
class FluxControlNetCollectionJoinInvocation(BaseInvocation):
    """Join a flux controlnet tensors or collections into a single collection of flux controlnet tensors"""

    controlnets_a: Optional[Union[FluxControlNetField, list[FluxControlNetField]]] = InputField(
        description="FLUX ControlNets",
        title="FLUX ControlNet or Collection",
        default=None,
        input=Input.Connection,
    )
    controlnets_b: Optional[Union[FluxControlNetField, list[FluxControlNetField]]] = InputField(
        description="FLUX ControlNets",
        title="FLUX ControlNet or Collection",
        default=None,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> FluxControlNetCollectionOutput:
        controlnets = join_collections(FluxControlNetField, self.controlnets_a, self.controlnets_b)
        return FluxControlNetCollectionOutput(collection=controlnets)


# --FLUX Redux Collection
@invocation_output("flux_redux_collection_output")
class FluxReduxCollectionOutput(BaseInvocationOutput):
    collection: list[FluxReduxConditioningField] = OutputField(
        description=FieldDescriptions.control, title="FLUX Redux Collection"
    )


@invocation(
    "flux_redux_collection",
    title="FLUX Redux Collection Primitive",
    tags=["flux", "redux", "primitives", "collection"],
    category="primitives",
    version="1.0.0",
)
class FluxReduxCollectionInvocation(BaseInvocation):
    """A collection of flux redux primitive values"""

    collection: list[FluxReduxConditioningField] = InputField(
        description="FLUX Redux Collection",
        title="FLUX Redux Collection",
    )

    def invoke(self, context: InvocationContext) -> FluxReduxCollectionOutput:
        return FluxReduxCollectionOutput(collection=self.collection)


@invocation(
    "flux_redux_collection_join",
    title="FLUX Redux Collection join",
    tags=["flux", "redux", "collection", "join"],
    category="util",
    version="1.0.0",
)
class FluxReduxCollectionJoinInvocation(BaseInvocation):
    """Join a flux redux tensor or collections into a single collection of flux redux tensors"""

    conditionings_a: Optional[Union[FluxReduxConditioningField, list[FluxReduxConditioningField]]] = InputField(
        description="FLUX Reduxs",
        title="FLUX Redux or Collection",
        default=None,
        input=Input.Connection,
    )
    conditionings_b: Optional[Union[FluxReduxConditioningField, list[FluxReduxConditioningField]]] = InputField(
        description="FLUX Reduxs",
        title="FLUX Redux or Collection",
        default=None,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> FluxReduxCollectionOutput:
        conditionings = join_collections(FluxReduxConditioningField, self.conditionings_a, self.conditionings_b)
        return FluxReduxCollectionOutput(collection=conditionings)


# ---------------------------------- Collection [Any] manipulation -----------------
@invocation_output("collection_sort_output")
class CollectionSortOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(
        description="The collection of output items", title="Collection", ui_type=UIType._Collection
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


@invocation_output("collection_join_output")
class CollectionJoinOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(
        description="The collection of output items", title="Collection", ui_type=UIType._Collection
    )


@invocation(
    "collection_join",
    title="Collection Join",
    tags=["collection", "join"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class CollectionJoinInvocation(BaseInvocation):
    """CollectionJoin Joins two collections into a single collection"""

    collection_a: list[Any] = InputField(
        description="collection",
        default=[],
        ui_type=UIType._Collection,
    )
    collection_b: list[Any] = InputField(
        description="collection",
        default=[],
        ui_type=UIType._Collection,
    )

    def invoke(self, context: InvocationContext) -> CollectionJoinOutput:
        return CollectionJoinOutput(collection=self.collection_a + self.collection_b)


@invocation_output("collection_index_output")
class CollectionIndexOutput(BaseInvocationOutput):
    """Used to connect iteration outputs. Will be expanded to a specific output."""

    item: Any = OutputField(
        description="The item being iterated over", title="Collection Item", ui_type=UIType._CollectionItem
    )
    index: int = OutputField(description="The index of the selected item", title="Index")
    total: int = OutputField(description="The total number of items in the collection", title="Total")


@invocation(
    "collection_index",
    title="Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.1",
    use_cache=False,
)
class CollectionIndexInvocation(IndexCollectionMixin, BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[Any] = InputField(description="collection", ui_type=UIType._Collection)

    def invoke(self, context: InvocationContext) -> CollectionIndexOutput:
        selected_item, selected_index, total_items = self._get_selected_item_with_info()
        return CollectionIndexOutput(item=selected_item, index=selected_index, total=total_items)


@invocation_output("collection_count_output")
class CollectionCountOutput(BaseInvocationOutput):
    """The output of the collection count node."""

    count: int = OutputField(description="The number of items in the collection", title="Count")


@invocation(
    "collection_count",
    title="Collection Count",
    tags=["collection", "count"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class CollectionCountInvocation(BaseInvocation):
    """Counts the number of items in a collection."""

    collection: list[Any] = InputField(description="The collection to count", default=[], ui_type=UIType._Collection)

    def invoke(self, context: InvocationContext) -> CollectionCountOutput:
        """Counts the items in the collection."""
        return CollectionCountOutput(count=len(self.collection))


@invocation_output("collection_slice_output")
class CollectionSliceOutput(BaseInvocationOutput):
    """The output of the collection slice node."""

    collection: list[Any] = OutputField(
        description="The sliced collection", title="Collection", ui_type=UIType._Collection
    )


@invocation(
    "collection_slice",
    title="Collection Slice",
    tags=["collection", "slice"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class CollectionSliceInvocation(BaseInvocation):
    """Slices a collection."""

    collection: list[Any] = InputField(description="The collection to slice", default=[], ui_type=UIType._Collection)
    start: int = InputField(default=0, description="The start index of the slice")
    stop: Optional[int] = InputField(default=None, description="The stop index of the slice (exclusive)")
    step: int = InputField(default=1, ge=1, description="The step of the slice")

    def invoke(self, context: InvocationContext) -> CollectionSliceOutput:
        """Slices the collection."""
        sliced_collection = self.collection[self.start : self.stop : self.step]
        return CollectionSliceOutput(collection=sliced_collection)


@invocation_output("collection_reverse_output")
class CollectionReverseOutput(BaseInvocationOutput):
    """The output of the collection reverse node."""

    collection: list[Any] = OutputField(
        description="The reversed collection", title="Collection", ui_type=UIType._Collection
    )


@invocation(
    "collection_reverse",
    title="Collection Reverse",
    tags=["collection", "reverse"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class CollectionReverseInvocation(BaseInvocation):
    """Reverses a collection."""

    collection: list[Any] = InputField(description="The collection to reverse", default=[], ui_type=UIType._Collection)

    def invoke(self, context: InvocationContext) -> CollectionReverseOutput:
        """Reverses the collection."""
        return CollectionReverseOutput(collection=self.collection[::-1])


@invocation_output("collection_unique_output")
class CollectionUniqueOutput(BaseInvocationOutput):
    """The output of the collection unique node."""

    collection: list[Any] = OutputField(
        description="The collection with unique items", title="Collection", ui_type=UIType._Collection
    )


@invocation(
    "collection_unique",
    title="Collection Unique",
    tags=["collection", "unique", "deduplicate"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class CollectionUniqueInvocation(BaseInvocation):
    """Removes duplicate items from a collection."""

    collection: list[Any] = InputField(
        description="The collection to deduplicate", default=[], ui_type=UIType._Collection
    )

    def invoke(self, context: InvocationContext) -> CollectionUniqueOutput:
        """Removes duplicate items from the collection."""
        seen_representations = set()
        unique_items = []
        for item in self.collection:
            # To handle unhashable types, we create a stable representation.
            if isinstance(item, BaseModel):
                representation = json.dumps(item.model_dump(), sort_keys=True)
            else:
                try:
                    # Attempt to hash the item directly if possible (for simple types).
                    if item not in seen_representations:
                        unique_items.append(item)
                        seen_representations.add(item)
                    continue
                except TypeError:
                    # If the item is not hashable (like a dict), serialize it.
                    representation = json.dumps(item, sort_keys=True, default=str)

            if representation not in seen_representations:
                unique_items.append(item)
                seen_representations.add(representation)

        return CollectionUniqueOutput(collection=unique_items)


# ---------------------------------- Collection type specific manipulation -----------------
@invocation(
    "image_collection_index",
    title="Image Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.1",
    use_cache=False,
)
class ImageCollectionIndexInvocation(IndexCollectionMixin, BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[ImageField] = InputField(description="image collection")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        selected_item = self._get_selected_item()
        return ImageOutput.build(context.images.get_dto(selected_item.image_name))


@invocation(
    "string_collection_index",
    title="String Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.1",
    use_cache=False,
)
class StringCollectionIndexInvocation(IndexCollectionMixin, BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[str] = InputField(description="string collection")

    def invoke(self, context: InvocationContext) -> StringOutput:
        selected_item = self._get_selected_item()
        return StringOutput(value=selected_item)


@invocation(
    "integer_collection_index",
    title="Integer Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.1",
    use_cache=False,
)
class IntegerCollectionIndexInvocation(IndexCollectionMixin, BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[int] = InputField(
        description="integer collection",
    )

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        selected_item = self._get_selected_item()
        return IntegerOutput(value=selected_item)


@invocation(
    "float_collection_index",
    title="Float Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.1.1",
    use_cache=False,
)
class FloatCollectionIndexInvocation(IndexCollectionMixin, BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[float] = InputField(description="float collection")

    def invoke(self, context: InvocationContext) -> FloatOutput:
        selected_item = self._get_selected_item()
        return FloatOutput(value=selected_item)


@invocation(
    "bool_collection_index",
    title="Bool Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.1",
    use_cache=False,
)
class BoolCollectionIndexInvocation(IndexCollectionMixin, BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[bool] = InputField(description="bool collection")

    def invoke(self, context: InvocationContext) -> BooleanOutput:
        selected_item = self._get_selected_item()
        return BooleanOutput(value=selected_item)


@invocation(
    "latents_collection_index",
    title="Latents Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class LatentsCollectionIndexInvocation(IndexCollectionMixin, BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[LatentsField] = InputField(description="latents collection")

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        selected_item = self._get_selected_item()

        return LatentsOutput.build(
            latents_name=selected_item.latents_name,
            latents=context.tensors.load(selected_item.latents_name),
            seed=selected_item.seed,
        )


@invocation(
    "flux_conditioning_index",
    title="Flux Conditioning Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class FluxConditioningCollectionIndexInvocation(IndexCollectionMixin, BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[FluxConditioningField] = InputField(
        default=[],
        description=FieldDescriptions.cond,
        title="FLUX Conditionings",
    )

    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        selected_item = self._get_selected_item()
        return FluxConditioningOutput(conditioning=selected_item)


@invocation(
    "flux_controlnet_index",
    title="Flux ControlNet Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class FluxControlNetCollectionIndexInvocation(IndexCollectionMixin, BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[FluxControlNetField] = InputField(
        default=[],
        description="FLUX ControlNet Collection",
        title="FLUX ControlNets",
    )

    def invoke(self, context: InvocationContext) -> FluxControlNetOutput:
        selected_item = self._get_selected_item()
        return FluxControlNetOutput(control=selected_item)


@invocation(
    "flux_redux_index",
    title="Flux Redux Collection Index",
    tags=["collection", "index"],
    category="util",
    version="1.0.0",
    use_cache=False,
)
class FluxReduxCollectionIndexInvocation(IndexCollectionMixin, BaseInvocation):
    """CollectionIndex Picks an index out of a collection with a random option"""

    collection: list[FluxReduxConditioningField] = InputField(
        default=[],
        description=FieldDescriptions.cond,
        title="FLUX Redux Collection",
    )

    def invoke(self, context: InvocationContext) -> FluxReduxOutput:
        selected_item = self._get_selected_item()
        return FluxReduxOutput(redux_cond=selected_item)
