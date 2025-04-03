"""Blend two images in their frequency domain. [numpy implementation]"""

import numpy as np
import torch
from torchvision.transforms.v2.functional import pil_to_tensor
from torchvision.transforms.v2.functional import to_pil_image as pil_image_from_tensor

from invokeai.backend.image_util.composition import (
    linear_srgb_from_oklab,
    linear_srgb_from_srgb,
    oklab_from_linear_srgb,
    srgb_from_linear_srgb,
)
from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    LatentsField,
    LatentsOutput,
    invocation,
)


def _linear_transition_mask(shape_f: tuple, start_position: float, transition_width: float):
    """Creates a smooth left ellipse mask with linear transition (vectorized)."""
    radius_x = shape_f[-1] / 2
    radius_y = shape_f[-2] / 2
    ii_y, ii_x = np.indices(shape_f[-2:])
    center_row = shape_f[-2] / 2

    # Corrected distance calculation for left ellipse
    distances_squared = (ii_x) ** 2 + ((ii_y - center_row) * (radius_x / radius_y)) ** 2
    max_distance_squared = radius_x**2 + radius_y**2
    normalized_distances = np.sqrt(distances_squared / max_distance_squared)

    start_distance = start_position
    end_distance = start_position + transition_width

    mask = np.ones(shape_f[-2:], dtype=np.float32)
    mask[(normalized_distances >= start_distance) & (normalized_distances < end_distance)] = 1.0 - (
        normalized_distances[(normalized_distances >= start_distance) & (normalized_distances < end_distance)] - start_distance
    ) / (end_distance - start_distance)
    mask[normalized_distances >= end_distance] = 0.0

    return mask


def _corner_mask_rounded(shape_in: tuple, shape_f: tuple, threshold: float):
    """Mask the area nearest (0, height/2) by Euclidean distance.

    The masked area is a semicircle.
    """
    max_distance_squared = shape_in[-1] ** 2 + (shape_in[-2] / 2) ** 2
    ii = np.indices(shape_f)
    center_row = shape_f[-2] / 2
    mask = (ii[-1] ** 2 + (ii[-2] - center_row) ** 2) < (threshold**2 * max_distance_squared)
    return mask


def blend_np(high: np.ndarray, low: np.ndarray, threshold: float, weighted: bool = False, sharpness: float = 10.0) -> np.ndarray:
    """Blend together frequency components of two images."""
    assert high.shape == low.shape, "inputs must be same size"

    high_freqs = np.fft.fftshift(np.fft.rfft2(high), axes=-2)
    low_freqs = np.fft.fftshift(np.fft.rfft2(low), axes=-2)

    if weighted:
        mask = _linear_transition_mask(high_freqs.shape, threshold**3, sharpness**3)
        combined_freqs = (mask * high_freqs) + ((1 - mask) * low_freqs)
    else:
        mask = _corner_mask_rounded(high.shape, high_freqs.shape, threshold)
        combined_freqs = np.where(mask, low_freqs, high_freqs)

    return np.fft.irfft2(np.fft.ifftshift(combined_freqs, axes=-2))


def blend(high, low, threshold, weighted: bool = False, sharpness: float = 0.1):
    # bet we can do a pytorch implementation, but numpy was easier to test initially.
    try:
        high = high.numpy()
        low = low.numpy()
    except TypeError:
        # Some types like bfloat16 can't convert directly to numpy.
        high = high.to(dtype=torch.float32).numpy()
        low = low.to(dtype=torch.float32).numpy()

    result_array = blend_np(high, low, threshold, weighted, sharpness)
    return torch.from_numpy(result_array)


@invocation(
    "frequency_blend_latents",
    title="Frequency Blend Latents",
    version="0.9.0",
)
class FrequencyBlendLatents(BaseInvocation):
    high_input: LatentsField = InputField(description="Image from which to take high frequencies.")
    low_input: LatentsField = InputField(description="Image from which to take low frequencies.")
    threshold: float = InputField(
        description="How much of the high-frequency to use. Negative values swap the inputs.", ge=-1, le=1, default=0.02
    )
    weighted_blend: bool = InputField(description="Weighted blend?", default=True)
    sharpness: float = InputField(description="Sharpness of weighted blend transition (0-1 lower is sharper)", gt=0, le=1, default=0.1)

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        high_input: torch.Tensor = context.tensors.load(self.high_input.latents_name)
        low_input: torch.Tensor = context.tensors.load(self.low_input.latents_name)
        threshold = self.threshold
        if threshold < 0:
            threshold = abs(threshold)
            high_input, low_input = low_input, high_input

        result = blend(high_input, low_input, threshold, self.weighted_blend)
        name = context.tensors.save(tensor=result)
        return LatentsOutput.build(latents_name=name, latents=result)


def oklab_from_srgb(rgb_tensor):
    # or should this use ImageCMS?
    return oklab_from_linear_srgb(linear_srgb_from_srgb(rgb_tensor))


def srgb_from_oklab(oklab_tensor):
    return srgb_from_linear_srgb(linear_srgb_from_oklab(oklab_tensor))


@invocation(
    "frequency_blend_image",
    title="Frequency Blend Image",
    version="0.1.0",
)
class FrequencyBlendImage(BaseInvocation):
    high_input: ImageField = InputField(description="Image from which to take high frequencies.")
    low_input: ImageField = InputField(description="Image from which to take low frequencies.")
    threshold: float = InputField(
        description="How much of the high-frequency to use. Negative values swap the inputs.", ge=-1, le=1, default=0.02
    )
    weighted_blend: bool = InputField(description="Weighted blend?", default=True)
    sharpness: float = InputField(description="Sharpness of weighted blend transition (0-1 lower is sharper)", gt=0, le=1, default=0.1)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        get_as_oklab = lambda image_input: oklab_from_srgb(
            pil_to_tensor(context.images.get_pil(image_input.image_name, mode="RGB")) / 255.0
        )

        high_input: torch.Tensor = get_as_oklab(self.high_input)
        low_input: torch.Tensor = get_as_oklab(self.low_input)
        threshold = self.threshold
        if threshold < 0:
            threshold = abs(threshold)
            high_input, low_input = low_input, high_input

        result = blend(high_input, low_input, threshold, self.weighted_blend, self.sharpness)
        result_image = pil_image_from_tensor(srgb_from_oklab(result.to(dtype=torch.float32)))
        return ImageOutput.build(context.images.save(result_image))
