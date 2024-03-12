import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type {
  CanvasPasteBackInvocation,
  CreateGradientMaskInvocation,
  ImageDTO,
  ImageToLatentsInvocation,
  NoiseInvocation,
  NonNullableGraph,
} from 'services/api/types';

import { addControlNetToLinearGraph } from './addControlNetToLinearGraph';
import { addIPAdapterToLinearGraph } from './addIPAdapterToLinearGraph';
import { addLoRAsToGraph } from './addLoRAsToGraph';
import { addNSFWCheckerToGraph } from './addNSFWCheckerToGraph';
import { addSeamlessToLinearGraph } from './addSeamlessToLinearGraph';
import { addT2IAdaptersToLinearGraph } from './addT2IAdapterToLinearGraph';
import { addVAEToGraph } from './addVAEToGraph';
import { addWatermarkerToGraph } from './addWatermarkerToGraph';
import {
  CANVAS_INPAINT_GRAPH,
  CANVAS_OUTPUT,
  CLIP_SKIP,
  DENOISE_LATENTS,
  INPAINT_CREATE_MASK,
  INPAINT_IMAGE,
  INPAINT_IMAGE_RESIZE_DOWN,
  INPAINT_IMAGE_RESIZE_UP,
  LATENTS_TO_IMAGE,
  MAIN_MODEL_LOADER,
  MASK_RESIZE_DOWN,
  MASK_RESIZE_UP,
  NEGATIVE_CONDITIONING,
  NOISE,
  POSITIVE_CONDITIONING,
  SEAMLESS,
} from './constants';
import { getBoardField, getIsIntermediate } from './graphBuilderUtils';

/**
 * Builds the Canvas tab's Inpaint graph.
 */
export const buildCanvasInpaintGraph = async (
  state: RootState,
  canvasInitImage: ImageDTO,
  canvasMaskImage: ImageDTO
): Promise<NonNullableGraph> => {
  const log = logger('nodes');
  const {
    positivePrompt,
    negativePrompt,
    model,
    cfgScale: cfg_scale,
    cfgRescaleMultiplier: cfg_rescale_multiplier,
    scheduler,
    steps,
    img2imgStrength: strength,
    seed,
    vaePrecision,
    shouldUseCpuNoise,
    clipSkip,
    seamlessXAxis,
    seamlessYAxis,
    canvasCoherenceMode,
    canvasCoherenceMinDenoise,
    canvasCoherenceEdgeSize,
    maskBlur,
  } = state.generation;

  if (!model) {
    log.error('No model found in state');
    throw new Error('No model found in state');
  }

  // The bounding box determines width and height, not the width and height params
  const { width, height } = state.canvas.boundingBoxDimensions;

  // We may need to set the inpaint width and height to scale the image
  const { scaledBoundingBoxDimensions, boundingBoxScaleMethod } = state.canvas;
  const is_intermediate = true;
  const fp32 = vaePrecision === 'fp32';

  const isUsingScaledDimensions = ['auto', 'manual'].includes(boundingBoxScaleMethod);

  let modelLoaderNodeId = MAIN_MODEL_LOADER;

  const use_cpu = shouldUseCpuNoise;

  const graph: NonNullableGraph = {
    id: CANVAS_INPAINT_GRAPH,
    nodes: {
      [modelLoaderNodeId]: {
        type: 'main_model_loader',
        id: modelLoaderNodeId,
        is_intermediate,
        model,
      },
      [CLIP_SKIP]: {
        type: 'clip_skip',
        id: CLIP_SKIP,
        is_intermediate,
        skipped_layers: clipSkip,
      },
      [POSITIVE_CONDITIONING]: {
        type: 'compel',
        id: POSITIVE_CONDITIONING,
        is_intermediate,
        prompt: positivePrompt,
      },
      [NEGATIVE_CONDITIONING]: {
        type: 'compel',
        id: NEGATIVE_CONDITIONING,
        is_intermediate,
        prompt: negativePrompt,
      },
      [INPAINT_IMAGE]: {
        type: 'i2l',
        id: INPAINT_IMAGE,
        is_intermediate,
        fp32,
      },
      [NOISE]: {
        type: 'noise',
        id: NOISE,
        use_cpu,
        seed,
        is_intermediate,
      },
      [INPAINT_CREATE_MASK]: {
        type: 'create_gradient_mask',
        id: INPAINT_CREATE_MASK,
        is_intermediate,
        coherence_mode: canvasCoherenceMode,
        minimum_denoise: canvasCoherenceMinDenoise,
        edge_radius: canvasCoherenceEdgeSize,
      },
      [DENOISE_LATENTS]: {
        type: 'denoise_latents',
        id: DENOISE_LATENTS,
        is_intermediate,
        steps: steps,
        cfg_scale: cfg_scale,
        cfg_rescale_multiplier,
        scheduler: scheduler,
        denoising_start: 1 - strength,
        denoising_end: 1,
      },
      [LATENTS_TO_IMAGE]: {
        type: 'l2i',
        id: LATENTS_TO_IMAGE,
        is_intermediate,
        fp32,
      },
      [CANVAS_OUTPUT]: {
        type: 'canvas_paste_back',
        id: CANVAS_OUTPUT,
        is_intermediate: getIsIntermediate(state),
        board: getBoardField(state),
        mask_blur: maskBlur,
        source_image: canvasInitImage,
      },
    },
    edges: [
      // Connect Model Loader to CLIP Skip and UNet
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'unet',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'unet',
        },
      },
      {
        source: {
          node_id: modelLoaderNodeId,
          field: 'clip',
        },
        destination: {
          node_id: CLIP_SKIP,
          field: 'clip',
        },
      },
      // Connect CLIP Skip to Conditioning
      {
        source: {
          node_id: CLIP_SKIP,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      },
      {
        source: {
          node_id: CLIP_SKIP,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      },
      // Connect Everything To Inpaint Node
      {
        source: {
          node_id: POSITIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'positive_conditioning',
        },
      },
      {
        source: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'conditioning',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'negative_conditioning',
        },
      },
      {
        source: {
          node_id: NOISE,
          field: 'noise',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'noise',
        },
      },
      {
        source: {
          node_id: INPAINT_IMAGE,
          field: 'latents',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'latents',
        },
      },
      {
        source: {
          node_id: INPAINT_CREATE_MASK,
          field: 'denoise_mask',
        },
        destination: {
          node_id: DENOISE_LATENTS,
          field: 'denoise_mask',
        },
      },
      // Decode Inpainted Latents To Image
      {
        source: {
          node_id: DENOISE_LATENTS,
          field: 'latents',
        },
        destination: {
          node_id: LATENTS_TO_IMAGE,
          field: 'latents',
        },
      },
    ],
  };

  // Handle Scale Before Processing
  if (isUsingScaledDimensions) {
    const scaledWidth: number = scaledBoundingBoxDimensions.width;
    const scaledHeight: number = scaledBoundingBoxDimensions.height;

    // Add Scaling Nodes
    graph.nodes[INPAINT_IMAGE_RESIZE_UP] = {
      type: 'img_resize',
      id: INPAINT_IMAGE_RESIZE_UP,
      is_intermediate,
      width: scaledWidth,
      height: scaledHeight,
      image: canvasInitImage,
    };
    graph.nodes[MASK_RESIZE_UP] = {
      type: 'img_resize',
      id: MASK_RESIZE_UP,
      is_intermediate,
      width: scaledWidth,
      height: scaledHeight,
      image: canvasMaskImage,
    };
    graph.nodes[INPAINT_IMAGE_RESIZE_DOWN] = {
      type: 'img_resize',
      id: INPAINT_IMAGE_RESIZE_DOWN,
      is_intermediate,
      width: width,
      height: height,
    };
    graph.nodes[MASK_RESIZE_DOWN] = {
      type: 'img_resize',
      id: MASK_RESIZE_DOWN,
      is_intermediate,
      width: width,
      height: height,
    };

    (graph.nodes[NOISE] as NoiseInvocation).width = scaledWidth;
    (graph.nodes[NOISE] as NoiseInvocation).height = scaledHeight;

    // Connect Nodes
    graph.edges.push(
      // Scale Inpaint Image and Mask
      {
        source: {
          node_id: INPAINT_IMAGE_RESIZE_UP,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_IMAGE,
          field: 'image',
        },
      },
      {
        source: {
          node_id: MASK_RESIZE_UP,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_CREATE_MASK,
          field: 'mask',
        },
      },
      // Resize Down
      {
        source: {
          node_id: LATENTS_TO_IMAGE,
          field: 'image',
        },
        destination: {
          node_id: INPAINT_IMAGE_RESIZE_DOWN,
          field: 'image',
        },
      },
      {
        source: {
          node_id: INPAINT_CREATE_MASK,
          field: 'expanded_mask_area',
        },
        destination: {
          node_id: MASK_RESIZE_DOWN,
          field: 'image',
        },
      },
      // Paste Back
      {
        source: {
          node_id: INPAINT_IMAGE_RESIZE_DOWN,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'target_image',
        },
      },
      {
        source: {
          node_id: MASK_RESIZE_DOWN,
          field: 'image',
        },
        destination: {
          node_id: CANVAS_OUTPUT,
          field: 'mask',
        },
      }
    );
  } else {
    // Add Images To Nodes
    (graph.nodes[NOISE] as NoiseInvocation).width = width;
    (graph.nodes[NOISE] as NoiseInvocation).height = height;

    graph.nodes[INPAINT_IMAGE] = {
      ...(graph.nodes[INPAINT_IMAGE] as ImageToLatentsInvocation),
      image: canvasInitImage,
    };

    graph.nodes[INPAINT_CREATE_MASK] = {
      ...(graph.nodes[INPAINT_CREATE_MASK] as CreateGradientMaskInvocation),
      mask: canvasMaskImage,
    };

    // Paste Back
    graph.nodes[CANVAS_OUTPUT] = {
      ...(graph.nodes[CANVAS_OUTPUT] as CanvasPasteBackInvocation),
      mask: canvasMaskImage,
    };

    graph.edges.push({
      source: {
        node_id: LATENTS_TO_IMAGE,
        field: 'image',
      },
      destination: {
        node_id: CANVAS_OUTPUT,
        field: 'target_image',
      },
    });
  }

  // Add Seamless To Graph
  if (seamlessXAxis || seamlessYAxis) {
    addSeamlessToLinearGraph(state, graph, modelLoaderNodeId);
    modelLoaderNodeId = SEAMLESS;
  }

  // Add VAE
  await addVAEToGraph(state, graph, modelLoaderNodeId);

  // add LoRA support
  await addLoRAsToGraph(state, graph, DENOISE_LATENTS, modelLoaderNodeId);

  // add controlnet, mutating `graph`
  await addControlNetToLinearGraph(state, graph, DENOISE_LATENTS);

  // Add IP Adapter
  await addIPAdapterToLinearGraph(state, graph, DENOISE_LATENTS);
  await addT2IAdaptersToLinearGraph(state, graph, DENOISE_LATENTS);
  // NSFW & watermark - must be last thing added to graph
  if (state.system.shouldUseNSFWChecker) {
    // must add before watermarker!
    addNSFWCheckerToGraph(state, graph, CANVAS_OUTPUT);
  }

  if (state.system.shouldUseWatermarker) {
    // must add after nsfw checker!
    addWatermarkerToGraph(state, graph, CANVAS_OUTPUT);
  }

  return graph;
};
