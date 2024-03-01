import type { EntityState } from '@reduxjs/toolkit';
import type {
  ParameterControlNetModel,
  ParameterIPAdapterModel,
  ParameterT2IAdapterModel,
} from 'features/parameters/types/parameterSchemas';
import type { components } from 'services/api/schema';
import type {
  CannyImageProcessorInvocation,
  ColorMapImageProcessorInvocation,
  ContentShuffleImageProcessorInvocation,
  DepthAnythingImageProcessorInvocation,
  DWOpenposeImageProcessorInvocation,
  HedImageProcessorInvocation,
  LineartAnimeImageProcessorInvocation,
  LineartImageProcessorInvocation,
  MediapipeFaceProcessorInvocation,
  MidasDepthImageProcessorInvocation,
  MlsdImageProcessorInvocation,
  NormalbaeImageProcessorInvocation,
  PidiImageProcessorInvocation,
  ZoeDepthImageProcessorInvocation,
} from 'services/api/types';
import type { O } from 'ts-toolbelt';
import { z } from 'zod';

/**
 * Any ControlNet processor node
 */
export type ControlAdapterProcessorNode =
  | CannyImageProcessorInvocation
  | ColorMapImageProcessorInvocation
  | ContentShuffleImageProcessorInvocation
  | DepthAnythingImageProcessorInvocation
  | HedImageProcessorInvocation
  | LineartAnimeImageProcessorInvocation
  | LineartImageProcessorInvocation
  | MediapipeFaceProcessorInvocation
  | MidasDepthImageProcessorInvocation
  | MlsdImageProcessorInvocation
  | NormalbaeImageProcessorInvocation
  | DWOpenposeImageProcessorInvocation
  | PidiImageProcessorInvocation
  | ZoeDepthImageProcessorInvocation;

/**
 * Any ControlNet processor type
 */
export type ControlAdapterProcessorType = NonNullable<ControlAdapterProcessorNode['type'] | 'none'>;

/**
 * The Canny processor node, with parameters flagged as required
 */
export type RequiredCannyImageProcessorInvocation = O.Required<
  CannyImageProcessorInvocation,
  'type' | 'low_threshold' | 'high_threshold'
>;

/**
 * The Color Map processor node, with parameters flagged as required
 */
export type RequiredColorMapImageProcessorInvocation = O.Required<
  ColorMapImageProcessorInvocation,
  'type' | 'color_map_tile_size'
>;

/**
 * The ContentShuffle processor node, with parameters flagged as required
 */
export type RequiredContentShuffleImageProcessorInvocation = O.Required<
  ContentShuffleImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'w' | 'h' | 'f'
>;

/**
 * The DepthAnything processor node, with parameters flagged as required
 */
export type RequiredDepthAnythingImageProcessorInvocation = O.Required<
  DepthAnythingImageProcessorInvocation,
  'type' | 'model_size' | 'resolution' | 'offload'
>;

const zDepthAnythingModelSize = z.enum(['large', 'base', 'small']);
export type DepthAnythingModelSize = z.infer<typeof zDepthAnythingModelSize>;
export const isDepthAnythingModelSize = (v: unknown): v is DepthAnythingModelSize =>
  zDepthAnythingModelSize.safeParse(v).success;

/**
 * The HED processor node, with parameters flagged as required
 */
export type RequiredHedImageProcessorInvocation = O.Required<
  HedImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'scribble'
>;

/**
 * The Lineart Anime processor node, with parameters flagged as required
 */
export type RequiredLineartAnimeImageProcessorInvocation = O.Required<
  LineartAnimeImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution'
>;

/**
 * The Lineart processor node, with parameters flagged as required
 */
export type RequiredLineartImageProcessorInvocation = O.Required<
  LineartImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'coarse'
>;

/**
 * The MediapipeFace processor node, with parameters flagged as required
 */
export type RequiredMediapipeFaceProcessorInvocation = O.Required<
  MediapipeFaceProcessorInvocation,
  'type' | 'max_faces' | 'min_confidence'
>;

/**
 * The MidasDepth processor node, with parameters flagged as required
 */
export type RequiredMidasDepthImageProcessorInvocation = O.Required<
  MidasDepthImageProcessorInvocation,
  'type' | 'a_mult' | 'bg_th'
>;

/**
 * The MLSD processor node, with parameters flagged as required
 */
export type RequiredMlsdImageProcessorInvocation = O.Required<
  MlsdImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'thr_v' | 'thr_d'
>;

/**
 * The NormalBae processor node, with parameters flagged as required
 */
export type RequiredNormalbaeImageProcessorInvocation = O.Required<
  NormalbaeImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution'
>;

/**
 * The DW Openpose processor node, with parameters flagged as required
 */
export type RequiredDWOpenposeImageProcessorInvocation = O.Required<
  DWOpenposeImageProcessorInvocation,
  'type' | 'image_resolution' | 'draw_body' | 'draw_face' | 'draw_hands'
>;

/**
 * The Pidi processor node, with parameters flagged as required
 */
export type RequiredPidiImageProcessorInvocation = O.Required<
  PidiImageProcessorInvocation,
  'type' | 'detect_resolution' | 'image_resolution' | 'safe' | 'scribble'
>;

/**
 * The ZoeDepth processor node, with parameters flagged as required
 */
export type RequiredZoeDepthImageProcessorInvocation = O.Required<ZoeDepthImageProcessorInvocation, 'type'>;

/**
 * Any ControlNet Processor node, with its parameters flagged as required
 */
export type RequiredControlAdapterProcessorNode =
  | O.Required<
      | RequiredCannyImageProcessorInvocation
      | RequiredColorMapImageProcessorInvocation
      | RequiredContentShuffleImageProcessorInvocation
      | RequiredDepthAnythingImageProcessorInvocation
      | RequiredHedImageProcessorInvocation
      | RequiredLineartAnimeImageProcessorInvocation
      | RequiredLineartImageProcessorInvocation
      | RequiredMediapipeFaceProcessorInvocation
      | RequiredMidasDepthImageProcessorInvocation
      | RequiredMlsdImageProcessorInvocation
      | RequiredNormalbaeImageProcessorInvocation
      | RequiredDWOpenposeImageProcessorInvocation
      | RequiredPidiImageProcessorInvocation
      | RequiredZoeDepthImageProcessorInvocation,
      'id'
    >
  | { type: 'none' };

export type ControlMode = NonNullable<components['schemas']['ControlNetInvocation']['control_mode']>;

const zResizeMode = z.enum(['just_resize', 'crop_resize', 'fill_resize', 'just_resize_simple']);
export type ResizeMode = z.infer<typeof zResizeMode>;
export const isResizeMode = (v: unknown): v is ResizeMode => zResizeMode.safeParse(v).success;

export type ControlNetConfig = {
  type: 'controlnet';
  id: string;
  isEnabled: boolean;
  model: ParameterControlNetModel | null;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
  controlMode: ControlMode;
  resizeMode: ResizeMode;
  controlImage: string | null;
  processedControlImage: string | null;
  processorType: ControlAdapterProcessorType;
  processorNode: RequiredControlAdapterProcessorNode;
  shouldAutoConfig: boolean;
};

export type T2IAdapterConfig = {
  type: 't2i_adapter';
  id: string;
  isEnabled: boolean;
  model: ParameterT2IAdapterModel | null;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
  resizeMode: ResizeMode;
  controlImage: string | null;
  processedControlImage: string | null;
  processorType: ControlAdapterProcessorType;
  processorNode: RequiredControlAdapterProcessorNode;
  shouldAutoConfig: boolean;
};

export type IPAdapterConfig = {
  type: 'ip_adapter';
  id: string;
  isEnabled: boolean;
  controlImage: string | null;
  model: ParameterIPAdapterModel | null;
  weight: number;
  beginStepPct: number;
  endStepPct: number;
};

export type ControlAdapterConfig = ControlNetConfig | IPAdapterConfig | T2IAdapterConfig;

export type ControlAdapterType = ControlAdapterConfig['type'];

export type ControlAdaptersState = EntityState<ControlAdapterConfig, string> & {
  pendingControlImages: string[];
};

export const isControlNet = (controlAdapter: ControlAdapterConfig): controlAdapter is ControlNetConfig => {
  return controlAdapter.type === 'controlnet';
};

export const isIPAdapter = (controlAdapter: ControlAdapterConfig): controlAdapter is IPAdapterConfig => {
  return controlAdapter.type === 'ip_adapter';
};

export const isT2IAdapter = (controlAdapter: ControlAdapterConfig): controlAdapter is T2IAdapterConfig => {
  return controlAdapter.type === 't2i_adapter';
};

export const isControlNetOrT2IAdapter = (
  controlAdapter: ControlAdapterConfig
): controlAdapter is ControlNetConfig | T2IAdapterConfig => {
  return isControlNet(controlAdapter) || isT2IAdapter(controlAdapter);
};
