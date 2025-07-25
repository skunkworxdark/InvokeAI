import type { Dimensions } from 'features/controlLayers/store/types';
import type { components, paths } from 'services/api/schema';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import type { JsonObject, SetRequired } from 'type-fest';
import z from 'zod';

export type S = components['schemas'];

export type ListImagesArgs = NonNullable<paths['/api/v1/images/']['get']['parameters']['query']>;
export type ListImagesResponse = paths['/api/v1/images/']['get']['responses']['200']['content']['application/json'];

export type GetImageNamesResult =
  paths['/api/v1/images/names']['get']['responses']['200']['content']['application/json'];
export type GetImageNamesArgs = NonNullable<paths['/api/v1/images/names']['get']['parameters']['query']>;

export type ListBoardsArgs = NonNullable<paths['/api/v1/boards/']['get']['parameters']['query']>;

export type CreateBoardArg = paths['/api/v1/boards/']['post']['parameters']['query'];

export type UpdateBoardArg = paths['/api/v1/boards/{board_id}']['patch']['parameters']['path'] & {
  changes: paths['/api/v1/boards/{board_id}']['patch']['requestBody']['content']['application/json'];
};

export type GraphAndWorkflowResponse =
  paths['/api/v1/images/i/{image_name}/workflow']['get']['responses']['200']['content']['application/json'];

export type EnqueueBatchArg =
  paths['/api/v1/queue/{queue_id}/enqueue_batch']['post']['requestBody']['content']['application/json'];

export type InputFieldJSONSchemaExtra = S['InputFieldJSONSchemaExtra'];
export type OutputFieldJSONSchemaExtra = S['OutputFieldJSONSchemaExtra'];
export type InvocationJSONSchemaExtra = S['UIConfigBase'];

// App Info
export type AppVersion = S['AppVersion'];
export type AppConfig = S['AppConfig'];

const zResourceOrigin = z.enum(['internal', 'external']);
type ResourceOrigin = z.infer<typeof zResourceOrigin>;
assert<Equals<ResourceOrigin, S['ResourceOrigin']>>();
const zImageCategory = z.enum(['general', 'mask', 'control', 'user', 'other']);
export type ImageCategory = z.infer<typeof zImageCategory>;
assert<Equals<ImageCategory, S['ImageCategory']>>();

// Images
const _zImageDTO = z.object({
  image_name: z.string(),
  image_url: z.string(),
  thumbnail_url: z.string(),
  image_origin: zResourceOrigin,
  image_category: zImageCategory,
  width: z.number().int().gt(0),
  height: z.number().int().gt(0),
  created_at: z.string(),
  updated_at: z.string(),
  deleted_at: z.string().nullish(),
  is_intermediate: z.boolean(),
  session_id: z.string().nullish(),
  node_id: z.string().nullish(),
  starred: z.boolean(),
  has_workflow: z.boolean(),
  board_id: z.string().nullish(),
});
export type ImageDTO = z.infer<typeof _zImageDTO>;
assert<Equals<ImageDTO, S['ImageDTO']>>();

export type BoardDTO = S['BoardDTO'];
export type OffsetPaginatedResults_ImageDTO_ = S['OffsetPaginatedResults_ImageDTO_'];

// Models
export type ModelType = S['ModelType'];
export type BaseModelType = S['BaseModelType'];

// Model Configs

export type ControlLoRAModelConfig = S['ControlLoRALyCORISConfig'] | S['ControlLoRADiffusersConfig'];
// TODO(MM2): Can we make key required in the pydantic model?
export type LoRAModelConfig = S['LoRADiffusersConfig'] | S['LoRALyCORISConfig'] | S['LoRAOmiConfig'];
// TODO(MM2): Can we rename this from Vae -> VAE
export type VAEModelConfig = S['VAECheckpointConfig'] | S['VAEDiffusersConfig'];
export type ControlNetModelConfig = S['ControlNetDiffusersConfig'] | S['ControlNetCheckpointConfig'];
export type IPAdapterModelConfig = S['IPAdapterInvokeAIConfig'] | S['IPAdapterCheckpointConfig'];
export type T2IAdapterModelConfig = S['T2IAdapterConfig'];
export type CLIPLEmbedModelConfig = S['CLIPLEmbedDiffusersConfig'];
export type CLIPGEmbedModelConfig = S['CLIPGEmbedDiffusersConfig'];
export type CLIPEmbedModelConfig = CLIPLEmbedModelConfig | CLIPGEmbedModelConfig;
export type LlavaOnevisionConfig = S['LlavaOnevisionConfig'];
export type T5EncoderModelConfig = S['T5EncoderConfig'];
export type T5EncoderBnbQuantizedLlmInt8bModelConfig = S['T5EncoderBnbQuantizedLlmInt8bConfig'];
export type SpandrelImageToImageModelConfig = S['SpandrelImageToImageConfig'];
type TextualInversionModelConfig = S['TextualInversionFileConfig'] | S['TextualInversionFolderConfig'];
type DiffusersModelConfig = S['MainDiffusersConfig'];
export type CheckpointModelConfig = S['MainCheckpointConfig'];
type CLIPVisionDiffusersConfig = S['CLIPVisionDiffusersConfig'];
export type SigLipModelConfig = S['SigLIPConfig'];
export type FLUXReduxModelConfig = S['FluxReduxConfig'];
export type ApiModelConfig = S['ApiModelConfig'];
export type MainModelConfig = DiffusersModelConfig | CheckpointModelConfig | ApiModelConfig;
export type FLUXKontextModelConfig = MainModelConfig;
export type ChatGPT4oModelConfig = ApiModelConfig;
export type AnyModelConfig =
  | ControlLoRAModelConfig
  | LoRAModelConfig
  | VAEModelConfig
  | ControlNetModelConfig
  | IPAdapterModelConfig
  | T5EncoderModelConfig
  | T5EncoderBnbQuantizedLlmInt8bModelConfig
  | CLIPEmbedModelConfig
  | T2IAdapterModelConfig
  | SpandrelImageToImageModelConfig
  | TextualInversionModelConfig
  | MainModelConfig
  | CLIPVisionDiffusersConfig
  | SigLipModelConfig
  | FLUXReduxModelConfig
  | LlavaOnevisionConfig;

/**
 * Checks if a list of submodels contains any that match a given variant or type
 * @param submodels The list of submodels to check
 * @param checkStr The string to check against for variant or type
 * @returns A boolean
 */
const checkSubmodel = (submodels: AnyModelConfig['submodels'], checkStr: string): boolean => {
  for (const submodel in submodels) {
    if (
      submodel &&
      submodels[submodel] &&
      (submodels[submodel].model_type === checkStr || submodels[submodel].variant === checkStr)
    ) {
      return true;
    }
  }
  return false;
};

/**
 * Checks if a main model config has submodels that match a given variant or type
 * @param identifiers A list of strings to check against for variant or type in submodels
 * @param config The model config
 * @returns A boolean
 */
const checkSubmodels = (identifiers: string[], config: AnyModelConfig): boolean => {
  return identifiers.every(
    (identifier) =>
      config.type === 'main' &&
      config.submodels &&
      (identifier in config.submodels || checkSubmodel(config.submodels, identifier))
  );
};

export const isLoRAModelConfig = (config: AnyModelConfig): config is LoRAModelConfig => {
  return config.type === 'lora';
};

export const isControlLoRAModelConfig = (config: AnyModelConfig): config is ControlLoRAModelConfig => {
  return config.type === 'control_lora';
};

export const isVAEModelConfig = (config: AnyModelConfig, excludeSubmodels?: boolean): config is VAEModelConfig => {
  return config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config));
};

export const isNonFluxVAEModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    config.base !== 'flux'
  );
};

export const isFluxVAEModelConfig = (config: AnyModelConfig, excludeSubmodels?: boolean): config is VAEModelConfig => {
  return (
    (config.type === 'vae' || (!excludeSubmodels && config.type === 'main' && checkSubmodels(['vae'], config))) &&
    config.base === 'flux'
  );
};

export const isControlNetModelConfig = (config: AnyModelConfig): config is ControlNetModelConfig => {
  return config.type === 'controlnet';
};

export const isControlLayerModelConfig = (
  config: AnyModelConfig
): config is ControlNetModelConfig | T2IAdapterModelConfig | ControlLoRAModelConfig => {
  return config.type === 'controlnet' || config.type === 't2i_adapter' || config.type === 'control_lora';
};

export const isIPAdapterModelConfig = (config: AnyModelConfig): config is IPAdapterModelConfig => {
  return config.type === 'ip_adapter';
};

export const isCLIPVisionModelConfig = (config: AnyModelConfig): config is CLIPVisionDiffusersConfig => {
  return config.type === 'clip_vision';
};

export const isLLaVAModelConfig = (config: AnyModelConfig): config is LlavaOnevisionConfig => {
  return config.type === 'llava_onevision';
};

export const isT2IAdapterModelConfig = (config: AnyModelConfig): config is T2IAdapterModelConfig => {
  return config.type === 't2i_adapter';
};

export const isT5EncoderModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is T5EncoderModelConfig | T5EncoderBnbQuantizedLlmInt8bModelConfig => {
  return (
    config.type === 't5_encoder' ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['t5_encoder'], config))
  );
};

export const isCLIPEmbedModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is CLIPEmbedModelConfig => {
  return (
    config.type === 'clip_embed' ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['clip_embed'], config))
  );
};

export const isCLIPLEmbedModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is CLIPLEmbedModelConfig => {
  return (
    (config.type === 'clip_embed' && config.variant === 'large') ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['clip_embed', 'large'], config))
  );
};

export const isCLIPGEmbedModelConfig = (
  config: AnyModelConfig,
  excludeSubmodels?: boolean
): config is CLIPGEmbedModelConfig => {
  return (
    (config.type === 'clip_embed' && config.variant === 'gigantic') ||
    (!excludeSubmodels && config.type === 'main' && checkSubmodels(['clip_embed', 'gigantic'], config))
  );
};

export const isSpandrelImageToImageModelConfig = (
  config: AnyModelConfig
): config is SpandrelImageToImageModelConfig => {
  return config.type === 'spandrel_image_to_image';
};

export const isSigLipModelConfig = (config: AnyModelConfig): config is SigLipModelConfig => {
  return config.type === 'siglip';
};

export const isFluxReduxModelConfig = (config: AnyModelConfig): config is FLUXReduxModelConfig => {
  return config.type === 'flux_redux';
};

export const isChatGPT4oModelConfig = (config: AnyModelConfig): config is ChatGPT4oModelConfig => {
  return config.type === 'main' && config.base === 'chatgpt-4o';
};

export const isImagen3ModelConfig = (config: AnyModelConfig): config is ApiModelConfig => {
  return config.type === 'main' && config.base === 'imagen3';
};

export const isImagen4ModelConfig = (config: AnyModelConfig): config is ApiModelConfig => {
  return config.type === 'main' && config.base === 'imagen4';
};

export const isFluxKontextApiModelConfig = (config: AnyModelConfig): config is ApiModelConfig => {
  return config.type === 'main' && config.base === 'flux-kontext';
};

export const isFluxKontextModelConfig = (config: AnyModelConfig): config is FLUXKontextModelConfig => {
  return config.type === 'main' && config.base === 'flux' && config.name.toLowerCase().includes('kontext');
};

export const isNonRefinerMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base !== 'sdxl-refiner';
};

export const isCheckpointMainModelConfig = (config: AnyModelConfig): config is CheckpointModelConfig => {
  return config.type === 'main' && (config.format === 'checkpoint' || config.format === 'bnb_quantized_nf4b');
};

export const isRefinerMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'sdxl-refiner';
};

export const isSDXLMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'sdxl';
};

export const isSD3MainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'sd-3';
};

export const isCogView4MainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'cogview4';
};

export const isFluxMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'flux';
};

export const isFluxFillMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'flux' && config.variant === 'inpaint';
};

export const isNonSDXLMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && (config.base === 'sd-1' || config.base === 'sd-2');
};

export const isTIModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'embedding';
};

export type ModelInstallJob = S['ModelInstallJob'];
export type ModelInstallStatus = S['InstallStatus'];

// Graphs
export type Graph = S['Graph'];
export type NonNullableGraph = SetRequired<Graph, 'nodes' | 'edges'>;
export type Batch = S['Batch'];
export const zWorkflowRecordOrderBy = z.enum(['name', 'created_at', 'updated_at', 'opened_at']);
export type WorkflowRecordOrderBy = z.infer<typeof zWorkflowRecordOrderBy>;
assert<Equals<S['WorkflowRecordOrderBy'], WorkflowRecordOrderBy>>();

export const zSQLiteDirection = z.enum(['ASC', 'DESC']);
export type SQLiteDirection = z.infer<typeof zSQLiteDirection>;
assert<Equals<S['SQLiteDirection'], SQLiteDirection>>();
export type WorkflowRecordListItemWithThumbnailDTO = S['WorkflowRecordListItemWithThumbnailDTO'];

type KeysOfUnion<T> = T extends T ? keyof T : never;

export type AnyInvocation = Exclude<
  NonNullable<S['Graph']['nodes']>[string],
  S['CoreMetadataInvocation'] | S['MetadataInvocation'] | S['MetadataItemInvocation'] | S['MergeMetadataInvocation']
>;
export type AnyInvocationIncMetadata = NonNullable<S['Graph']['nodes']>[string];

export type InvocationType = AnyInvocation['type'];
type InvocationOutputMap = S['InvocationOutputMap'];
export type AnyInvocationOutput = InvocationOutputMap[InvocationType];

export type Invocation<T extends InvocationType> = Extract<AnyInvocation, { type: T }>;
// export type InvocationOutput<T extends InvocationType> = InvocationOutputMap[T];

type NonInputFields = 'id' | 'type' | 'is_intermediate' | 'use_cache' | 'board' | 'metadata';
export type AnyInvocationInputField = Exclude<KeysOfUnion<Required<AnyInvocation>>, NonInputFields>;
export type InputFields<T extends AnyInvocation> = Extract<keyof T, AnyInvocationInputField>;

type ExcludeIndexSignature<T> = {
  [K in keyof T as string extends K ? never : K]: T[K];
};

export type CoreMetadataFields = Exclude<
  keyof ExcludeIndexSignature<components['schemas']['CoreMetadataInvocation']>,
  NonInputFields
>;

type NonOutputFields = 'type';
export type AnyInvocationOutputField = Exclude<KeysOfUnion<Required<AnyInvocationOutput>>, NonOutputFields>;
export type OutputFields<T extends AnyInvocation> = Extract<
  keyof InvocationOutputMap[T['type']],
  AnyInvocationOutputField
>;

// Node Outputs
export type ImageOutput = S['ImageOutput'];

export type BoardRecordOrderBy = S['BoardRecordOrderBy'];
export type StarterModel = S['StarterModel'];

export type GetHFTokenStatusResponse =
  paths['/api/v2/models/hf_login']['get']['responses']['200']['content']['application/json'];
export type SetHFTokenResponse = NonNullable<
  paths['/api/v2/models/hf_login']['post']['responses']['200']['content']['application/json']
>;
export type ResetHFTokenResponse = NonNullable<
  paths['/api/v2/models/hf_login']['delete']['responses']['200']['content']['application/json']
>;
export type SetHFTokenArg = NonNullable<
  paths['/api/v2/models/hf_login']['post']['requestBody']['content']['application/json']
>;

export type UploadImageArg = {
  /**
   * The file object to upload
   */
  file: File;
  /**
   * THe category of image to upload
   */
  image_category: ImageCategory;
  /**
   * Whether the uploaded image is an intermediate image (intermediate images are not shown int he gallery)
   */
  is_intermediate: boolean;
  /**
   * The session with which to associate the uploaded image
   */
  session_id?: string;
  /**
   * The board id to add the image to
   */
  board_id?: string;
  /**
   * Whether or not to crop the image to its bounding box before saving
   */
  crop_visible?: boolean;
  /**
   * Metadata to embed in the image when saving it
   */
  metadata?: JsonObject;
  /**
   * Whether this upload should be "silent" (no toast on upload, no changing of gallery view)
   */
  silent?: boolean;
  /**
   * Whether this is the first upload of a batch (used when displaying user feedback with toasts - ignored if the upload is silent)
   */
  isFirstUploadOfBatch?: boolean;
  /**
   * If provided, the uploaded image will resized to the given dimensions.
   */
  resize_to?: Dimensions;
};

export type ImageUploadEntryResponse = S['ImageUploadEntry'];
export type ImageUploadEntryRequest = paths['/api/v1/images/']['post']['requestBody']['content']['application/json'];
