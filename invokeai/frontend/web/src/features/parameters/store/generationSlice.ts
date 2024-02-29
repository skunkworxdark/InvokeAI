import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { isAnyControlAdapterAdded } from 'features/controlAdapters/store/controlAdaptersSlice';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { initialAspectRatioState } from 'features/parameters/components/ImageSize/constants';
import type { AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { CLIP_SKIP_MAP } from 'features/parameters/types/constants';
import type {
  ParameterCanvasCoherenceMode,
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterMaskBlurMethod,
  ParameterModel,
  ParameterPrecision,
  ParameterScheduler,
  ParameterVAEModel,
} from 'features/parameters/types/parameterSchemas';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';
import { getIsSizeOptimal, getOptimalDimension } from 'features/parameters/util/optimalDimension';
import { configChanged } from 'features/system/store/configSlice';
import { clamp } from 'lodash-es';
import type { ImageDTO } from 'services/api/types';

import type { GenerationState } from './types';

export const initialGenerationState: GenerationState = {
  _version: 1,
  cfgScale: 7.5,
  cfgRescaleMultiplier: 0,
  height: 512,
  img2imgStrength: 0.75,
  infillMethod: 'patchmatch',
  iterations: 1,
  positivePrompt: '',
  negativePrompt: '',
  scheduler: 'euler',
  maskBlur: 16,
  maskBlurMethod: 'box',
  canvasCoherenceMode: 'unmasked',
  canvasCoherenceSteps: 20,
  canvasCoherenceStrength: 0.3,
  seed: 0,
  shouldFitToWidthHeight: true,
  shouldRandomizeSeed: true,
  steps: 50,
  infillTileSize: 32,
  infillPatchmatchDownscaleSize: 1,
  width: 512,
  model: null,
  vae: null,
  vaePrecision: 'fp32',
  seamlessXAxis: false,
  seamlessYAxis: false,
  clipSkip: 0,
  shouldUseCpuNoise: true,
  shouldShowAdvancedOptions: false,
  aspectRatio: { ...initialAspectRatioState },
};

export const generationSlice = createSlice({
  name: 'generation',
  initialState: initialGenerationState,
  reducers: {
    setPositivePrompt: (state, action: PayloadAction<string>) => {
      state.positivePrompt = action.payload;
    },
    setNegativePrompt: (state, action: PayloadAction<string>) => {
      state.negativePrompt = action.payload;
    },
    setIterations: (state, action: PayloadAction<number>) => {
      state.iterations = action.payload;
    },
    setSteps: (state, action: PayloadAction<number>) => {
      state.steps = action.payload;
    },
    setCfgScale: (state, action: PayloadAction<ParameterCFGScale>) => {
      state.cfgScale = action.payload;
    },
    setCfgRescaleMultiplier: (state, action: PayloadAction<ParameterCFGRescaleMultiplier>) => {
      state.cfgRescaleMultiplier = action.payload;
    },
    setScheduler: (state, action: PayloadAction<ParameterScheduler>) => {
      state.scheduler = action.payload;
    },
    setSeed: (state, action: PayloadAction<number>) => {
      state.seed = action.payload;
      state.shouldRandomizeSeed = false;
    },
    setImg2imgStrength: (state, action: PayloadAction<number>) => {
      state.img2imgStrength = action.payload;
    },
    setSeamlessXAxis: (state, action: PayloadAction<boolean>) => {
      state.seamlessXAxis = action.payload;
    },
    setSeamlessYAxis: (state, action: PayloadAction<boolean>) => {
      state.seamlessYAxis = action.payload;
    },
    setShouldFitToWidthHeight: (state, action: PayloadAction<boolean>) => {
      state.shouldFitToWidthHeight = action.payload;
    },
    resetSeed: (state) => {
      state.seed = -1;
    },
    resetParametersState: (state) => {
      return {
        ...state,
        ...initialGenerationState,
      };
    },
    setShouldRandomizeSeed: (state, action: PayloadAction<boolean>) => {
      state.shouldRandomizeSeed = action.payload;
    },
    clearInitialImage: (state) => {
      state.initialImage = undefined;
    },
    setMaskBlur: (state, action: PayloadAction<number>) => {
      state.maskBlur = action.payload;
    },
    setMaskBlurMethod: (state, action: PayloadAction<ParameterMaskBlurMethod>) => {
      state.maskBlurMethod = action.payload;
    },
    setCanvasCoherenceMode: (state, action: PayloadAction<ParameterCanvasCoherenceMode>) => {
      state.canvasCoherenceMode = action.payload;
    },
    setCanvasCoherenceSteps: (state, action: PayloadAction<number>) => {
      state.canvasCoherenceSteps = action.payload;
    },
    setCanvasCoherenceStrength: (state, action: PayloadAction<number>) => {
      state.canvasCoherenceStrength = action.payload;
    },
    setInfillMethod: (state, action: PayloadAction<string>) => {
      state.infillMethod = action.payload;
    },
    setInfillTileSize: (state, action: PayloadAction<number>) => {
      state.infillTileSize = action.payload;
    },
    setInfillPatchmatchDownscaleSize: (state, action: PayloadAction<number>) => {
      state.infillPatchmatchDownscaleSize = action.payload;
    },
    initialImageChanged: (state, action: PayloadAction<ImageDTO>) => {
      const { image_name, width, height } = action.payload;
      state.initialImage = { imageName: image_name, width, height };
    },
    modelChanged: {
      reducer: (
        state,
        action: PayloadAction<ParameterModel | null, string, { previousModel?: ParameterModel | null }>
      ) => {
        const newModel = action.payload;
        state.model = newModel;

        if (newModel === null) {
          return;
        }

        // Clamp ClipSkip Based On Selected Model
        // TODO(psyche): remove this special handling when https://github.com/invoke-ai/InvokeAI/issues/4583 is resolved
        // WIP PR here: https://github.com/invoke-ai/InvokeAI/pull/4624
        if (newModel.base_model === 'sdxl') {
          // We don't support clip skip for SDXL yet - it's not in the graphs
          state.clipSkip = 0;
        } else {
          const { maxClip } = CLIP_SKIP_MAP[newModel.base_model];
          state.clipSkip = clamp(state.clipSkip, 0, maxClip);
        }

        if (action.meta.previousModel?.base_model === newModel.base_model) {
          // The base model hasn't changed, we don't need to optimize the size
          return;
        }

        const optimalDimension = getOptimalDimension(newModel);
        if (getIsSizeOptimal(state.width, state.height, optimalDimension)) {
          return;
        }
        const { width, height } = calculateNewSize(state.aspectRatio.value, optimalDimension * optimalDimension);
        state.width = width;
        state.height = height;
      },
      prepare: (payload: ParameterModel | null, previousModel?: ParameterModel | null) => ({
        payload,
        meta: {
          previousModel,
        },
      }),
    },
    vaeSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
      // null is a valid VAE!
      state.vae = action.payload;
    },
    vaePrecisionChanged: (state, action: PayloadAction<ParameterPrecision>) => {
      state.vaePrecision = action.payload;
    },
    setClipSkip: (state, action: PayloadAction<number>) => {
      state.clipSkip = action.payload;
    },
    shouldUseCpuNoiseChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldUseCpuNoise = action.payload;
    },
    widthChanged: (state, action: PayloadAction<number>) => {
      state.width = action.payload;
    },
    heightChanged: (state, action: PayloadAction<number>) => {
      state.height = action.payload;
    },
    widthRecalled: (state, action: PayloadAction<number>) => {
      state.width = action.payload;
      state.aspectRatio.value = action.payload / state.height;
      state.aspectRatio.id = 'Free';
      state.aspectRatio.isLocked = false;
    },
    heightRecalled: (state, action: PayloadAction<number>) => {
      state.height = action.payload;
      state.aspectRatio.value = state.width / action.payload;
      state.aspectRatio.id = 'Free';
      state.aspectRatio.isLocked = false;
    },
    aspectRatioChanged: (state, action: PayloadAction<AspectRatioState>) => {
      state.aspectRatio = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(configChanged, (state, action) => {
      const defaultModel = action.payload.sd?.defaultModel;

      if (defaultModel && !state.model) {
        const [base_model, model_type, model_name] = defaultModel.split('/');

        const result = zParameterModel.safeParse({
          model_name,
          base_model,
          model_type,
        });

        if (result.success) {
          state.model = result.data;

          const optimalDimension = getOptimalDimension(result.data);

          state.width = optimalDimension;
          state.height = optimalDimension;
        }
      }
    });

    // TODO: This is a temp fix to reduce issues with T2I adapter having a different downscaling
    // factor than the UNet. Hopefully we get an upstream fix in diffusers.
    builder.addMatcher(isAnyControlAdapterAdded, (state, action) => {
      if (action.payload.type === 't2i_adapter') {
        state.width = roundToMultiple(state.width, 64);
        state.height = roundToMultiple(state.height, 64);
      }
    });
  },
  selectors: {
    selectOptimalDimension: (slice) => getOptimalDimension(slice.model),
  },
});

export const {
  clearInitialImage,
  resetParametersState,
  resetSeed,
  setCfgScale,
  setCfgRescaleMultiplier,
  setImg2imgStrength,
  setInfillMethod,
  setIterations,
  setPositivePrompt,
  setNegativePrompt,
  setScheduler,
  setMaskBlur,
  setMaskBlurMethod,
  setCanvasCoherenceMode,
  setCanvasCoherenceSteps,
  setCanvasCoherenceStrength,
  setSeed,
  setShouldFitToWidthHeight,
  setShouldRandomizeSeed,
  setSteps,
  setInfillTileSize,
  setInfillPatchmatchDownscaleSize,
  initialImageChanged,
  modelChanged,
  vaeSelected,
  setSeamlessXAxis,
  setSeamlessYAxis,
  setClipSkip,
  shouldUseCpuNoiseChanged,
  vaePrecisionChanged,
  aspectRatioChanged,
  widthChanged,
  heightChanged,
  widthRecalled,
  heightRecalled,
} = generationSlice.actions;

export const { selectOptimalDimension } = generationSlice.selectors;

export const selectGenerationSlice = (state: RootState) => state.generation;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateGenerationState = (state: any): GenerationState => {
  if (!('_version' in state)) {
    state._version = 1;
    state.aspectRatio = initialAspectRatioState;
  }
  return state;
};

export const generationPersistConfig: PersistConfig<GenerationState> = {
  name: generationSlice.name,
  initialState: initialGenerationState,
  migrate: migrateGenerationState,
  persistDenylist: [],
};
