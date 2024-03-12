import { skipToken } from '@reduxjs/toolkit/query';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { isNil } from 'lodash-es';
import { useMemo } from 'react';
import { useGetModelConfigWithTypeGuard } from 'services/api/hooks/useGetModelConfigWithTypeGuard';
import { isNonRefinerMainModelConfig } from 'services/api/types';

const initialStatesSelector = createMemoizedSelector(selectConfigSlice, (config) => {
  const { steps, guidance, scheduler, cfgRescaleMultiplier, vaePrecision } = config.sd;

  return {
    initialSteps: steps.initial,
    initialCfg: guidance.initial,
    initialScheduler: scheduler,
    initialCfgRescaleMultiplier: cfgRescaleMultiplier.initial,
    initialVaePrecision: vaePrecision,
  };
});

export const useMainModelDefaultSettings = (modelKey?: string | null) => {
  const { modelConfig, isLoading } = useGetModelConfigWithTypeGuard(modelKey ?? skipToken, isNonRefinerMainModelConfig);

  const { initialSteps, initialCfg, initialScheduler, initialCfgRescaleMultiplier, initialVaePrecision } =
    useAppSelector(initialStatesSelector);

  const defaultSettingsDefaults = useMemo(() => {
    return {
      vae: {
        isEnabled: !isNil(modelConfig?.default_settings?.vae),
        value: modelConfig?.default_settings?.vae || 'default',
      },
      vaePrecision: {
        isEnabled: !isNil(modelConfig?.default_settings?.vae_precision),
        value: modelConfig?.default_settings?.vae_precision || initialVaePrecision || 'fp32',
      },
      scheduler: {
        isEnabled: !isNil(modelConfig?.default_settings?.scheduler),
        value: modelConfig?.default_settings?.scheduler || initialScheduler || 'euler',
      },
      steps: {
        isEnabled: !isNil(modelConfig?.default_settings?.steps),
        value: modelConfig?.default_settings?.steps || initialSteps,
      },
      cfgScale: {
        isEnabled: !isNil(modelConfig?.default_settings?.cfg_scale),
        value: modelConfig?.default_settings?.cfg_scale || initialCfg,
      },
      cfgRescaleMultiplier: {
        isEnabled: !isNil(modelConfig?.default_settings?.cfg_rescale_multiplier),
        value: modelConfig?.default_settings?.cfg_rescale_multiplier || initialCfgRescaleMultiplier,
      },
    };
  }, [
    modelConfig?.default_settings,
    initialSteps,
    initialCfg,
    initialScheduler,
    initialCfgRescaleMultiplier,
    initialVaePrecision,
  ]);

  return { defaultSettingsDefaults, isLoading };
};
