import { CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectBrushShadowOpacity,
  settingsBrushShadowOpacityChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamBrushShadowOpacity = () => {
  const dispatch = useAppDispatch();
  const brushShadowOpacity = useAppSelector(selectBrushShadowOpacity);
  const { t } = useTranslation();

  const handleBrushShadowOpacityChange = useCallback(
    (v: number) => {
      dispatch(settingsBrushShadowOpacityChanged(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('controlLayers.brushShadowOpacity')}</FormLabel>
      <CompositeSlider
        min={0}
        max={1}
        step={0.01}
        value={brushShadowOpacity}
        onChange={handleBrushShadowOpacityChange}
        marks
      />
    </FormControl>
  );
};

export default memo(ParamBrushShadowOpacity);
