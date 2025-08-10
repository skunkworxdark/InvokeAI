import { CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectBrushSoftness, settingsBrushSoftnessChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamBrushSoftness = () => {
  const dispatch = useAppDispatch();
  const brushSoftness = useAppSelector(selectBrushSoftness);
  const { t } = useTranslation();

  const handleBrushSoftnessChange = useCallback(
    (v: number) => {
      dispatch(settingsBrushSoftnessChanged(v));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('controlLayers.brushSoftness')}</FormLabel>
      <CompositeSlider
        min={0}
        max={100}
        step={1}
        value={brushSoftness}
        onChange={handleBrushSoftnessChange}
        marks
      />
    </FormControl>
  );
};

export default memo(ParamBrushSoftness);
