import { useStore } from '@nanostores/react';
import ParamBrushSoftness from 'features/controlLayers/components/Tool/ToolBrushSoftness';
import { ToolBrushWidth } from 'features/controlLayers/components/Tool/ToolBrushWidth';
import { ToolEraserWidth } from 'features/controlLayers/components/Tool/ToolEraserWidth';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo } from 'react';

export const ToolSettings = memo(() => {
  const canvasManager = useCanvasManager();
  const tool = useStore(canvasManager.tool.$tool);
  if (tool === 'brush') {
    return (
      <>
        <ToolBrushWidth />
        <ParamBrushSoftness />
      </>
    );
  }
  if (tool === 'eraser') {
    return <ToolEraserWidth />;
  }
  return null;
});

ToolSettings.displayName = 'ToolSettings';
