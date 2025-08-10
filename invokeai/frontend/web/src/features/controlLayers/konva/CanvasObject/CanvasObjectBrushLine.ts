import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasBrushLineState } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';

export class CanvasObjectBrushLine extends CanvasModuleBase {
  readonly type = 'object_brush_line';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer;
  readonly manager: CanvasManager;
  readonly log: Logger;

  state: CanvasBrushLineState;
  konva: {
    group: Konva.Group;
    line: Konva.Line;
  };

  constructor(state: CanvasBrushLineState, parent: CanvasEntityObjectRenderer | CanvasEntityBufferObjectRenderer) {
    super();
    const { id, clip } = state;
    this.id = id;
    this.parent = parent;
    this.manager = parent.manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug({ state }, 'Creating module');

    this.konva = {
      group: new Konva.Group({
        name: `${this.type}:group`,
        clip,
        listening: false,
      }),
      line: new Konva.Line({
        name: `${this.type}:line`,
        listening: false,
        shadowForStrokeEnabled: false,
        tension: 0.3,
        lineCap: 'round',
        lineJoin: 'round',
        globalCompositeOperation: 'source-over',
        perfectDrawEnabled: false,
      }),
    };
    this.konva.group.add(this.konva.line);
    this.state = state;
  }

  update(state: CanvasBrushLineState, force = false): boolean {
    if (force || this.state !== state) {
      this.log.trace({ state }, 'Updating brush line');
      const { points, color, strokeWidth } = state;
      const { brushSoftness } = this.manager.store.getState().canvasSettings;

      if (brushSoftness > 0) {
        // A soft brush is a gradient from opaque to transparent.
        const softnessRatio = brushSoftness / 100;
        // The gradient stop for the solid center of the brush.
        // At 0 softness, the solid part is the whole brush (stop at 1).
        // As softness increases, the solid part shrinks (stop moves towards 0).
        const solidCenterStop = 1 - softnessRatio;
        const opaqueColor = rgbaColorToString(color);
        const transparentColor = rgbaColorToString({ ...color, a: 0 });

        this.konva.line.setAttrs({
          points: points.length === 2 ? [...points, ...points] : points,
          strokeWidth,
          strokeLinearGradientColorStops: [0, opaqueColor, solidCenterStop, opaqueColor, 1, transparentColor],
          // The gradient is defined as being perpendicular to the line. Konva handles the rotation.
          strokeLinearGradientStartPoint: { x: 0, y: -strokeWidth / 2 },
          strokeLinearGradientEndPoint: { x: 0, y: strokeWidth / 2 },
          // Gradients do not need shadows.
          shadowForStrokeEnabled: false,
          // We must clear the stroke, as it is replaced by the gradient.
          stroke: undefined,
        });
      } else {
        // A hard brush is a simple solid-color stroke with no shadow.
        this.konva.line.setAttrs({
          points: points.length === 2 ? [...points, ...points] : points,
          stroke: rgbaColorToString(color),
          strokeWidth,
          strokeLinearGradientColorStops: [], // Clear any existing gradient
          shadowForStrokeEnabled: false,
        });
      }
      this.state = state;
      return true;
    }

    return false;
  }

  setVisibility(isVisible: boolean): void {
    this.log.trace({ isVisible }, 'Setting brush line visibility');
    this.konva.group.visible(isVisible);
  }

  destroy = () => {
    this.log.debug('Destroying module');
    this.konva.group.destroy();
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      parent: this.parent.id,
      state: deepClone(this.state),
    };
  };
}
