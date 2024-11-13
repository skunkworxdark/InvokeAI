import type { CanvasReferenceImageState } from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { BaseModelType, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

type AddIPAdaptersResult = {
  addedIPAdapters: number;
};

export const addIPAdapters = (
  ipAdapters: CanvasReferenceImageState[],
  g: Graph,
  collector: Invocation<'collect'>,
  base: BaseModelType
): AddIPAdaptersResult => {
  const validIPAdapters = ipAdapters.filter((entity) => isValidIPAdapter(entity, base));

  const result: AddIPAdaptersResult = {
    addedIPAdapters: 0,
  };

  for (const ipa of validIPAdapters) {
    result.addedIPAdapters++;

    addIPAdapter(ipa, g, collector);
  }

  return result;
};

const addIPAdapter = (entity: CanvasReferenceImageState, g: Graph, collector: Invocation<'collect'>) => {
  const { id, ipAdapter } = entity;
  const { weight, model, clipVisionModel, method, beginEndStepPct, image } = ipAdapter;
  assert(image, 'IP Adapter image is required');
  assert(model, 'IP Adapter model is required');

  let ipAdapterNode: Invocation<'flux_ip_adapter' | 'ip_adapter'>;

  if (model.base === 'flux') {
    assert(
      clipVisionModel === 'ViT-L',
      `ViT-L is the only supported CLIP Vision model for FLUX IP adapter, got ${clipVisionModel}`
    );
    ipAdapterNode = g.addNode({
      id: `ip_adapter_${id}`,
      type: 'flux_ip_adapter',
      weight,
      ip_adapter_model: model,
      clip_vision_model: clipVisionModel,
      begin_step_percent: beginEndStepPct[0],
      end_step_percent: beginEndStepPct[1],
      image: {
        image_name: image.image_name,
      },
    });
  } else {
    // model.base === SD1.5 or SDXL
    assert(
      clipVisionModel === 'ViT-H' || clipVisionModel === 'ViT-G',
      'ViT-G and ViT-H are the only supported CLIP Vision models for SD1.5 and SDXL IP adapters'
    );
    ipAdapterNode = g.addNode({
      id: `ip_adapter_${id}`,
      type: 'ip_adapter',
      weight,
      method,
      ip_adapter_model: model,
      clip_vision_model: clipVisionModel,
      begin_step_percent: beginEndStepPct[0],
      end_step_percent: beginEndStepPct[1],
      image: {
        image_name: image.image_name,
      },
    });
  }

  g.addEdge(ipAdapterNode, 'ip_adapter', collector, 'item');
};

const isValidIPAdapter = ({ isEnabled, ipAdapter }: CanvasReferenceImageState, base: BaseModelType): boolean => {
  // Must be have a model that matches the current base and must have a control image
  const hasModel = Boolean(ipAdapter.model);
  const modelMatchesBase = ipAdapter.model?.base === base;
  const hasImage = Boolean(ipAdapter.image);
  return isEnabled && hasModel && modelMatchesBase && hasImage;
};
