import type { RootState } from 'app/store/store';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { filter, size } from 'lodash-es';
import type { CoreMetadataInvocation, NonNullableGraph, SDXLLoRALoaderInvocation } from 'services/api/types';

import {
  LORA_LOADER,
  NEGATIVE_CONDITIONING,
  POSITIVE_CONDITIONING,
  SDXL_MODEL_LOADER,
  SDXL_REFINER_INPAINT_CREATE_MASK,
  SEAMLESS,
} from './constants';
import { upsertMetadata } from './metadata';

export const addSDXLLoRAsToGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string,
  modelLoaderNodeId: string = SDXL_MODEL_LOADER
): Promise<void> => {
  /**
   * LoRA nodes get the UNet and CLIP models from the main model loader and apply the LoRA to them.
   * They then output the UNet and CLIP models references on to either the next LoRA in the chain,
   * or to the inference/conditioning nodes.
   *
   * So we need to inject a LoRA chain into the graph.
   */

  // TODO(MM2): check base model
  const enabledLoRAs = filter(state.lora.loras, (l) => l.isEnabled ?? false);
  const loraCount = size(enabledLoRAs);

  if (loraCount === 0) {
    return;
  }

  const loraMetadata: CoreMetadataInvocation['loras'] = [];

  // Handle Seamless Plugs
  const unetLoaderId = modelLoaderNodeId;
  let clipLoaderId = modelLoaderNodeId;
  if ([SEAMLESS, SDXL_REFINER_INPAINT_CREATE_MASK].includes(modelLoaderNodeId)) {
    clipLoaderId = SDXL_MODEL_LOADER;
  }

  // Remove modelLoaderNodeId unet/clip/clip2 connections to feed it to LoRAs
  graph.edges = graph.edges.filter(
    (e) =>
      !(e.source.node_id === unetLoaderId && ['unet'].includes(e.source.field)) &&
      !(e.source.node_id === clipLoaderId && ['clip'].includes(e.source.field)) &&
      !(e.source.node_id === clipLoaderId && ['clip2'].includes(e.source.field))
  );

  // we need to remember the last lora so we can chain from it
  let lastLoraNodeId = '';
  let currentLoraIndex = 0;

  enabledLoRAs.forEach(async (lora) => {
    const { weight } = lora;
    const currentLoraNodeId = `${LORA_LOADER}_${lora.model.key}`;
    const parsedModel = zModelIdentifierField.parse(lora.model);

    const loraLoaderNode: SDXLLoRALoaderInvocation = {
      type: 'sdxl_lora_loader',
      id: currentLoraNodeId,
      is_intermediate: true,
      lora: parsedModel,
      weight,
    };

    loraMetadata.push({ model: parsedModel, weight });

    // add to graph
    graph.nodes[currentLoraNodeId] = loraLoaderNode;
    if (currentLoraIndex === 0) {
      // first lora = start the lora chain, attach directly to model loader
      graph.edges.push({
        source: {
          node_id: unetLoaderId,
          field: 'unet',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'unet',
        },
      });

      graph.edges.push({
        source: {
          node_id: clipLoaderId,
          field: 'clip',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'clip',
        },
      });

      graph.edges.push({
        source: {
          node_id: clipLoaderId,
          field: 'clip2',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'clip2',
        },
      });
    } else {
      // we are in the middle of the lora chain, instead connect to the previous lora
      graph.edges.push({
        source: {
          node_id: lastLoraNodeId,
          field: 'unet',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'unet',
        },
      });
      graph.edges.push({
        source: {
          node_id: lastLoraNodeId,
          field: 'clip',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'clip',
        },
      });

      graph.edges.push({
        source: {
          node_id: lastLoraNodeId,
          field: 'clip2',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'clip2',
        },
      });
    }

    if (currentLoraIndex === loraCount - 1) {
      // final lora, end the lora chain - we need to connect up to inference and conditioning nodes
      graph.edges.push({
        source: {
          node_id: currentLoraNodeId,
          field: 'unet',
        },
        destination: {
          node_id: baseNodeId,
          field: 'unet',
        },
      });

      graph.edges.push({
        source: {
          node_id: currentLoraNodeId,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      });

      graph.edges.push({
        source: {
          node_id: currentLoraNodeId,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      });

      graph.edges.push({
        source: {
          node_id: currentLoraNodeId,
          field: 'clip2',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip2',
        },
      });

      graph.edges.push({
        source: {
          node_id: currentLoraNodeId,
          field: 'clip2',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip2',
        },
      });
    }

    // increment the lora for the next one in the chain
    lastLoraNodeId = currentLoraNodeId;
    currentLoraIndex += 1;
  });

  upsertMetadata(graph, { loras: loraMetadata });
};
