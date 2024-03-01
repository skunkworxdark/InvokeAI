import type { RootState } from 'app/store/store';
import { selectValidIPAdapters } from 'features/controlAdapters/store/controlAdaptersSlice';
import { omit } from 'lodash-es';
import type {
  CollectInvocation,
  CoreMetadataInvocation,
  IPAdapterInvocation,
  IPAdapterMetadataField,
  NonNullableGraph,
} from 'services/api/types';

import { IP_ADAPTER_COLLECT } from './constants';
import { upsertMetadata } from './metadata';

export const addIPAdapterToLinearGraph = (state: RootState, graph: NonNullableGraph, baseNodeId: string): void => {
  const validIPAdapters = selectValidIPAdapters(state.controlAdapters).filter(
    (ca) => ca.model?.base === state.generation.model?.base
  );

  if (validIPAdapters.length) {
    // Even though denoise_latents' ip adapter input is collection or scalar, keep it simple and always use a collect
    const ipAdapterCollectNode: CollectInvocation = {
      id: IP_ADAPTER_COLLECT,
      type: 'collect',
      is_intermediate: true,
    };
    graph.nodes[IP_ADAPTER_COLLECT] = ipAdapterCollectNode;
    graph.edges.push({
      source: { node_id: IP_ADAPTER_COLLECT, field: 'collection' },
      destination: {
        node_id: baseNodeId,
        field: 'ip_adapter',
      },
    });

    const ipAdapterMetdata: CoreMetadataInvocation['ipAdapters'] = [];

    validIPAdapters.forEach((ipAdapter) => {
      if (!ipAdapter.model) {
        return;
      }
      const { id, weight, model, beginStepPct, endStepPct } = ipAdapter;
      const ipAdapterNode: IPAdapterInvocation = {
        id: `ip_adapter_${id}`,
        type: 'ip_adapter',
        is_intermediate: true,
        weight: weight,
        ip_adapter_model: model,
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
      };

      if (ipAdapter.controlImage) {
        ipAdapterNode.image = {
          image_name: ipAdapter.controlImage,
        };
      } else {
        return;
      }

      graph.nodes[ipAdapterNode.id] = ipAdapterNode as IPAdapterInvocation;

      ipAdapterMetdata.push(omit(ipAdapterNode, ['id', 'type', 'is_intermediate']) as IPAdapterMetadataField);

      graph.edges.push({
        source: { node_id: ipAdapterNode.id, field: 'ip_adapter' },
        destination: {
          node_id: ipAdapterCollectNode.id,
          field: 'item',
        },
      });
    });

    upsertMetadata(graph, { ipAdapters: ipAdapterMetdata });
  }
};
