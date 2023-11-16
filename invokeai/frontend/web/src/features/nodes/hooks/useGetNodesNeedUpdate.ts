import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { getNeedsUpdate } from './useNodeVersion';

const selector = createSelector(
  stateSelector,
  (state) => {
    const nodes = state.nodes.nodes;
    const templates = state.nodes.nodeTemplates;

    const needsUpdate = nodes.some((node) => {
      const template = templates[node.data.type];
      return getNeedsUpdate(node, template);
    });
    return needsUpdate;
  },
  defaultSelectorOptions
);

export const useGetNodesNeedUpdate = () => {
  const getNeedsUpdate = useAppSelector(selector);
  return getNeedsUpdate;
};
