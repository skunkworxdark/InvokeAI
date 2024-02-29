import { createAction, isAnyOf } from '@reduxjs/toolkit';
import type { WorkflowV2 } from 'features/nodes/types/workflow';
import type { Graph } from 'services/api/types';

export const textToImageGraphBuilt = createAction<Graph>('nodes/textToImageGraphBuilt');
export const imageToImageGraphBuilt = createAction<Graph>('nodes/imageToImageGraphBuilt');
export const canvasGraphBuilt = createAction<Graph>('nodes/canvasGraphBuilt');
export const nodesGraphBuilt = createAction<Graph>('nodes/nodesGraphBuilt');

export const isAnyGraphBuilt = isAnyOf(
  textToImageGraphBuilt,
  imageToImageGraphBuilt,
  canvasGraphBuilt,
  nodesGraphBuilt
);

export const workflowLoadRequested = createAction<{
  workflow: unknown;
  asCopy: boolean;
}>('nodes/workflowLoadRequested');

export const updateAllNodesRequested = createAction('nodes/updateAllNodesRequested');

export const workflowLoaded = createAction<WorkflowV2>('workflow/workflowLoaded');
