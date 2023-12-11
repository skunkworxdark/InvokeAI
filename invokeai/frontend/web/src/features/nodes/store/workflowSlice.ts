import { PayloadAction, createSlice } from '@reduxjs/toolkit';
import { workflowLoaded } from 'features/nodes/store/actions';
import {
  nodeEditorReset,
  nodesDeleted,
  isAnyNodeOrEdgeMutation,
} from 'features/nodes/store/nodesSlice';
import { WorkflowsState as WorkflowState } from 'features/nodes/store/types';
import { FieldIdentifier } from 'features/nodes/types/field';
import { cloneDeep, isEqual, uniqBy } from 'lodash-es';

export const initialWorkflowState: WorkflowState = {
  name: '',
  author: '',
  description: '',
  version: '',
  contact: '',
  tags: '',
  notes: '',
  exposedFields: [],
  meta: { version: '2.0.0', category: 'user' },
  isTouched: true,
};

const workflowSlice = createSlice({
  name: 'workflow',
  initialState: initialWorkflowState,
  reducers: {
    workflowExposedFieldAdded: (
      state,
      action: PayloadAction<FieldIdentifier>
    ) => {
      state.exposedFields = uniqBy(
        state.exposedFields.concat(action.payload),
        (field) => `${field.nodeId}-${field.fieldName}`
      );
      state.isTouched = true;
    },
    workflowExposedFieldRemoved: (
      state,
      action: PayloadAction<FieldIdentifier>
    ) => {
      state.exposedFields = state.exposedFields.filter(
        (field) => !isEqual(field, action.payload)
      );
      state.isTouched = true;
    },
    workflowNameChanged: (state, action: PayloadAction<string>) => {
      state.name = action.payload;
      state.isTouched = true;
    },
    workflowDescriptionChanged: (state, action: PayloadAction<string>) => {
      state.description = action.payload;
      state.isTouched = true;
    },
    workflowTagsChanged: (state, action: PayloadAction<string>) => {
      state.tags = action.payload;
      state.isTouched = true;
    },
    workflowAuthorChanged: (state, action: PayloadAction<string>) => {
      state.author = action.payload;
      state.isTouched = true;
    },
    workflowNotesChanged: (state, action: PayloadAction<string>) => {
      state.notes = action.payload;
      state.isTouched = true;
    },
    workflowVersionChanged: (state, action: PayloadAction<string>) => {
      state.version = action.payload;
      state.isTouched = true;
    },
    workflowContactChanged: (state, action: PayloadAction<string>) => {
      state.contact = action.payload;
      state.isTouched = true;
    },
    workflowIDChanged: (state, action: PayloadAction<string>) => {
      state.id = action.payload;
    },
    workflowReset: () => cloneDeep(initialWorkflowState),
    workflowSaved: (state) => {
      state.isTouched = false;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(workflowLoaded, (state, action) => {
      const { nodes: _nodes, edges: _edges, ...workflowExtra } = action.payload;
      return { ...cloneDeep(workflowExtra), isTouched: true };
    });

    builder.addCase(nodesDeleted, (state, action) => {
      action.payload.forEach((node) => {
        state.exposedFields = state.exposedFields.filter(
          (f) => f.nodeId !== node.id
        );
      });
    });

    builder.addCase(nodeEditorReset, () => cloneDeep(initialWorkflowState));

    builder.addMatcher(isAnyNodeOrEdgeMutation, (state) => {
      state.isTouched = true;
    });
  },
});

export const {
  workflowExposedFieldAdded,
  workflowExposedFieldRemoved,
  workflowNameChanged,
  workflowDescriptionChanged,
  workflowTagsChanged,
  workflowAuthorChanged,
  workflowNotesChanged,
  workflowVersionChanged,
  workflowContactChanged,
  workflowIDChanged,
  workflowReset,
  workflowSaved,
} = workflowSlice.actions;

export default workflowSlice.reducer;
