import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { api } from 'services/api';
import { modelsApi } from 'services/api/endpoints/models';
import {
  socketModelInstallCompleted,
  socketModelInstallDownloading,
  socketModelInstallError,
} from 'services/events/actions';

export const addModelInstallEventListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: socketModelInstallDownloading,
    effect: async (action, { dispatch }) => {
      const { bytes, total_bytes, id } = action.payload.data;

      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.bytes = bytes;
            modelImport.total_bytes = total_bytes;
            modelImport.status = 'downloading';
          }
          return draft;
        })
      );
    },
  });

  startAppListening({
    actionCreator: socketModelInstallCompleted,
    effect: (action, { dispatch }) => {
      const { id } = action.payload.data;

      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.status = 'completed';
          }
          return draft;
        })
      );
      dispatch(api.util.invalidateTags(['Model']));
    },
  });

  startAppListening({
    actionCreator: socketModelInstallError,
    effect: (action, { dispatch }) => {
      const { id, error, error_type } = action.payload.data;

      dispatch(
        modelsApi.util.updateQueryData('listModelInstalls', undefined, (draft) => {
          const modelImport = draft.find((m) => m.id === id);
          if (modelImport) {
            modelImport.status = 'error';
            modelImport.error_reason = error_type;
            modelImport.error = error;
          }
          return draft;
        })
      );
    },
  });
};
