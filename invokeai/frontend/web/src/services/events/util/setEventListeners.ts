import { $baseUrl } from 'app/store/nanostores/baseUrl';
import { $bulkDownloadId } from 'app/store/nanostores/bulkDownloadId';
import { $queueId } from 'app/store/nanostores/queueId';
import type { AppDispatch } from 'app/store/store';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import {
  socketBulkDownloadCompleted,
  socketBulkDownloadFailed,
  socketBulkDownloadStarted,
  socketConnected,
  socketDisconnected,
  socketGeneratorProgress,
  socketGraphExecutionStateComplete,
  socketInvocationComplete,
  socketInvocationError,
  socketInvocationRetrievalError,
  socketInvocationStarted,
  socketModelInstallCompleted,
  socketModelInstallDownloading,
  socketModelInstallError,
  socketModelLoadCompleted,
  socketModelLoadStarted,
  socketQueueItemStatusChanged,
  socketSessionRetrievalError,
} from 'services/events/actions';
import type { ClientToServerEvents, ServerToClientEvents } from 'services/events/types';
import type { Socket } from 'socket.io-client';

type SetEventListenersArg = {
  socket: Socket<ServerToClientEvents, ClientToServerEvents>;
  dispatch: AppDispatch;
};

export const setEventListeners = (arg: SetEventListenersArg) => {
  const { socket, dispatch } = arg;

  /**
   * Connect
   */
  socket.on('connect', () => {
    dispatch(socketConnected());
    const queue_id = $queueId.get();
    socket.emit('subscribe_queue', { queue_id });
    if (!$baseUrl.get()) {
      const bulk_download_id = $bulkDownloadId.get();
      socket.emit('subscribe_bulk_download', { bulk_download_id });
    }
  });

  socket.on('connect_error', (error) => {
    if (error && error.message) {
      const data: string | undefined = (error as unknown as { data: string | undefined }).data;
      if (data === 'ERR_UNAUTHENTICATED') {
        dispatch(
          addToast(
            makeToast({
              title: error.message,
              status: 'error',
              duration: 10000,
            })
          )
        );
      }
    }
  });

  /**
   * Disconnect
   */
  socket.on('disconnect', () => {
    dispatch(socketDisconnected());
  });

  /**
   * Invocation started
   */
  socket.on('invocation_started', (data) => {
    dispatch(socketInvocationStarted({ data }));
  });

  /**
   * Generator progress
   */
  socket.on('generator_progress', (data) => {
    dispatch(socketGeneratorProgress({ data }));
  });

  /**
   * Invocation error
   */
  socket.on('invocation_error', (data) => {
    dispatch(socketInvocationError({ data }));
  });

  /**
   * Invocation complete
   */
  socket.on('invocation_complete', (data) => {
    dispatch(
      socketInvocationComplete({
        data,
      })
    );
  });

  /**
   * Graph complete
   */
  socket.on('graph_execution_state_complete', (data) => {
    dispatch(
      socketGraphExecutionStateComplete({
        data,
      })
    );
  });

  /**
   * Model load started
   */
  socket.on('model_load_started', (data) => {
    dispatch(
      socketModelLoadStarted({
        data,
      })
    );
  });

  /**
   * Model load completed
   */
  socket.on('model_load_completed', (data) => {
    dispatch(
      socketModelLoadCompleted({
        data,
      })
    );
  });

  /**
   * Model Install Downloading
   */
  socket.on('model_install_downloading', (data) => {
    dispatch(
      socketModelInstallDownloading({
        data,
      })
    );
  });

  /**
   * Model Install Completed
   */
  socket.on('model_install_completed', (data) => {
    dispatch(
      socketModelInstallCompleted({
        data,
      })
    );
  });

  /**
   * Model Install Error
   */
  socket.on('model_install_error', (data) => {
    dispatch(
      socketModelInstallError({
        data,
      })
    );
  });

  /**
   * Session retrieval error
   */
  socket.on('session_retrieval_error', (data) => {
    dispatch(
      socketSessionRetrievalError({
        data,
      })
    );
  });

  /**
   * Invocation retrieval error
   */
  socket.on('invocation_retrieval_error', (data) => {
    dispatch(
      socketInvocationRetrievalError({
        data,
      })
    );
  });

  socket.on('queue_item_status_changed', (data) => {
    dispatch(socketQueueItemStatusChanged({ data }));
  });

  socket.on('bulk_download_started', (data) => {
    dispatch(socketBulkDownloadStarted({ data }));
  });

  socket.on('bulk_download_completed', (data) => {
    dispatch(socketBulkDownloadCompleted({ data }));
  });

  socket.on('bulk_download_failed', (data) => {
    dispatch(socketBulkDownloadFailed({ data }));
  });
};
