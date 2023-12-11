import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectListImagesBaseQueryArgs } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { IMAGE_LIMIT } from 'features/gallery/store/types';
import { clamp } from 'lodash-es';
import { useCallback } from 'react';
import { boardsApi } from 'services/api/endpoints/boards';
import {
  imagesApi,
  useLazyListImagesQuery,
} from 'services/api/endpoints/images';
import { ListImagesArgs } from 'services/api/types';
import { imagesAdapter } from 'services/api/util';

export const nextPrevImageButtonsSelector = createMemoizedSelector(
  [stateSelector, selectListImagesBaseQueryArgs],
  (state, baseQueryArgs) => {
    const { data, status } =
      imagesApi.endpoints.listImages.select(baseQueryArgs)(state);

    const { data: totalsData } =
      state.gallery.galleryView === 'images'
        ? boardsApi.endpoints.getBoardImagesTotal.select(
            baseQueryArgs.board_id ?? 'none'
          )(state)
        : boardsApi.endpoints.getBoardAssetsTotal.select(
            baseQueryArgs.board_id ?? 'none'
          )(state);

    const lastSelectedImage =
      state.gallery.selection[state.gallery.selection.length - 1];

    const isFetching = status === 'pending';

    if (!data || !lastSelectedImage || totalsData?.total === 0) {
      return {
        isFetching,
        queryArgs: baseQueryArgs,
        isOnFirstImage: true,
        isOnLastImage: true,
      };
    }

    const queryArgs: ListImagesArgs = {
      ...baseQueryArgs,
      offset: data.ids.length,
      limit: IMAGE_LIMIT,
    };

    const selectors = imagesAdapter.getSelectors();

    const images = selectors.selectAll(data);

    const currentImageIndex = images.findIndex(
      (i) => i.image_name === lastSelectedImage.image_name
    );
    const nextImageIndex = clamp(currentImageIndex + 1, 0, images.length - 1);
    const prevImageIndex = clamp(currentImageIndex - 1, 0, images.length - 1);

    const nextImageId = images[nextImageIndex]?.image_name;
    const prevImageId = images[prevImageIndex]?.image_name;

    const nextImage = nextImageId
      ? selectors.selectById(data, nextImageId)
      : undefined;
    const prevImage = prevImageId
      ? selectors.selectById(data, prevImageId)
      : undefined;

    const imagesLength = images.length;

    return {
      loadedImagesCount: images.length,
      currentImageIndex,
      areMoreImagesAvailable: (totalsData?.total ?? 0) > imagesLength,
      isFetching: status === 'pending',
      nextImage,
      prevImage,
      queryArgs,
    };
  }
);

export const useNextPrevImage = () => {
  const dispatch = useAppDispatch();

  const {
    nextImage,
    prevImage,
    areMoreImagesAvailable,
    isFetching,
    queryArgs,
    loadedImagesCount,
    currentImageIndex,
  } = useAppSelector(nextPrevImageButtonsSelector);

  const handlePrevImage = useCallback(() => {
    prevImage && dispatch(imageSelected(prevImage));
  }, [dispatch, prevImage]);

  const handleNextImage = useCallback(() => {
    nextImage && dispatch(imageSelected(nextImage));
  }, [dispatch, nextImage]);

  const [listImages] = useLazyListImagesQuery();

  const handleLoadMoreImages = useCallback(() => {
    listImages(queryArgs);
  }, [listImages, queryArgs]);

  return {
    handlePrevImage,
    handleNextImage,
    isOnFirstImage: currentImageIndex === 0,
    isOnLastImage:
      currentImageIndex !== undefined &&
      currentImageIndex === loadedImagesCount - 1,
    nextImage,
    prevImage,
    areMoreImagesAvailable,
    handleLoadMoreImages,
    isFetching,
  };
};
