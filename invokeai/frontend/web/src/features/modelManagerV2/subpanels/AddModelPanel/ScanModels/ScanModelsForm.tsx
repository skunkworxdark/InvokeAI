import { Button, Flex, FormControl, FormErrorMessage, FormLabel, Input } from '@invoke-ai/ui-library';
import type { ChangeEventHandler } from 'react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useLazyScanModelsQuery } from 'services/api/endpoints/models';

import { ScanModelsResults } from './ScanModelsResults';

export const ScanModelsForm = () => {
  const [scanPath, setScanPath] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const { t } = useTranslation();

  const [_scanModels, { isLoading, data }] = useLazyScanModelsQuery();

  const handleSubmitScan = useCallback(async () => {
    _scanModels({ scan_path: scanPath }).catch((error) => {
      if (error) {
        setErrorMessage(error.data.detail);
      }
    });
  }, [_scanModels, scanPath]);

  const handleSetScanPath: ChangeEventHandler<HTMLInputElement> = useCallback((e) => {
    setScanPath(e.target.value);
    setErrorMessage('');
  }, []);

  return (
    <Flex flexDir="column" height="100%">
      <FormControl isInvalid={!!errorMessage.length} w="full">
        <Flex flexDir="column" w="full">
          <Flex gap={2} alignItems="flex-end" justifyContent="space-between">
            <Flex direction="column" w="full">
              <FormLabel>{t('common.folder')}</FormLabel>
              <Input value={scanPath} onChange={handleSetScanPath} />
            </Flex>

            <Button onClick={handleSubmitScan} isLoading={isLoading} isDisabled={scanPath.length === 0}>
              {t('modelManager.scanFolder')}
            </Button>
          </Flex>
          {!!errorMessage.length && <FormErrorMessage>{errorMessage}</FormErrorMessage>}
        </Flex>
      </FormControl>
      {data && <ScanModelsResults results={data} />}
    </Flex>
  );
};
