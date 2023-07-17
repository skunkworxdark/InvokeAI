import { FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { Select, SelectProps } from '@mantine/core';
import { useMantineSelectStyles } from 'mantine-theme/hooks/useMantineSelectStyles';
import { RefObject, memo } from 'react';

export type IAISelectDataType = {
  value: string;
  label: string;
  tooltip?: string;
};

type IAISelectProps = Omit<SelectProps, 'label'> & {
  tooltip?: string;
  inputRef?: RefObject<HTMLInputElement>;
  label?: string;
};

const IAIMantineSelect = (props: IAISelectProps) => {
  const { tooltip, inputRef, label, disabled, ...rest } = props;

  const styles = useMantineSelectStyles();

  return (
    <Tooltip label={tooltip} placement="top" hasArrow>
      <Select
        label={
          label ? (
            <FormControl isDisabled={disabled}>
              <FormLabel>{label}</FormLabel>
            </FormControl>
          ) : undefined
        }
        disabled={disabled}
        ref={inputRef}
        styles={styles}
        {...rest}
      />
    </Tooltip>
  );
};

export default memo(IAIMantineSelect);
