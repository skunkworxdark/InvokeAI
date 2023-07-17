import {
  Box,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  HStack,
  Tooltip,
} from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useIsValidConnection } from 'features/nodes/hooks/useIsValidConnection';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationTemplate,
} from 'features/nodes/types/types';
import { map } from 'lodash-es';
import { ReactNode, memo, useCallback } from 'react';
import FieldHandle from '../FieldHandle';
import InputFieldComponent from '../InputFieldComponent';

interface IAINodeInputProps {
  nodeId: string;

  input: InputFieldValue;
  template?: InputFieldTemplate | undefined;
  connected: boolean;
}

function IAINodeInput(props: IAINodeInputProps) {
  const { nodeId, input, template, connected } = props;
  const isValidConnection = useIsValidConnection();

  return (
    <Box
      className="nopan"
      position="relative"
      borderColor={
        !template
          ? 'error.400'
          : !connected &&
            ['always', 'connectionOnly'].includes(
              String(template?.inputRequirement)
            ) &&
            input.value === undefined
          ? 'warning.400'
          : undefined
      }
    >
      <FormControl isDisabled={!template ? true : connected} pl={2}>
        {!template ? (
          <HStack justifyContent="space-between" alignItems="center">
            <FormLabel>Unknown input: {input.name}</FormLabel>
          </HStack>
        ) : (
          <>
            <HStack justifyContent="space-between" alignItems="center">
              <HStack>
                <Tooltip
                  label={template?.description}
                  placement="top"
                  hasArrow
                  shouldWrapChildren
                  openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
                >
                  <FormLabel>{template?.title}</FormLabel>
                </Tooltip>
              </HStack>
              <InputFieldComponent
                nodeId={nodeId}
                field={input}
                template={template}
              />
            </HStack>

            {!['never', 'directOnly'].includes(
              template?.inputRequirement ?? ''
            ) && (
              <FieldHandle
                nodeId={nodeId}
                field={template}
                isValidConnection={isValidConnection}
                handleType="target"
              />
            )}
          </>
        )}
      </FormControl>
    </Box>
  );
}

interface IAINodeInputsProps {
  nodeId: string;
  template: InvocationTemplate;
  inputs: Record<string, InputFieldValue>;
}

const IAINodeInputs = (props: IAINodeInputsProps) => {
  const { nodeId, template, inputs } = props;

  const edges = useAppSelector((state: RootState) => state.nodes.edges);

  const renderIAINodeInputs = useCallback(() => {
    const IAINodeInputsToRender: ReactNode[] = [];
    const inputSockets = map(inputs);

    inputSockets.forEach((inputSocket, index) => {
      const inputTemplate = template.inputs[inputSocket.name];

      const isConnected = Boolean(
        edges.filter((connectedInput) => {
          return (
            connectedInput.target === nodeId &&
            connectedInput.targetHandle === inputSocket.name
          );
        }).length
      );

      if (index < inputSockets.length) {
        IAINodeInputsToRender.push(
          <Divider key={`${inputSocket.id}.divider`} />
        );
      }

      IAINodeInputsToRender.push(
        <IAINodeInput
          key={inputSocket.id}
          nodeId={nodeId}
          input={inputSocket}
          template={inputTemplate}
          connected={isConnected}
        />
      );
    });

    return (
      <Flex className="nopan" flexDir="column" gap={2} p={2}>
        {IAINodeInputsToRender}
      </Flex>
    );
  }, [edges, inputs, nodeId, template.inputs]);

  return renderIAINodeInputs();
};

export default memo(IAINodeInputs);
