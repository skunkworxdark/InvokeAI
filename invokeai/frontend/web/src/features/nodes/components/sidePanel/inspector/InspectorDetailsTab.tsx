import {
  Box,
  Flex,
  FormControl,
  FormLabel,
  HStack,
  Text,
} from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import NotesTextarea from 'features/nodes/components/flow/nodes/Invocation/NotesTextarea';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import {
  InvocationNode,
  InvocationTemplate,
  isInvocationNode,
} from 'features/nodes/types/invocation';
import { getNeedsUpdate } from 'features/nodes/util/node/nodeUpdate';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import EditableNodeTitle from './details/EditableNodeTitle';

const selector = createMemoizedSelector(stateSelector, ({ nodes }) => {
  const lastSelectedNodeId =
    nodes.selectedNodes[nodes.selectedNodes.length - 1];

  const lastSelectedNode = nodes.nodes.find(
    (node) => node.id === lastSelectedNodeId
  );

  const lastSelectedNodeTemplate = lastSelectedNode
    ? nodes.nodeTemplates[lastSelectedNode.data.type]
    : undefined;

  return {
    node: lastSelectedNode,
    template: lastSelectedNodeTemplate,
  };
});

const InspectorDetailsTab = () => {
  const { node, template } = useAppSelector(selector);
  const { t } = useTranslation();

  if (!template || !isInvocationNode(node)) {
    return (
      <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />
    );
  }

  return <Content node={node} template={template} />;
};

export default memo(InspectorDetailsTab);

type ContentProps = {
  node: InvocationNode;
  template: InvocationTemplate;
};

const Content = memo(({ node, template }: ContentProps) => {
  const { t } = useTranslation();
  const needsUpdate = useMemo(
    () => getNeedsUpdate(node, template),
    [node, template]
  );
  return (
    <Box
      sx={{
        position: 'relative',
        w: 'full',
        h: 'full',
      }}
    >
      <ScrollableContent>
        <Flex
          sx={{
            flexDir: 'column',
            position: 'relative',
            p: 1,
            gap: 2,
            w: 'full',
          }}
        >
          <EditableNodeTitle nodeId={node.data.id} />
          <HStack>
            <FormControl>
              <FormLabel>{t('nodes.nodeType')}</FormLabel>
              <Text fontSize="sm" fontWeight={600}>
                {template.title}
              </Text>
            </FormControl>
            <Flex
              flexDir="row"
              alignItems="center"
              justifyContent="space-between"
              w="full"
            >
              <FormControl isInvalid={needsUpdate}>
                <FormLabel>{t('nodes.nodeVersion')}</FormLabel>
                <Text fontSize="sm" fontWeight={600}>
                  {node.data.version}
                </Text>
              </FormControl>
            </Flex>
          </HStack>
          <NotesTextarea nodeId={node.data.id} />
        </Flex>
      </ScrollableContent>
    </Box>
  );
});

Content.displayName = 'Content';
