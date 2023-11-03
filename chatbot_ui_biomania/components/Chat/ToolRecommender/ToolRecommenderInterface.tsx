import React from 'react';
import { Box, Divider, List, ListItem, ListItemText, Typography, Button, IconButton } from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
import { Tool } from '@/types/chat';

interface ToolRecommenderProps {
  tools: Tool[];
  onClose: () => void;
}

const ToolRecommender: React.FC<ToolRecommenderProps> = ({ tools, onClose }) => {
  const [selectedTool, setSelectedTool] = React.useState<Tool | null>(null);

  const handleToolClick = (tool: Tool) => {
    setSelectedTool(tool);
  };

  return (
    <Box maxWidth="md" bgcolor="white" p={2} height="calc(100% - 100px)" width="100%">
      <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
        Tool Recommendations
      </Typography>
      <Divider />
      <Box display="flex" mt={2} height="100%">
        <Box flex={1} bgcolor="#f5f5f5" p={2} overflow="auto" position="sticky" top={0}>
          <Typography variant="h6" component="h2" gutterBottom sx={{
            color: "black"
          }}>
            Tools
          </Typography>
          <List>
            {tools.map((tool) => (
              <ListItem
                key={tool.name}
                button
                onClick={() => handleToolClick(tool)}
                sx={{
                  cursor: 'pointer',
                }}
              >
                <ListItemText primary={tool.name} />
              </ListItem>
            ))}
          </List>
        </Box>
        <Divider orientation="vertical" flexItem />
        <Box flex={2} p={2} overflow="auto" maxHeight={400}>
          <Typography variant="h6" component="h2" gutterBottom sx={{
            color: "black"
          }}>
            Description
          </Typography>
          {selectedTool ? (
            <Typography variant="body1">{selectedTool.description}</Typography>
          ) : (
            <Typography variant="body1">Select a tool to view its description</Typography>
          )}
        </Box>
      </Box>
      {/*<Box textAlign="right" mt={2}>*/}
      {/*  <Button variant="text" onClick={onClose}>*/}
      {/*    Close*/}
      {/*  </Button>*/}
      {/*</Box>*/}
    </Box>
  );
};

export default ToolRecommender;
