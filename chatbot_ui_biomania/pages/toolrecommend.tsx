// pages/index.tsx

import React from 'react';
import ToolRecommenderDialog from "@/components/Chat/ToolRecommender/ToolRecommenderDialog";
import { Tool } from '@/types/chat';
import {Button} from "@mui/material";
const tools = [
  { name: 'Tool 1', description: 'Description 1' },
  { name: 'Tool 2', description: 'Description 2' },
] as Tool[];

const HomePage: React.FC = () => {
  const [open, setOpen] = React.useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  return (
    <div>
      <h1>Welcome to the Tool Recommender</h1>
      <Button variant="contained" onClick={handleOpen}>
        Open Tool Recommender
      </Button>
      <ToolRecommenderDialog tools={tools} open={open} onClose={handleClose} />
    </div>
  );
};

export default HomePage;
