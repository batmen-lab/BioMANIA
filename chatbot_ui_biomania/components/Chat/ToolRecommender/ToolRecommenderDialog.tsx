// components/ToolRecommenderDialog.tsx

import React, { useState } from 'react';
import { Dialog, DialogTitle, DialogContent, Button, IconButton, Grow, styled } from '@mui/material';
import ToolRecommenderInterface from "@/components/Chat/ToolRecommender/ToolRecommenderInterface";
import { Close as CloseIcon } from '@mui/icons-material';
import { Tool } from '@/types/chat';
import DialogActions from "@mui/material/DialogActions";


interface ToolRecommenderDialogProps {
  tools: Tool[];
  open: boolean;
  onClose: () => void;
}

const StyledDialog = styled(Dialog)(({ theme }) => ({
  '& .MuiDialog-paper': {
    transition: theme.transitions.create('transform', {
      duration: theme.transitions.duration.enteringScreen,
      easing: 'cubic-bezier(0.25, 1, 0.5, 1)',
    }),
  },
}));


const ToolRecommenderDialog: React.FC<ToolRecommenderDialogProps> = ({tools, open, onClose }) => {
  return (
    <>
      <StyledDialog
        open={open}
        onClose={onClose}
        fullScreen
        TransitionComponent={Grow}
        transitionDuration={500}

      >
        <DialogTitle>
          Tool Recommender
          <IconButton
            aria-label="Close"
            onClick={onClose}
            sx={{
              position: 'absolute',
              top: 10,
              right: 10,
            }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <ToolRecommenderInterface tools={tools} onClose={onClose} />
        </DialogContent>
        {/*<DialogActions>*/}
        {/*  <Button onClick={onClose}>Close</Button>*/}
        {/*</DialogActions>*/}
      </StyledDialog>
    </>
  );
};

export default ToolRecommenderDialog;
