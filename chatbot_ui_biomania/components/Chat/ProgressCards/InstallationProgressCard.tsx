import LinearProgress from '@mui/material/LinearProgress';
import Box from '@mui/material/Box';
import Typography from "@mui/material/Typography";
import { useState, useEffect } from 'react';

interface InstallationProgressCardProps {
  progress: string;
  message: string;
}

// Use module-scoped variable to store all messages
let accumulatedMessages: string[] = [];

export const InstallationProgressCard: React.FC<InstallationProgressCardProps> = ({ progress, message }) => {
  const [collapsed, setCollapsed] = useState(true);
  const [details, setDetails] = useState<string[]>(accumulatedMessages);
  const progressValue = parseInt(progress, 10);

  useEffect(() => {
    accumulatedMessages.push(message);
    setDetails(accumulatedMessages);  // Update the details state to reflect accumulated messages
  }, [message]);

  return (
    <Box sx={{ mb: 2, width: '100%' }}>
      <LinearProgress variant="determinate" value={progressValue} sx={{ width: 500 }} />
      <Box sx={{ mt: 1 }}>
        <Typography variant="body1">{message}</Typography>
      </Box>
    </Box>
  );
}
