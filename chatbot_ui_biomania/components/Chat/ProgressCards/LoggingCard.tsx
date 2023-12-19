import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Divider from '@mui/material/Divider';
import { useTheme } from '@mui/material/styles';
import TableCard from './TableCard';
import ImageProgressCard from './ImageProgressCard';
import ReactMarkdown from 'react-markdown';

import React, { useState } from 'react';
import Button from '@mui/material/Button';
import Collapse from '@mui/material/Collapse';

interface LoggingCardProps {
  title: string;
  logString: string;
  tableData: string;
  logColor?: string;
  imageData?: string;
}

const generate_random_id = () => {
  return Math.random().toString(36).substr(2, 9);
};

const LoggingCard = ({ title, logString, tableData, logColor = 'black', imageData }: LoggingCardProps) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  let titleColor = 'black';

  const successMatch = /\[Success\]/;
  const failMatch = /\[Fail\]/;

  if (successMatch.test(title)) {
    titleColor = 'green';
  } else if (failMatch.test(title)) {
    titleColor = 'red';
  }

  const theme = useTheme();

  const formattedLogString = logString.replace(/\n/g, '  \n');

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <Paper elevation={2} sx={{ width: '200', position: 'relative', padding: '16px', margin: '16px 0', backgroundColor: 'white', overflow: 'hidden' }}>
      <Box sx={{ display: 'flex', flexDirection: 'column' }}>
        <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace', fontSize: theme.typography.body1.fontSize }}>
          <span style={{ color: titleColor }}>{title}</span>
        </Typography>
        <Divider sx={{ my: 1 }} />
        {imageData && (
          <ImageProgressCard
            key={generate_random_id()}
            imageSrc={'data:image/png;base64,' + imageData}
          />
        )}
        {tableData && tableData.trim() !== '' && <TableCard data={tableData} />}
        <Box sx={{ width: '200' }}>
          <Collapse in={!isCollapsed} timeout="auto" unmountOnExit sx={{ width: '200' }}>
            <Typography variant="body1" sx={{ fontFamily: 'monospace', color: logColor }}>
              <ReactMarkdown>{formattedLogString}</ReactMarkdown>
            </Typography>
          </Collapse>
        </Box>
        <Button
          variant="text"
          onClick={toggleCollapse}
          sx={{
            position: 'absolute',
            top: 0,
            right: 0,
            fontSize: '0.8rem',
          }}
        >
          {isCollapsed ? "Show" : "Hide"}  {isCollapsed ? "↓" : "↑"}
        </Button>
      </Box>
    </Paper>
  );
};

export default LoggingCard;
