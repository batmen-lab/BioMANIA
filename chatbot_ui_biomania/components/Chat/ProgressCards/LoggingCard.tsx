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
    <Paper elevation={2} sx={{ width: '200', height: 'auto', position: 'relative', padding: '8px', margin: '0', backgroundColor: 'white', overflow: 'hidden' }}>
      <Box sx={{ display: 'flex', flexDirection: 'column' }}>
        <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace', fontSize: '0.9rem', lineHeight: '1.2', marginBottom: '0px' }}>
          <span style={{ color: titleColor }}>{title}</span>
        </Typography>
        <Divider sx={{ my: 0.5 }} />
        <Collapse in={!isCollapsed} timeout="auto" unmountOnExit>
          {imageData && (
            <ImageProgressCard
              key={generate_random_id()}
              imageSrc={'data:image/png;base64,' + imageData}
            />
          )}
          {tableData && tableData.trim() !== '' && (
            <TableCard data={tableData} />
          )}
          <Typography variant="body1" sx={{
            fontFamily: 'monospace',
            color: logColor,
            fontSize: '0.9rem',
            lineHeight: '1.2',
            mt: 0,
            mb: 0,
            p: 0,
            '& > p': {
              margin: 0,
            },
          }}>
            <ReactMarkdown>{formattedLogString}</ReactMarkdown>
          </Typography>
        </Collapse>
        <Button
          variant="text"
          onClick={toggleCollapse}
          sx={{
            position: 'absolute',
            top: 0,
            right: 0,
            fontSize: '0.8rem',
            padding: '4px',
          }}
        >
          {isCollapsed ? "Show" : "Hide"}  {isCollapsed ? "↓" : "↑"}
        </Button>
      </Box>
    </Paper>
  );
};

export default LoggingCard;
