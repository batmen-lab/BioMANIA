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
import rehypeRaw from 'rehype-raw';

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
  const confirmationMatch = /Enter Parameters|Can you confirm|User Confirmation|Could you confirm whether this API should be called\? Please enter y\/n\./;
  const planMatch = /Multi step Task Planning|SubTask Execution|Continue to the next subtask|Step \d+: .*/;
  const erroranalysisMatch = /Error Analysis/;

  if (successMatch.test(title)) {
    titleColor = 'green';
  } else if (failMatch.test(title)) {
    titleColor = 'red';
  } else if (confirmationMatch.test(title)) {
    titleColor = 'orange';
  } else if (planMatch.test(title)) {
    titleColor = 'blue';
  } else if (erroranalysisMatch.test(title)) {
    titleColor = 'blue';
  }

  const theme = useTheme();

  const formattedLogString = logString.replace(/\n/g, '  \n');

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <Paper elevation={2} sx={{ width: '100%', maxWidth: '50vw', height: 'auto', position: 'relative', padding: '8px', margin: '0', backgroundColor: 'white', overflow: 'hidden' }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', paddingLeft: '0px' }}>
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
            <Box sx={{ overflowX: 'auto', width: '100%', paddingLeft: '0px' }}>
              <TableCard data={tableData} />
            </Box>
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
            <ReactMarkdown components={{ p: 'span' }} rehypePlugins={[rehypeRaw]}>
              {formattedLogString}
            </ReactMarkdown>
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
