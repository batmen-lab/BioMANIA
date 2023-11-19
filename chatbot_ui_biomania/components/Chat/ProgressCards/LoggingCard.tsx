import React from 'react';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Divider from '@mui/material/Divider';
import { useTheme } from '@mui/material/styles';
import TableCard from './TableCard';
import ImageProgressCard from './ImageProgressCard';
import ReactMarkdown from 'react-markdown';

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

  return (
    <Paper elevation={2} sx={{ padding: '16px', margin: '16px 0', backgroundColor: 'white' }}>
      <Box sx={{ display: 'flex', flexDirection: 'column' }}>
        <Typography variant="caption" component="div" sx={{ fontFamily: 'monospace', fontSize: theme.typography.body1.fontSize }}>
          <span style={{ color: titleColor }}>{title}</span>
        </Typography>
        <Divider sx={{ my: 1 }} />
        <Typography variant="body1" sx={{ fontFamily: 'monospace', color: logColor }}>
          <ReactMarkdown>{formattedLogString}</ReactMarkdown>
        </Typography>
        {imageData && (
          <ImageProgressCard
            key={generate_random_id()}
            imageSrc={'data:image/png;base64,' + imageData}
          />
        )}
        {tableData && tableData.trim() !== '' && <TableCard data={tableData} />}
      </Box>
    </Paper>
  );
};

export default LoggingCard;
