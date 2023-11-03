import React, { useState } from 'react';
import Collapse from '@mui/material/Collapse';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';

interface TextCollapseCardProps {
  fullContent: string;
}

function TextCollapseCard({ fullContent }: TextCollapseCardProps) {
  const [isCollapsed, setIsCollapsed] = useState(true);

  const collapseStyle = {
    maxHeight: isCollapsed ? '0' : '1000px',
    overflow: 'hidden',
    transition: 'max-height 0.3s ease-in-out',
  };

  return (
    <div style={{ width: '100%' }}>
      <Paper elevation={3} style={{ backgroundColor: 'white', borderRadius: '5px', padding: '10px', margin: '10px 0' }}>
        <Typography paragraph>
          <Button
            variant="text"
            sx={{
              color: 'black',
              textTransform: 'none',
              fontSize: '0.8rem',
              fontWeight: 'normal',
              ml: 0,
              '&:hover': {
                backgroundColor: 'transparent',
                textDecoration: 'underline',
              }
            }}
            onClick={() => setIsCollapsed(!isCollapsed)}
          >
            {isCollapsed ? "Show Content" : "Hide Content"}  {isCollapsed ? "↓" : "↑"}
          </Button>
        </Typography>
      </Paper>

      <Collapse in={!isCollapsed} timeout="auto" unmountOnExit style={collapseStyle}>
        <Paper elevation={3} style={{ backgroundColor: 'white', borderRadius: '5px', padding: '10px', margin: '10px 0' }}>
          <Typography>
            {fullContent}
          </Typography>
        </Paper>
      </Collapse>
    </div>
  );
}

export default TextCollapseCard;
