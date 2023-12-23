import React, { useState, useEffect, useRef } from 'react';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';

interface ParamInfo {
  type: string;
  description: string;
  optional: boolean;
  optional_value: boolean;
  default: any;
}

interface OptionalCardProps {
  params: Record<string, ParamInfo>;
}

const OptionalCard: React.FC<OptionalCardProps> = ({ params }) => {
  const [activeParam, setActiveParam] = useState<string | null>(null);
  const [inputValue, setInputValue] = useState('');
  const labelContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (Object.keys(params).length > 0) {
      setActiveParam(Object.keys(params)[0]);
    }
  }, [params]);

  const handleParamClick = (param: string) => {
    setActiveParam(param);
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  };

  const scroll = (direction: 'left' | 'right') => {
    if (labelContainerRef.current) {
      const scrollAmount = direction === 'left' ? -100 : 100;
      labelContainerRef.current.scrollLeft += scrollAmount;
    }
  };

  return (
    <Paper elevation={2} sx={{ width: '100%', padding: '8px', margin: '8px 0', backgroundColor: 'white' }}>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Button onClick={() => scroll('left')}>&lt;</Button>
        <div ref={labelContainerRef} style={{ overflowX: 'auto', whiteSpace: 'nowrap', display: 'flex' }}>
          {Object.keys(params).map((param) => (
            <div key={param} style={{ flex: 'none', marginRight: '10px' }}>
              <Button
                onClick={() => handleParamClick(param)}
                sx={{
                  textTransform: 'none',
                  fontSize: '0.75rem',
                  padding: '1px 1px',
                  backgroundColor: param === activeParam ? '#e0e0e0' : 'white',
                }}
                variant="outlined"
              >
                {param}
              </Button>
            </div>
          ))}
        </div>
        <Button onClick={() => scroll('right')}>&gt;</Button>
      </Box>
      {activeParam && (
        <div style={{ margin: '4px 0' }}>
          <Typography variant="body2" sx={{ fontSize: '0.8rem', marginBottom: '2px' }}>
            Description: {params[activeParam].description}
          </Typography>
          <Box display="flex" alignItems="center">
            <Typography variant="body2" sx={{ fontSize: '0.8rem', marginRight: '8px', marginBottom: '2px' }}>
              Type: {params[activeParam].type}. Input:
            </Typography>
            <TextField 
              value={inputValue} 
              onChange={handleInputChange} 
              sx={{ flexGrow: 1, marginBottom: '2px', input: { padding: '2px 10px' } }}
            />
          </Box>
        </div>
      )}
    </Paper>
  );
};

export default OptionalCard;
