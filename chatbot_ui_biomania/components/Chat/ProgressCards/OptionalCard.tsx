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
  onParamsChange: (newParams: Record<string, any>) => void;
}

const OptionalCard: React.FC<OptionalCardProps> = ({ params, onParamsChange }) => {
  const [activeParam, setActiveParam] = useState<string | null>(null);
  const [editedValues, setEditedValues] = useState<Record<string, any>>({});
  const labelContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (Object.keys(params).length > 0) {
      const firstParam = Object.keys(params)[0];
      // Set activeParam to the first param only if it's not already set
      // or if it's not in the editedValues
      setActiveParam(prev => prev && editedValues[prev] !== undefined ? prev : firstParam);
    }
  }, [params, editedValues]);

  // 从 localStorage 初始化 editedValues
  useEffect(() => {
    const savedValues = localStorage.getItem('editedValues');
    if (savedValues) {
      setEditedValues(JSON.parse(savedValues));
    }
  }, []);
  useEffect(() => {
    localStorage.setItem('editedValues', JSON.stringify(editedValues));
  }, [editedValues]);

  const handleParamClick = (param: string) => {
    if (activeParam && editedValues[activeParam] !== undefined) {
      const newParams = { ...params };
      newParams[activeParam].default = editedValues[activeParam];
      onParamsChange(newParams);
    }
    setActiveParam(param);
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setEditedValues({ ...editedValues, [activeParam!]: event.target.value });
  };
  
  const handleInputBlur = () => {
    if (activeParam && editedValues[activeParam] !== undefined) {
      onParamsChange({ ...params, [activeParam]: { ...params[activeParam], default: editedValues[activeParam] } });
    }
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
          <Typography variant="body2" sx={{ fontSize: '0.8rem', marginBottom: '2px', color: 'grey' }}>
            Default: {params[activeParam].default}
          </Typography>
          <Box display="flex" alignItems="center">
            <Typography variant="body2" sx={{ fontSize: '0.8rem', marginRight: '8px', marginBottom: '2px' }}>
              Type: {params[activeParam].type}. Input:
            </Typography>
            <TextField 
              value={editedValues[activeParam] !== undefined ? editedValues[activeParam] : params[activeParam].default} 
              onChange={handleInputChange}
              onBlur={handleInputBlur}
              sx={{
                flexGrow: 1, 
                marginBottom: '2px', 
                input: { 
                  padding: '2px 10px', 
                  color: 'grey', // Setting the text color to grey
                  fontSize: '0.8rem' // Setting a smaller font size
                }
              }}
            />
          </Box>
        </div>
      )}
    </Paper>
  );
};

export default OptionalCard;