import React, { useState } from 'react';
import { MemoizedReactMarkdown } from '@/components/Markdown/MemoizedReactMarkdown';
import { CodeBlock } from '@/components/Markdown/CodeBlock';
import remarkGfm from 'remark-gfm';
import Collapse from '@mui/material/Collapse';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';

interface CodeCardProps {
  codeString: string;
  language: string;
  fullContent: string; // new prop
}

function CodeCard({ codeString, language, fullContent }: CodeCardProps) {
  const [isCollapsed, setIsCollapsed] = useState(true);  // new state for collapse control

  const collapseStyle = {
    maxHeight: isCollapsed ? '0' : '1000px',
    overflow: 'hidden',
    transition: 'max-height 0.3s ease-in-out',
  };

  const codeStyle = {
    whiteSpace: 'pre-wrap',
  };

  return (
    <div style={{ width: '100%' }}>
      <MemoizedReactMarkdown 
        remarkPlugins={[remarkGfm]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline ? (
              <CodeBlock
                style={{ width: '100%', ...codeStyle }}
                language={language || (match && match[1]) || ''}
                value={codeString || String(children).replace(/\n$/, '')}
                {...props}
              />
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {`\`\`\`${language}\n${codeString}\n\`\`\``}
      </MemoizedReactMarkdown>
      
      {fullContent && (
        <>
          <Paper elevation={3} style={{ backgroundColor: 'white', padding: '1px', borderRadius: '1px', marginTop: '10px' , height: '30px'}}>
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
              {isCollapsed ? "Show Full Content" : "Hide Full Content"}  {isCollapsed ? "↓" : "↑"}
            </Button>
            </Typography>
          </Paper>

          <Collapse in={!isCollapsed} timeout="auto" unmountOnExit style={collapseStyle}>
            <MemoizedReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline ? (
                    <CodeBlock
                      style={{ width: '100%' , ...codeStyle }}
                      language={language || (match && match[1]) || ''}
                      value={fullContent || String(children).replace(/\n$/, '')}
                      {...props}
                    />
                  ) : (
                    <code className={className} {...props} >
                      {children}
                    </code>
                  );
                },
              }}
            >
              {`\`\`\`${language}\n${fullContent}\n\`\`\``}
            </MemoizedReactMarkdown>
          </Collapse>
        </>
      )}
    </div>
  );
}

export default CodeCard;
