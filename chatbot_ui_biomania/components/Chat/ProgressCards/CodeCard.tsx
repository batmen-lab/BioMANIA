import React, { useState } from 'react';
import { MemoizedReactMarkdown } from '@/components/Markdown/MemoizedReactMarkdown';
import { CodeBlock } from '@/components/Markdown/CodeBlock';
import remarkGfm from 'remark-gfm';
import Collapse from '@mui/material/Collapse';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';

interface CodeCardProps {
  codeString: string;
  language: string;
  fullContent: string;
}

function CodeCard({ codeString, language, fullContent }: CodeCardProps) {
  const [isCollapsed, setIsCollapsed] = useState(true);

  const codeStyle = {
    whiteSpace: 'pre-wrap',
  };

  return (
    <div style={{ padding: 0}}>
      <MemoizedReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre: ({ node, children, ...props }) => (
            <pre style={{ margin: 0, marginTop: 0.2, 
              marginBottom: 0.2 }} {...props}>
              {children}
            </pre>
          ),
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline ? (
              //<div style={{ padding: 0, marginTop: 0, marginBottom: 0 }}>
              <CodeBlock
                style={{ width: '100%', ...codeStyle, margin: 0, padding: 0,'& > p': {
                  margin: 0,
                  padding: 0,
                  marginTop: 0, 
                  marginBottom: 0
                }, }}
                language={language || (match && match[1]) || ''}
                value={codeString || String(children).replace(/\n$/, '')}
                {...props}
              />
              //</div>
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
          <Typography paragraph>
            <Button
              variant="text"
              sx={{
                color: 'black',
                textTransform: 'none',
                fontSize: '0.8rem',
                fontWeight: 'normal',
                '&:hover': {
                  backgroundColor: 'transparent',
                  textDecoration: 'underline',
                },
                margin: 0,
                padding: 0,
              }}
              onClick={() => setIsCollapsed(!isCollapsed)}
            >
              {isCollapsed ? "Show ↓" : "Hide ↑"}
            </Button>
          </Typography>

          <Collapse in={!isCollapsed} timeout="auto" unmountOnExit>
          <div style={{ padding: 0, marginTop: 0, marginBottom: 0 }}>
            <MemoizedReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline ? (
                    <CodeBlock
                      style={{ width: '100%', ...codeStyle, margin: 0, padding: 0,'& > p': {
                        margin: 0,
                        padding: 0,
                      } }}
                      language={language || (match && match[1]) || ''}
                      value={fullContent || String(children).replace(/\n$/, '')}
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
              {`\`\`\`${language}\n${fullContent}\n\`\`\``}
            </MemoizedReactMarkdown>
            </div>
          </Collapse>
        </>
      )}
    </div>
  );
}

export default CodeCard;
