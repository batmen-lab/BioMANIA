import React from 'react';
import { MemoizedReactMarkdown } from '@/components/Markdown/MemoizedReactMarkdown';
interface TableCardProps {
  data: string;
}

const TableCard: React.FC<TableCardProps> = ({ data }) => {
  const rows = data.split('\n').map(row => row.split(',').map(cell => cell.trim()));
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <tbody>
        {rows.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {row.map((cell, cellIndex) => (
              <td key={cellIndex} style={{ border: '1px solid black', padding: '5px' }}>
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default TableCard;
