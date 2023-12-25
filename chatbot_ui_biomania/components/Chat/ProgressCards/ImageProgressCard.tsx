import React, { useState } from 'react';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import DownloadIcon from '@mui/icons-material/CloudDownload';
import { CSSProperties } from 'react';

interface ImageProgressCardProps {
  imageSrc: string;
}

const ImageProgressCard: React.FC<ImageProgressCardProps> = ({ imageSrc }) => {
  const [scale, setScale] = useState(1);

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (e.deltaY < 0) {
      setScale(scale * 1.1);
    } else if (e.deltaY > 0) {
      setScale(scale / 1.1);
    }
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = imageSrc;
    link.download = 'image.webp';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const imageStyle: CSSProperties = {
    transform: `scale(${scale})`,
    transition: 'transform 0.2s ease',
    maxWidth: '100%',
    height: 'auto',
  };

  const iconStyle: CSSProperties = {
    position: 'absolute',
    right: '8px',
    bottom: '8px',
    color: '#fff',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: '50%',
    padding: '4px',
  };

  const downloadIconStyle: CSSProperties = {
    ...iconStyle,
    right: '40px',
  };

  return (
    <div onWheel={handleWheel} style={{ position: 'relative', overflow: 'hidden', cursor: 'zoom-in' }}>
      <img src={imageSrc} alt="Progress Image" style={imageStyle} />
      <ZoomInIcon style={iconStyle} />
      <DownloadIcon style={downloadIconStyle} onClick={handleDownload} />
    </div>
  );
};

export default ImageProgressCard;