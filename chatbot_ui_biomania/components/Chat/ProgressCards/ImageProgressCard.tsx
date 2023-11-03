import React from 'react';

interface ImageProgressCardProps {
  imageSrc: string;
}

const ImageProgressCard: React.FC<ImageProgressCardProps> = ({ imageSrc }) => {
  return (
    <div>
      <img src={imageSrc} alt="Progress Image" style={{ width: '100%', height: 'auto' }} />
    </div>
  );
};

export default ImageProgressCard;