import DiyProgress from "../../../components/DiyProgress/DiyProgress";

const changeProgressBgColor = (flag: number) => {
  if (flag === 1) {
      return '#5B8FF9';
  }
  if (flag === 2) {
      return '#5AD8A6';
  }
  return '#313C42';
}

interface ProgressWithInfoProps {
  percent: number;
  decimal: number;
  strokeColor: string;
  flag: number;
}

function ProgressWithInfo({ percent, decimal, strokeColor, flag }: ProgressWithInfoProps) {
  const getInfoText = () => {
    if (flag === 1) {
      return "正在加载第一部分...";
    }
    if (flag === 2) {
      return "正在处理数据...";
    }
    return "完成！";
  };

  return (
    <div className="progress-info-container">
      <span className="info-text">{getInfoText()}</span>
      <DiyProgress
        percent={percent}
        decimal={decimal}
        strokeColor={strokeColor}
      />
    </div>
  );
}

<ProgressWithInfo 
  percent={35} 
  decimal={0.35} 
  strokeColor={changeProgressBgColor(flag)} 
  flag={flag} 
/>
