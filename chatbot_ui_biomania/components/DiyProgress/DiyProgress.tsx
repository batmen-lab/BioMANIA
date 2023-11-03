import React, {useEffect} from 'react';
import './style.less';

interface DiyProgressProps {
    decimal: number | string;
    percent: number;        
    strokeWidth?: number;      
    strokeLinecap?: string;   
    strokeColor?: string;     
}


/**
 * @param props
 * @constructor
 */
const DiyProgress = (props: DiyProgressProps) => {
    const {
        percent = 0,
        decimal = 0,
        strokeWidth = 12,
        strokeLinecap = 'square',
        strokeColor = '#5B8FF9'
    } = props;

    useEffect(() => {
    }, []);


    return (
        <div className="bit-progress">
            <div className="bit-progress-outer">
                <div
                    className={["bit-progress-inner", percent < 20 ? 'bit-fixed-inner' : ''].join(' ')}
                    style={{borderRadius: strokeLinecap === 'round' ? 100 : 0}}
                >
                    <div
                        className={percent < 20 ? 'bit-fixed-text' : "bit-progress-text"}
                        style={{width: `${percent}%`}}
                    >{decimal}</div>
                    <div
                        className="bit-progress-bg"
                        style={{width: `${percent}%`, height: strokeWidth, backgroundColor: strokeColor}}
                    ></div>
                </div>
            </div>
        </div>
    )
}


export default DiyProgress;
