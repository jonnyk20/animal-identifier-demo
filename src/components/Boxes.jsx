import React from 'react';

const Boxes = ({ boxes }) => (
    <>
    {boxes.map((box, i) => {
        const { x, y, w, h, score, label } = box;
        console.log({ x, y, score, label });
        const style = {
            top: y,
            left: x,
            width: w,
            height: h
        }

        return (
            <div className="bounging-box" key={`box-${i}`} style={style}>
                {`${label} - %${score}`}
            </div>
        );
    })}
    </>
);

export default Boxes;