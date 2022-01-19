
import React from 'react';

const Header  = (props) => {
    return (
        <div className="header">
            <div className="header-inner">
                <h1>{props.text}</h1>
            </div>
        </div>
    );
}

export default Header;