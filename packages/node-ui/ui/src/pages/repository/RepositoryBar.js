import React from 'react';

export const RepositoryBar = (props) => {
    return (
        <div className="repository-bar">
            {props.children}   
        </div>
    );
}

export default RepositoryBar;