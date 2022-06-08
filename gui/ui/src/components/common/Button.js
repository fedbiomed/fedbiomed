import React from 'react'

export const Button = (props) => {
    return (
        <div style={props.style} className={`button ${props.type ? props.type : ''}`} onClick={props.onClick}>
            {props.children}
        </div>
    )
}

export const ButtonsWrapper = (props) => {
    return(
        <div className={`buttons-wrapper ${props.className}`}>
            {props.children}
        </div>
    )
}


export default Button