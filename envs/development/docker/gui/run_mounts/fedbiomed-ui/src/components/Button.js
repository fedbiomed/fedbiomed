import React from 'react'

export const Button = (props) => {
    return (
        <div class={`button ${props.type ? props.type : ''}`} onClick={props.onClick}>
            {props.children}
        </div>
    )
}

export default Button