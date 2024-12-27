import React from 'react'
import styles from "./Button.module.css"

export const Button = (props) => {
    return (
        <span style={{cursor: props.disable ? "not-allowed" : "pointer", ...props.wrapperStyle}}>
            <div style={props.style} className={`${styles.button} 
                                                 ${styles[props.type]} 
                                                 ${props.disable ? styles.buttonDisable : ""} `}
                 onClick={props.onClick}>
                    {props.children}
            </div>
        </span>
    )
}

export const ButtonsWrapper = (props) => {
    return(
        <div className={`${styles.buttonsWrapper} 
                         ${styles[props.alignment]} 
                         ${props.className}`}>
            {props.children}
        </div>
    )
}


export default Button