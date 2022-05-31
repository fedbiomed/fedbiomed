import React from 'react';
import styles from "./Accordion.module.css"

const Accordion = (props) => {

    const [show, setShow] = React.useState(false)

    const switcher = () => {
        if(show){
            setShow(false)
        }else{
            setShow(true)
        }
    }

    return (
        <div className={styles.wrapper}>
            <div className={styles.wrapperInner}>
                <div
                    onClick={switcher}
                    className={`${props.color} ${styles.dropdownLabel}`}>
                    {props.label}
                </div>
                <div className={styles.inner}>
                    <div className={`${styles.dropdownContent} ${ show ? styles.dropdownContentActive : ''}`}>
                        {props.children}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Accordion;