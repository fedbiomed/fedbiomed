import React from 'react';
import styles from "./Step.module.css"


const Step = (props) => {

    const child = props.children

    return (
        <div className={`${styles.step} ${props.disable ? styles.notAllowed : null}`}>
            <div className={styles.number}>
                <span>{props.step}</span>
            </div>
            <div className={styles.content}>
                <div className={styles.contentInner}>
                    <div className={styles.label}>
                        <label>{props.desc}</label>
                    </div>
                    <div className={styles.outlet}>
                        {child}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Step;