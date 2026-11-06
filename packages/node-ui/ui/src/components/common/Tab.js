import React from 'react';
import styles from "./Tab.module.css"
import {NavLink} from "react-router-dom"


const Tab = (props) => {

    return (
        <div className={styles.tab}>
            { props.tabs.map((item, key) => {
                return(
                    <NavLink
                        key={key} to={item.to}
                        className={({isActive}) =>
                            [styles.singleTab, props.className, isActive ? styles.active : null]
                                .filter(Boolean)
                                .join(" ")}
                    >
                        <div className={``}>
                            {item.name}
                        </div>
                    </NavLink>
                )
            })}
        </div>
    );
};

export default Tab;