import React from 'react';
import {Link} from 'react-router-dom'

const NavItem  = (props) => {

    const handleClick = (events) => {
        // logic executed when user clicked on panel
        if (props.action !== null) {
            // only executed if props.action is a function
            props.action()
            
            }
    }

    return (
        <div className={`nav-item ${props.active ? 'active' : ''}`} onClick={handleClick}>

            <Link to={{pathname: props.path}}>
                <div className="nav-item-inner" >
                    <div className="nav-icon">
                        <props.icon/>
                    </div>
                    <div className="nav-label">
                        {props.label}
                    </div>
                </div>
            </Link>
        </div>
    );
}

export default NavItem;