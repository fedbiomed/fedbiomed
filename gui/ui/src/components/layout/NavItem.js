import React from 'react';
import {Link} from 'react-router-dom'
const NavItem  = (props) => {



    return (
        <div className={`nav-item ${props.active ? 'active' : ''}`}>
            <Link to={{pathname: props.path}}>
                <div className="nav-item-inner">
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