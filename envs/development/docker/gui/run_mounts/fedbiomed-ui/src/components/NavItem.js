import React from 'react';
import {Link} from 'react-router-dom'
const NavItem  = (props) => {
    return (
        <div className="nav-item">
            <Link to={{pathname: props.path}}>{props.label}</Link>
        </div>
    );
}

export default NavItem;