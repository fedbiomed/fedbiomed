
import React from 'react';
import logo from "../assets/img/fedbiomed-logo-small.png"
import NavItem

from './NavItem';
// Define menu items
const items = [
    { key: '1', label: 'Repository', path: '/repository' },
    { key: '2', label: 'Add Dataset', path: '/add-dataset' },
    { key: '3', label: 'Node Configuration', path: '/configuration' },
  ]


const SideNav  = (props) => {

    //const location = React.useLocation()
    //const history = React.useHistory()

    //const [selectedKey, setSelectedKey] = React.useState(items.find(_item => location.pathname.startsWith(_item.path)).key)



    return (
        <div className="side-nav">
            <div className="side-nav-inner">
                <div className="brand">
                    <img src={logo}/>
                    <h1>Fed-BioMed - Node GUI</h1>
                </div>
                <div className="nav-items">
                    {
                        items.map((item) => {
                            return (
                                <NavItem key={item.key} label={item.label} path={item.path}/>
                            )
                        })
                    }
                </div>
            </div>
        </div>
    );
}

export default SideNav;