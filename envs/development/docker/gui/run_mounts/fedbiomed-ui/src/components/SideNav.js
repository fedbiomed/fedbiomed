
import React from 'react';
import logo from "../assets/img/fedbiomed-logo-small.png"
import NavItem from './NavItem';
import {ReactComponent as ConfIcon}  from '../assets/img/configuration.svg' 
import {ReactComponent as DataIcon}  from '../assets/img/database.svg' 
import {ReactComponent as FolderIcon}  from '../assets/img/folder.svg' 
import {ReactComponent as HomeIcon}  from '../assets/img/home.svg' 


// Define menu items
const items = [
    { key: '1', label: 'Home', path: '/', icon: HomeIcon },
    { key: '2', label: 'Repository', path: '/repository', icon: FolderIcon },
    { key: '3', label: 'Datasets', path: '/datasets', icon: DataIcon },
    { key: '4', label: 'Node Configuration', path: '/configuration', icon: ConfIcon },
  ]


const SideNav  = (props) => {

    //const location = React.useLocation()
    //const history = React.useHistory()

    //const [selectedKey, setSelectedKey] = React.useState(items.find(_item => location.pathname.startsWith(_item.path)).key)



    return (
        <div className="side-nav">
            <div className="side-nav-inner">
                <div className="brand">
                    <img alt="fedbiomed-logo" src={logo}/>
                    <h1>Fed-BioMed - Node GUI</h1>
                </div>
                <div className="nav-items">
                    {
                        items.map((item) => {
                            return (
                                <NavItem key={item.key} label={item.label} path={item.path} icon={item.icon}/>
                            )
                        })
                    }
                </div>
            </div>
        </div>
    );
}

export default SideNav;