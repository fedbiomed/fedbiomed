
import React, { Component }  from 'react';
import logo from "../../assets/img/fedbiomed-logo-small.png"
import NavItem from './NavItem';
import {ReactComponent as ConfIcon}  from '../../assets/img/configuration.svg'
import {ReactComponent as DataIcon}  from '../../assets/img/database.svg'
import {ReactComponent as FolderIcon}  from '../../assets/img/folder.svg'
import {ReactComponent as HomeIcon}  from '../../assets/img/home.svg'
import {ReactComponent as PlusIcon}  from '../../assets/img/plus.svg'
import {ReactComponent as FileIcon}  from '../../assets/img/file.svg'
import {useLocation} from "react-router-dom";


// Define menu items
const items = [
    { key: '1', label: 'Home', path: '/', icon: HomeIcon},
    { key: '2', label: 'List Data Files', path: '/repository/', icon: FolderIcon},
    { key: '3', label: 'TrainingPlans/Models', path: '/models/', icon: FileIcon},
    { key: '4', label: 'Manage Datasets', path: '/datasets/', icon: DataIcon },
    { key: '5', label: 'Add New Dataset', path: '/datasets/add-dataset/', icon: PlusIcon },
    { key: '6', label: 'Node Configuration', path: '/configuration/', icon: ConfIcon },
  ]


const SideNav= (props) => {

    const location = useLocation()

    const [selectedKey, setSelectedKey] = React.useState(items.find(_item => location.pathname.startsWith(_item.path)).key)

    const displayed = props.displayed;

    React.useEffect(() => {
        let item = items.find(_item => location.pathname === _item.path )
        if(item){
            setSelectedKey(item.key)
        }
    }, [setSelectedKey, location.pathname ])

    return (
        <div className="side-nav">
            <div className="side-nav-inner">
                <div className="brand">
                    <img alt="fedbiomed-logo" src={logo}/>
                    <h1>Fed-BioMed - Node GUI</h1>
                </div>
                <div className="nav-items">
                    {
                        items.map((item,key) => {
                            return (
                                <NavItem
                                    key={item.key}
                                    label={item.label}
                                    active={selectedKey === item.key ? true : false}
                                    path={item.path} icon={item.icon}
                                />
                            )
                        })
                    }
                </div>
            </div>
        </div>
    );
}

export default SideNav;