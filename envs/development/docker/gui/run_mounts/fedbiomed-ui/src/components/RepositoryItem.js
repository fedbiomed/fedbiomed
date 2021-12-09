import React from 'react';
import { ReactComponent as FolderIcon } from '../assets/img/folder.svg';
import { ReactComponent as FileIcon } from '../assets/img/file.svg';
import { ReactComponent as PlusIcon } from '../assets/img/plus.svg';
import { ReactComponent as LaunchIcon} from '../assets/img/launch.svg'
import {Link} from "react-router-dom"

export const RepositoryItem = (props) => {

    const [hover, setHover] = React.useState(false)


    return (
        <div 
            className="repository-item" 
            onClick={() => props.onItemClick(props.type, props.path)} 
            >
            { props.type === 'dir' ? (
                <div className="icon">
                    <FolderIcon/>
                </div>

                ) : (
                    <div className="icon">
                        <FileIcon/>
                    </div>
                )
            }
            <div className="name">
                {props.name}
            </div>
            {props.registered ? (
                <React.Fragment>
                    <div className="icon right action-display" title="This item is registered as dataset">
                        <div className="dot"/>
                    </div>
                    <div className="icon right action-display">
                        <Link to={{pathname: `/datasets/preview/${props.registered.dataset_id}`}}>
                            <LaunchIcon/>
                        </Link>
                    </div>
                </React.Fragment>
            ) : (
                <div title="Add as dataset" className="icon right action-add" onClick={(event) => props.onAddActionClick(event, props.type, props.path)}>
                    <PlusIcon/>
                </div>
            )}
        </div>
    );
}

export default RepositoryItem;