import React from 'react';
import { ReactComponent as FolderIcon } from '../assets/img/folder.svg';
import { ReactComponent as FileIcon } from '../assets/img/file.svg';
import { ReactComponent as PlusIcon } from '../assets/img/plus.svg';
import { ReactComponent as LaunchIcon} from '../assets/img/launch.svg'
import {Link} from "react-router-dom"
import {useNavigate} from "react-router-dom";
import {connect} from 'react-redux'


export const RepositoryItem = (props) => {

    const [hover, setHover] = React.useState(false)
    const navigator = useNavigate()

    const onAdd = (e, item) => {
            props.dispatch({type:'NEW_DATASET_TO_ADD' ,
                            payload: {
                                    path : item.path,
                                    extension: item.extension
                                    }
                             })

            if(props.onItemAddClick){
                props.onItemAddClick(item)
            }else{
                navigator('/datasets/add-dataset')
            }
    }

    /**
     * Get file extension
     * @param path
     * @returns {string|*}
     */
    const get_extension = (path) => {

        let file = path[path.length-1].split('.')
        if(file.length > 1){
            return file[file.length-1]
        }else {
            return '/'
        }
    }
    return (
        <div 
            className={`repository-item ${props.active ? 'active' : ''}`}
            onClick={() => props.onItemClick(props.indexBar, props.index, props.item.type, props.item.path)}
            >
            { props.item.type === 'dir' ? (
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
                {props.item.name}
            </div>
            {props.item.registered ? (
                <React.Fragment>
                    <div className="icon right action-display" title="This item is registered as dataset">
                        <div className="dot"/>
                    </div>
                    <div className="icon right action-display">
                        <Link to={{pathname: `/datasets/preview/${props.item.registered.dataset_id}`}}>
                            <LaunchIcon/>
                        </Link>
                    </div>
                </React.Fragment>
            ) : (
                <div title="Add as dataset" style={{width:'auto'}} className="icon right action-add"
                     onClick={(event) => onAdd(event,props.item)}>
                    <div className={"select-sm-button"}>
                        {props.actionText}
                    </div>
                </div>
            )}
        </div>
    );
}

export default connect()(RepositoryItem);