import React from 'react';
import { ReactComponent as FolderIcon } from '../../assets/img/folder.svg';
import { ReactComponent as FileIcon } from '../../assets/img/file.svg';
import { ReactComponent as LaunchIcon} from '../../assets/img/launch.svg'
import {Link} from "react-router-dom"
import {useNavigate} from "react-router-dom";
import {connect} from 'react-redux'
import {ALLOWED_EXTENSIONS} from "../../constants";
import Button from "../../components/common/Button"

export const RepositoryItem = (props) => {

    const navigator = useNavigate()
    //const [displayAddButton, setDisplayAddButton] = React.useState(displayAdd(props.item))

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
     * Check display add dataset button will be displayed
     * @param {object} item
     * @returns {boolean}
     */
    const displayAdd = (item) => {
        if(props.mode === 'repository'){
            if(item.type === 'dir'){
                return true
            }else if(item.type === 'file' && ALLOWED_EXTENSIONS.includes(item.extension)){
                return true
            }else{
                return false
            }
        }else{
            return false
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
            ) :  props.item.includes.length > 0 ? (
                <div className="icon right action-display" title="This item is registered as dataset">
                    <div className="dot empty"/>
                </div>
            ) : displayAdd(props.item) ? (
                        <Button title="Add as dataset" style={{width:'auto'}} className="icon right action-add"
                             onClick={(event) => onAdd(event,props.item)}>
                            <div className={"select-sm-button"}>
                                {props.actionText}
                            </div>
                        </Button>
                    ) : null
            }
        </div>
    );
}

export default connect()(RepositoryItem);