import React from 'react'
import { ReactComponent as FolderIcon } from '../../assets/img/folder.svg';
import { ReactComponent as FileIcon } from '../../assets/img/file.svg';
import { ReactComponent as LaunchIcon} from '../../assets/img/launch.svg'

import {Link, useNavigate} from "react-router-dom";
import {useDispatch} from "react-redux";
import {ALLOWED_EXTENSIONS} from "../../constants";
import Button from "../../components/common/Button";

export const RepositoryListRow = (props) => {
     const navigator = useNavigate()
    const dispatch = useDispatch()

    const onAdd = (e, item) => {
        dispatch({type:'NEW_DATASET_TO_ADD' ,
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
        <tr
            className={`${props.active ? 'active' : ''}`}
            onDoubleClick={ () => props.onItemDoubleClick(props.level, props.index, props.item.type, props.item.path)}
            onClick={() =>  props.onItemClick(props.level, props.index, props.item.type, props.item.path) }
        >
            <td className={'name'}>
                <div className={'name'}>
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
                 <div className={'text'}>
                    {props.item.name}
                 </div>
                </div>
            </td>
            <td className={"size"}>{props.item.size}</td>
            <td className={"date"}>{props.item.created}</td>
            <td className={"action"}>
                { displayAdd(props.item) && !props.item.registered ? (
                        <Button title="Add as dataset" style={{margin:0}}
                             onClick={(event) => onAdd(event,props.item)}>
                            <div className={"button"}>
                                {
                                    props.item.type === 'dir' ? 'Add Dataset' : 'Add Dataset'
                                }
                            </div>
                        </Button>
                    ) : null
                }
            </td>
            <td className={'state'}>
                {  props.item.registered ? (
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
                ) : props.item.includes.length > 0 ? (
                    <div className="icon right action-display" title="This item is registered as dataset">
                        <div className="dot empty"/>
                    </div> ) : null
                }
            </td>
        </tr>
    )
}

export default RepositoryListRow