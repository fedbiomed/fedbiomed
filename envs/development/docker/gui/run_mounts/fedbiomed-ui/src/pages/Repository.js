import React from 'react';
import {connect} from 'react-redux'
import { getFilesFromRepository } from '../store/actions/repositoryActions';
import RepositoryItem from '../components/RepositoryItem';
import RepositoryBar from '../components/RepositoryBar';
import Button, {ButtonsWrapper} from "../components/Button"
import {useNavigate} from "react-router-dom";

const Repository = (props) => {

    const [selected, setSelected] = React.useState(null)
    const navigator = useNavigate()

    React.useEffect(() => {

        props.getFiles({path : []})
    }, [props.repository.files.length])


    const onItemClick = (indexBar, index, type, path) => {

        if(type === "dir") {
            props.getFiles({path: path})
        }

        let indexOld = props.repository.files[indexBar].findIndex( x => x.active === true)
        if(indexOld > -1){
            props.repository.files[indexBar][indexOld].active = false
        }
        props.repository.files[indexBar][index].active = true
        setSelected(props.repository.files[indexBar][index])
    }


    const onAddActionClick = (event, type, path) => {
        event.stopPropagation()
    }

    /**
     * On repository item selected
     * @returns {object}
     */
    const onSelectClick = () => {

        props.dispatch({type:'NEW_DATASET_TO_ADD' , payload: selected.path})

        if(props.onSelect){
            props.onSelect(selected)
        }else{
             navigator('/datasets/add-dataset')
        }

        return selected
    }


    return (
        <React.Fragment>
            <div className="frame-header">
                <div style={{margin:'0px 0px'}} className={`header-content`}>
                    <p>
                        You can add new dataset by clicking Add Dataset button.
                    </p>
                </div>
            </div>
            <div className="frame-content">
                <div className="main-repository">
                    {Object.keys(props.repository.files).map( (item, key) => {
                        return (
                            <RepositoryBar key={`bat-${key}`}>            
                                {props.repository.files[item].map( (item,keyChild) => {
                                
                                    return (
                                        <RepositoryItem 
                                            key={`item-${keyChild}`}
                                            indexBar={key}
                                            index={keyChild}
                                            item={item}
                                            active={item.active}
                                            onItemClick={onItemClick}
                                            onAddActionClick={onAddActionClick}
                                            onItemAddClick={props.onItemAddClick ? props.onItemAddClick : null}
                                            actionText={props.actionText ? props.actionText : 'Add Dataset'}

                                        />
                                    )
                                })}
                            </RepositoryBar> 
                        )
                    })
                    } 
                </div>  
            </div> 
            <div className="frame-footer">
                <ButtonsWrapper className={"float-right"}>
                    {
                        selected ? (
                            <Button
                                style={{'min-width' : '100px'}}
                                onClick={onSelectClick}
                            >{
                                selected.type === 'file' ? (
                                    'Select File'
                                ):
                                selected.type === 'dir' ?
                                   'Select Folder'
                                 : null
                            }
                            </Button>
                        ) : null
                    }

                </ButtonsWrapper>
            </div>
        </React.Fragment>
    );
}

const mapStateToProps = (state) => {
    return {
        repository : state.repository
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        getFiles: (data) => dispatch(getFilesFromRepository(data)),
        dispatch: dispatch
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(Repository);