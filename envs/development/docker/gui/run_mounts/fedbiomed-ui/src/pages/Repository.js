import React from 'react';
import {connect, useDispatch} from 'react-redux'
import { getFilesFromRepository } from '../store/actions/repositoryActions';
import RepositoryItem from '../components/RepositoryItem';
import RepositoryBar from '../components/RepositoryBar';
import Button, {ButtonsWrapper} from "../components/Button"
import {useNavigate} from "react-router-dom";


const Repository = (props) => {

    const [selected, setSelected] = React.useState(null)
    const [path , setPath ] = React.useState(props.path)
    const navigator = useNavigate()
    const dispatch = useDispatch()
    const frameContent = React.useRef(null)
    const mainRepository = React.useRef(null)

    React.useEffect(() => {
        if(path){
            props.getFiles({path : path})
        }else{
            props.getFiles({path : []})
        }

    }, [props.repository.files.length])


    /**
     * Scroll effect when repository width is greater than
     * frame width
     */
    React.useEffect( () => {
        if(mainRepository.current.scrollWidth > frameContent.current.offsetWidth ){
            frameContent.current.scrollLeft += mainRepository.current.scrollWidth - frameContent.current.offsetWidth
        }
    })


    /**
     * Handling click action on single repository
     * item
     * @param {int} indexBar
     * @param {int} index
     * @param {string} type
     * @param {array} path
     */
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

    /**
     * Stop envet propagation when click on add
     * @param {HTMLDomEvent} event
     */
    const onAddActionClick = (event) => {
        event.stopPropagation()
    }

    /**
     * On repository item selected
     * @returns {object}
     */
    const onSelectClick = () => {

        dispatch({type:'NEW_DATASET_TO_ADD' , payload: selected})

        if(props.onSelect){
            props.onSelect(selected)
        }else{
            navigator('/datasets/add-dataset')
        }
        return selected
    }


    return (
        <React.Fragment>
            { props.mode === null ? (
                <div className="frame-header">
                    <div style={{margin:'0px 0px'}} className={`header-content`}>
                        <p>Following view displays the datafiles saved in the file system where node runs. To load the datafile, please click on Load Dataset button that comes up when you hover the items in the following list.</p>
                        <div className={'note'}>
                            <p>
                                <div style={{display:'inline-block', marginRight: 10}} className="dot"/>
                                Datasets loaded in the node.
                                <div style={{display:'inline-block', marginRight: 10, marginLeft:10}} className="dot empty"/>
                                Folders that includes dataset loaded in the node
                            </p>
                        </div>
                    </div>
                </div>
            ) : null}

            <div ref={frameContent} className="frame-content">
                <div ref={mainRepository} className="main-repository">
                    {Object.keys(props.repository.files).map( (itemBar, key) => {

                        if (itemBar >= props.after ){
                             return (
                                <RepositoryBar key={`bat-${key}`}>
                                    {props.repository.files[itemBar].map( (item,keyChild) => {

                                        if(props.mode === "file-browser" && item.registered ){
                                            return null
                                        }else{
                                            return (
                                                <RepositoryItem
                                                    key={`item-${keyChild}`}
                                                    indexBar={itemBar}
                                                    index={keyChild}
                                                    item={item}
                                                    active={item.active}
                                                    onItemClick={onItemClick}
                                                    onAddActionClick={onAddActionClick}
                                                    onItemAddClick={props.onItemAddClick ? props.onItemAddClick : null}
                                                    actionText={props.actionText ? props.actionText : 'Load Dataset'}
                                                    displayAdd={props.mode === "preview" || props.mode === "file-browser" ? false : true}
                                                />
                                            )
                                        }

                                    })}
                                    {
                                        props.repository.folders[itemBar].displays <  props.repository.folders[itemBar].number ? (
                                                <div className={"end"}>
                                                    Only displaying {props.repository.folders[itemBar].displays} out of {props.repository.folders[itemBar].number}
                                                </div>
                                        ) : null
                                    }
                                </RepositoryBar>
                            )
                        }else{
                            return null
                        }
                    })
                    } 
                </div>  
            </div> 
            <div className="frame-footer">
                { props.mode === "file-browser" ? (
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
                ) : null}

            </div>
        </React.Fragment>
    );
}

Repository.defaultProps = {
    after : 0,
    path  : null,
    mode  : null,
}


const mapStateToProps = (state) => {
    return {
        repository : state.repository
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        getFiles: (data) => dispatch(getFilesFromRepository(data))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(Repository);