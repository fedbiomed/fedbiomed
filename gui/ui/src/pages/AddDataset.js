import React from 'react';
import {Label, Text, Tag, TextArea, Select} from '../components/Inputs'
import Modal from '../components/Modal'
import Button, {ButtonsWrapper} from '../components/Button'
import {connect, useDispatch} from 'react-redux'
import Repository from "../pages/Repository"
import {addNewDataset} from "../store/actions/datasetsActions";
import {useNavigate} from "react-router-dom";
import {ADD_DATASET_ERROR_MESSAGES} from "../constants";

export const AddDataset = (props) => {


    const [repoModal, setRepoModal] = React.useState(false)
    const [newDataset, setNewDataset] = React.useState({})
    const dispatch = useDispatch()
    const navigator = useNavigate()

    React.useEffect(() => {
        if(props.addDatasetResult.success === true && props.addDatasetResult.waiting === false ){
            dispatch({type : 'RESET_ADD_DATASET_RESULT'})
            navigator('/datasets')
        }

        return () => {
            dispatch({type : 'RESET_NEW_DATASET'})
        }

    }, [props.addDatasetResult.success, dispatch, navigator, props.addDatasetResult.waiting])


    /**
     * Open modal window to select data folder 
     * or data  file
     */
    const openRepositoryModal = () => {
        setRepoModal(true)
    }

    const handleModalClose = () => {
        setRepoModal(false)
    }

    /**
     * Selected file from repository, can be folder too
     * @param {string} file 
     */
    const onItemAddClick = ( item) => {
        dispatch({type : 'NEW_DATASET_TO_ADD' ,
                        payload : {
                                path : item.path,
                                extension: item.extension
                                }
                        })
        setRepoModal(false)
    }

    /**
     * On user click select folder
     * @param selected
     */
    const onFolderFileSelect = (selected) => {
        dispatch({type : 'NEW_DATASET_TO_ADD' ,
                payload : {path : selected.path, extension: selected.extension}})

        setRepoModal(false)

    }

    /**
     * When user click on add dataset
     */
    const onDataAdd = () => {
            let dataset = newDataset
            dataset.path = props.new_dataset.path
            if( props.new_dataset.extension === ".csv"){
                dataset.type = 'csv'
            }
            let validation = validateInputData(dataset)
            if(validation){
                 dispatch({type :'ERROR_MODAL', payload: validation})
            }else{
                props.addNewDataset(dataset)
            }
    }


    const validateInputData = (data) => {
        let field
        let message
        let error
        if (Object.keys(newDataset).length > 0) {
            if(!data.path){
                error = 'Please select a dataset'
            }else{
                for(let key=0; key<Object.keys(ADD_DATASET_ERROR_MESSAGES).length; key++){
                    field = ADD_DATASET_ERROR_MESSAGES[key].key
                    message = ADD_DATASET_ERROR_MESSAGES[key].message
                    if(!data[field] || data[field] === "" || data[field] === null || data[field] === undefined){
                        error = message
                        break;
                    }
                }
            }
        }else{
            error = 'Please make sure all the fields has been field'
        }


        return error
    }

    /**
     * On input values has changed
     * @param e
     */
    const onInputValueChange = (e) => {
        setNewDataset({
            ...newDataset,
            [e.target.name] : e.target.value
        })
    }

    return (
        <React.Fragment>
            <div className="frame-content">
                <div className="row">
                   <div className={`form-control`}>
                       <Label>Please select datafile or folder you would like to add</Label>
                        <div className={"repository-select"}>
                            <Button onClick={openRepositoryModal}>Select Data File</Button>
                            <div className={`path`}>
                                { props.new_dataset.path ?  '/'+ props.new_dataset.path.join('/'): null}
                            </div>
                        </div>
                    </div>
                    <div className="form-control">
                        <Label>Please select data type</Label>
                        <Select name="type" onChange={onInputValueChange}>
                            {
                                props.new_dataset.extension === ".csv" ?
                                    (
                                         <option value="csv">CSV Dataset</option>
                                    ) : (
                                        <>
                                            <option>Please select...</option>
                                            <option value="csv">CSV Dataset</option>
                                            <option value="images">Image Dataset</option>
                                        </>
                                    )
                            }
                        </Select>
                    </div>
                </div>
                <div className="row">
                    <div className="form-control" >
                        <Label>Dataset Name</Label>
                        <Text
                            name={"name"}
                            type="text"
                            placeholder="Enter name for dataset"
                            onChange={onInputValueChange} />
                    </div>
                    <div className="form-control" >
                        <Label>Enter tags for dataset</Label>
                        <Tag
                            name={"tags"}
                            type="text"
                            onChange={onInputValueChange}
                            placeholder="Enter name for dataset" />
                    </div>
                </div>
                <div className={`row`}>
                    <div className="form-control">
                        <Label>Description</Label>
                        <TextArea name="desc"
                                  type="text"
                                  placeholder="Enter name for dataset"
                                  onChange={onInputValueChange} />
                    </div>
                </div>
                <ButtonsWrapper className={"float-right"}>
                    <Button onClick={onDataAdd}>Add Dataset</Button>
                </ButtonsWrapper>
            </div>
            <div className="frame-footer">

            </div>
            <Modal show={repoModal} width="90%" onModalClose={handleModalClose}>
                <Modal.Header>
                    <h1>Select File or Folder</h1>
                </Modal.Header>
                <Modal.Content>
                    <Repository
                        onItemAddClick={onItemAddClick}
                        onSelect={onFolderFileSelect}
                        mode={'file-browser'}
                        maxHeight="400px"
                    />
                </Modal.Content>
            </Modal>
        </React.Fragment>
    );
}

const mapStateToProps = (state) => {
    return {
        new_dataset         : state.datasets.new_dataset,
        addDatasetResult    : state.datasets.add_dataset
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        addNewDataset : (data) => dispatch(addNewDataset(data))
    }
}

export default connect(mapStateToProps , mapDispatchToProps)(AddDataset);