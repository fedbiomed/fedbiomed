import React from 'react';
import {Label, Text, Tag, TextArea, Select} from '../components/Inputs'
import Modal from '../components/Modal'
import Button, {ButtonsWrapper} from '../components/Button'
import {connect, useSelector, useDispatch} from 'react-redux'
import Repository from "../pages/Repository"
import {EP_DATASET_ADD} from "../constants";
import axios from 'axios'
import {addNewDataset} from "../store/actions/datasetsActions";
import {useNavigate} from "react-router-dom";

export const AddDataset = (props) => {


    const [repoModal, setRepoModal] = React.useState(false)
    const [newDataset, setNewDataset] = React.useState({})
    const selectDataType = React.useRef(null)
    const dispatch = useDispatch()
    const [resultModal, setResultModal] = React.useState(false)
    const navigator = useNavigate()

    React.useEffect(() => {
        if(props.addDatasetResult.error === true){
            setResultModal(true)
        }else if(props.addDatasetResult.success === true){
            setResultModal(true)
        }
    }, [props.addDatasetResult.error,  props.addDatasetResult.success])


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

    const onResultModalClose = () => {
        if(props.addDatasetResult.error === true){
            setResultModal(false)
            dispatch({type : 'RESET_ADD_DATASET_RESULT'})
        }else{
            dispatch({type : 'RESET_NEW_DATASET'})
            navigator('/datasets')
        }
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
        console.log(dataset)
        props.addNewDataset(dataset)
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
                        <Select ref={selectDataType} name="type" onChange={onInputValueChange}>
                            {
                                props.new_dataset.extension === ".csv" ?
                                    (
                                         <option value="csv">CSV Dataset</option>
                                    ) : (
                                        <>
                                            <option value="csv">CSV Dataset</option>
                                            <option value="images">Image Dataset</option>
                                        </>
                                    )
                            }
                        </Select>
                    </div>
                </div>
                <div className="row">
                    <div className="form-control">
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
                    />
                </Modal.Content>
            </Modal>
            <Modal show={resultModal} onModalClose={onResultModalClose}>
                <Modal.Header>
                    {
                        props.addDatasetResult.error ? (
                            <h3>Error!</h3>
                        ) : (
                             <h3>Success</h3>
                        )
                    }
                </Modal.Header>
                <Modal.Content>
                    {props.addDatasetResult.message}
                </Modal.Content>
                <Modal.Footer>
                    <ButtonsWrapper className={"float-right"}>
                        <Button onClick={onResultModalClose}>Close</Button>
                    </ButtonsWrapper>
                </Modal.Footer>
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