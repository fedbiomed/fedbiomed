import React from 'react';
import {Label, Text, Tag, TextArea, Select} from '../../components/common/Inputs'
import Modal from '../../components/common/Modal'
import Button, {ButtonsWrapper} from '../../components/common/Button'
import {connect, useDispatch} from 'react-redux'
import Repository from "../repository"
import {addNewDataset, addDefaultDataset} from "../../store/actions/datasetsActions";
import {useNavigate} from "react-router-dom";
import {ADD_DATASET_ERROR_MESSAGES} from "../../constants";
import FileBrowser from "../../components/common/FileBrowser";

export const CommonStandards = (props) => {


    const [repoModal, setRepoModal] = React.useState(false)
    const [newDataset, setNewDataset] = React.useState({})
    const dispatch = useDispatch()
    const navigator = useNavigate()


    // Hook for add dataset
    React.useEffect(() => {
        if(props.addDatasetResult.success === true && props.addDatasetResult.waiting === false ){
            dispatch({type : 'RESET_ADD_DATASET_RESULT'})
            navigator('/datasets')
        }

        if(props.default_dataset.success === true && props.default_dataset.waiting === false){
            dispatch({type : 'RESET_DEFAULT_DATASET_RESULT'})
            navigator('/datasets')
        }

        return () => {
            dispatch({type : 'RESET_NEW_DATASET'})
        }

    }, [props.addDatasetResult.success,
             props.addDatasetResult.waiting,
             props.default_dataset.waiting,
             props.default_dataset.success,
             dispatch,
             navigator,
    ])

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
                if(dataset.type === 'default'){
                    props.addDefaultDataset(dataset)
                }else{
                    props.addNewDataset(dataset)
                }

            }
    }


    const validateInputData = (data) => {
        let field
        let message
        let error

        if (Object.keys(newDataset).length > 0) {
            for(let key=0; key<Object.keys(ADD_DATASET_ERROR_MESSAGES).length; key++){
                field = ADD_DATASET_ERROR_MESSAGES[key].key
                message = ADD_DATASET_ERROR_MESSAGES[key].message
                if(!data[field] || data[field] === "" || data[field] === null || data[field] === undefined){
                    error = message
                    break;
                }
            }
        }else{
            error = 'Please make sure all the fields has been filled'
        }
        return error
    }

    /**
     * On input values has changed
     * @param e
     */
    const onInputValueChange = (e) => {
        let val = e.target.value;
        setNewDataset({
            ...newDataset,
            [e.target.name] : val,
        })
    }

    const onDataTypeChange = (e) => {
        let val = e.target.value
        if(val === 'default'){
            setNewDataset({
                ...newDataset,
                [e.target.name] : val,
                tags : ['#MNIST', '#dataset'],
                name : 'MNIST',
                desc : 'MNIST Default dataset'
            })
        }else{
            setNewDataset({
                ...newDataset,
                [e.target.name] : val
            })
        }
    }

    return (
        <React.Fragment>
            <div className="frame-content">
                <div className="row">
                   <div className={`form-control`}>
                       <Label>Please select datafile or folder you would like to add</Label>
                       <FileBrowser
                        folderPath = {props.new_dataset ? props.new_dataset.path : null}
                        onSelect={onFolderFileSelect}
                        buttonText = "Select Data File/Folder"
                        />
                    </div>
                    <div className="form-control">
                        <Label>Please select data type</Label>
                        <Select name="type" onChange={onDataTypeChange}>
                            {
                                props.new_dataset.extension === ".csv" ?
                                    (
                                         <option value="csv">CSV Dataset</option>
                                    ) : (
                                        <>
                                            <option>Please select...</option>
                                            <option value="default">Default MNIST Dataset</option>
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
                        <Label>Dataset Name <span style={{fontSize:11}}>(min 4 character)</span>
                        </Label>
                        <Text
                            name={"name"}
                            type="text"
                            placeholder="Enter name for dataset"
                            onChange={onInputValueChange}
                            value={newDataset.name}
                        />
                    </div>
                    <div className="form-control" >
                        <Label>Enter tags for dataset <span style={{fontSize:11}}>(Please press enter or space to register tag)</span></Label>
                        <Tag
                            name={"tags"}
                            type="text"
                            onChange={onInputValueChange}
                            placeholder="Enter tags"
                            tags={newDataset.tags}
                        />

                    </div>
                </div>
                <div className={`row`}>
                    <div className="form-control">
                        <Label>Description <span style={{fontSize:11}}>(min 4 character)</span> </Label>
                        <TextArea name="desc"
                                  type="text"
                                  placeholder="Please type a description for dataset"
                                  onChange={onInputValueChange}
                                  value={newDataset.desc}
                        />
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
        addDatasetResult    : state.datasets.add_dataset,
        default_dataset     : state.datasets.default_dataset
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        addNewDataset : (data) => dispatch(addNewDataset(data)),
        addDefaultDataset : (data) => dispatch(addDefaultDataset(data))
    }
}

export default connect(mapStateToProps , mapDispatchToProps)(CommonStandards);