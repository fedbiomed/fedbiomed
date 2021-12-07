import React from 'react';
import {Label, Text, Tag, TextArea, Select} from '../components/Inputs'
import Modal from '../components/Modal'
import Button from '../components/Button'


export const AddDataset = (props) => {


    const [repoModal, setRepoModal] = React.useState(false)

    /**
     * Open modal window to select data folder 
     * or data  file
     */
    const openRepositoryModal = () => {

    }


    /**
     * Selected file from repository, can be folder too
     * @param {string} file 
     */
    const selectFile = (file) => {

    }

    return (
        <React.Fragment>
            <div className="frame-header">
                <h2>Add New Dataset</h2>   
            </div>
            <div className="frame-content">
                <div className="row">
                    <div className="form-control">
                        <Label>Dataset Name</Label>
                        <Text type="text" placeholder="Enter name for dataset" />
                    </div>
                    <div className="form-control">
                        <Label>Description</Label>
                        <TextArea type="text" placeholder="Enter name for dataset" />
                    </div>
                </div>
                <div className="row">
                    <div className="form-control" >
                        <Label>Enter tags for dataset</Label>
                        <Tag type="text" placeholder="Enter name for dataset" />
                    </div>
                    <div className="form-control">
                        <Label>Please select data type</Label>
                        <Select>
                            <option value="csv">CSV Dataset</option>
                            <option value="image">Image Dataset</option>
                        </Select>
                    </div>
                </div>
                <Button onClick={openRepositoryModal}>Select Data File</Button>
            </div>
            <div className="frame-footer">

            </div>
            <Modal show={repoModal} width="90%">
                <Modal.Header></Modal.Header>
            </Modal>
        </React.Fragment>
    );
}

export default AddDataset;