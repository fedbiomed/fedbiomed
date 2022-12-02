import React from 'react';
import Repository from "../../pages/repository";
import Modal from "./Modal"
import {Label} from "./Inputs";
import Button from "./Button";
import {EuiTitle} from '@elastic/eui'

const FileBrowser = (props) => {

    const [show, setShow] = React.useState(false)  // define object `setShow` using hook


    /**
     * On folder or file is selected
     * @param {str} selection
     */
    const onSelect = (selection) => {
        props.onSelect(selection)
        setShow(false)
    }


    /**
     * On Modal is closed this method will be triggered
     */
    const onModalClose = () => {
        setShow(false)
    }


    const openRepositoryModal = () => {
        setShow(true)
    }

    return (
        <React.Fragment>
           <div className={`form-control`}>
               {
                   props.label ? (
                        <Label>{props.label}</Label>
                   ) : null
               }
                <div className={"repository-select"}>
                    <Button onClick={openRepositoryModal}>{props.buttonText ? props.buttonText : "Select File"}</Button>
                    <div className={`path`}>
                        { props.folderPath ?  '/'+ props.folderPath.join('/') : null}
                    </div>
                </div>
            </div>
            <Modal show={show} width="90%" onModalClose={onModalClose}>
                    <Modal.Header>
                        <EuiTitle>
                            <h1>Select File or Folder</h1>
                        </EuiTitle>
                    </Modal.Header>
                    <Modal.Content>
                        <Repository
                            onSelect={onSelect}
                            mode={'file-browser'}
                            maxHeight="500px"
                            onlyFolders={props.onlyFolders}
                            onlyExtensions={props.onlyExtensions}
                        />
                    </Modal.Content>
            </Modal>
        </React.Fragment>
    );
};

export default FileBrowser;