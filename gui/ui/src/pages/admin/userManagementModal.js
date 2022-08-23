import React, { useState } from 'react';
import {
  EuiButton,
  EuiModal,
  EuiModalBody,
  EuiModalFooter,
  EuiModalHeader,
  EuiModalHeaderTitle,
  EuiCodeBlock,
  EuiSpacer,
  PropertySortType,
  EuiFlyoutBody,
  EuiConfirmModal
} from '@elastic/eui';

import PasswordChange from '../authentication/PasswordChange';

// this module should be use to update Modal module content
const UserManagementModal = (props) => {
    const [show, setShow] = React.useState(props.show)
    //const [isModalVisible, setIsModalVisible] = useState(false);


    React.useEffect(() => {
        // update local state
        setShow(props.show)
    }, [props.show])


    // const closeModal = () => setIsModalVisible(false);
    // const showModal = () => setIsModalVisible(true);    
    const closeModal = () => {
        setShow(false)
        props.onClose()
    }

    const confirmModal = () => {
        console.log("doc action")
        closeModal()
    }
    return (
        <React.Fragment>
        <div>

            <EuiConfirmModal
            title={props.title}
             onCancel={closeModal}
             onConfirm={confirmModal}
             cancelButtonText="Cancel"
             confirmButtonText="Confirm"
     
             buttonColor="danger"
             defaultFocusedButton="cancel">
                <p>{props.text}</p>
             </EuiConfirmModal>

        </div>
        </React.Fragment>
    )
}

const UserPasswordResetManagement = (props) => {
    const [show, setShow] = React.useState(props.show);
    React.useEffect(() => {
        // update local state
        setShow(props.show)
    }, [props.show])

    return (
        <React.Fragment>
            <EuiFlyoutBody>
                <PasswordChange/>
            </EuiFlyoutBody>
        </React.Fragment>
    )
}

export  {UserManagementModal, UserPasswordResetManagement};
