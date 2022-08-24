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
  EuiFlyout,
  EuiFlyoutBody,
  EuiConfirmModal,
  EuiText
} from '@elastic/eui';

import Register from '../authentication/Register';

// this module should be use to update Modal module content
const UserManagementModal = (props) => {
    const [show, setShow] = React.useState(props.show)
   
    React.useEffect(() => {
        // update local state
        setShow(props.show)
    }, [props.show])

  
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
    const [isPasswordReset, setIsPasswordReset] = React.useState(false);
    React.useEffect(() => {
        // update local state
        setShow(props.show)
    }, [props.show])

    const closeModal = () => {
        setShow(false)
        props.onClose()
    }

    const ResetPassword = () => {
        setIsPasswordReset(true)
    }
    let message;  // message to display to the user, once 

    console.log(props.user)
    if (props.displayPassword){
        message = "New temporary password for user " + props.user[0] + props.user[1] +"is (password)"
    }else{
        message = "Temporary password sent to user : !"
    }

    return (
        <React.Fragment>
            <EuiModal>
                <EuiModalHeader>
                <EuiModalHeaderTitle>Reset Password?</EuiModalHeaderTitle>
                </EuiModalHeader>
                <EuiModalBody>
                    {isPasswordReset?<EuiText>{message}</EuiText>:"Reset password will generate a random password for user. Do you want to reset the password?"}
                </EuiModalBody>
                <EuiModalFooter>
                    {!isPasswordReset && <EuiButton onClick={ResetPassword} fill>
                        Reset Password
                    </EuiButton>}
                    <EuiButton onClick={closeModal} fill>
                        Close
                    </EuiButton>
                
                </EuiModalFooter>
            </EuiModal>

        </React.Fragment>
    )
}

const UserPrivilegeManagement = (props) => {
    const [show, setShow] = React.useState(props.show);
    React.useEffect(() => {
        // update local state
        setShow(props.show)
    }, [props.show])

    return (<React.Fragment>
        <EuiModal>
            
        </EuiModal>
        </React.Fragment>)
}

const UserAccountCreation = (props) => {
    const [show, setShow] = React.useState(props.show);
    React.useEffect(() => {
        // update local state
        setShow(props.show)
    }, [props.show])

    return (<React.Fragment>
        <EuiModal>
            <Register/>
        </EuiModal>
        </React.Fragment>)
}
export  {UserManagementModal, UserPasswordResetManagement, UserPrivilegeManagement, UserAccountCreation};
