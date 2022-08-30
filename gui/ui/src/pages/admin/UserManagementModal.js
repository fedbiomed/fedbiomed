import React, { useState } from 'react';
import {
  EuiButton,
  EuiModal,
  EuiModalBody,
  EuiModalFooter,
  EuiModalHeader,
  EuiModalHeaderTitle,
  EuiConfirmModal,
  EuiText
} from '@elastic/eui';

import RegisterFrom from "../authentication/RegisterFrom";



// this module should be use to update Modal module content
const UserManagementConfirmation = (props) => {
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
        props.onConfirm()
        closeModal()
    }
    if(show){
        return (
            <React.Fragment>
            <div>
                <EuiConfirmModal
                title={props.title}
                 onCancel={closeModal}
                 onConfirm={confirmModal}
                 cancelButtonText="Cancel"
                 confirmButtonText="Confirm"
                 buttonColor={props.color ? props.color : 'danger'}
                 defaultFocusedButton="cancel">
                    <p>{props.text}</p>
                 </EuiConfirmModal>
            </div>
            </React.Fragment>
        )
    }else{
        return null
    }

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


    /**
     * Handler to run after registration is completed without and error
     * @param user {object}
     */
    const afterRegisterHandler = (user) => {
        props.onClose()

        if(props.afterRegister){
            props.afterRegister()
        }
    }


    if(show){
        return (<React.Fragment>
                    <EuiModal padding={'l'} onClose={props.onClose}>
                        <EuiModalHeader>
                          <EuiModalHeaderTitle>
                            <h1>Create new user</h1>
                          </EuiModalHeaderTitle>
                        </EuiModalHeader>
                        <EuiModalBody>
                            <RegisterFrom afterRegister={afterRegisterHandler} navigate_to={false} as={'admin'}/>
                        </EuiModalBody>
                    </EuiModal>
                </React.Fragment>)
    }else{
        return null
    }

}


export  {UserManagementConfirmation, UserPasswordResetManagement, UserPrivilegeManagement, UserAccountCreation};
