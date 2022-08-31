import React, { useState } from 'react';
import axios from 'axios'
import {
  EuiButton,
    EuiCallOut,
  EuiModal,
  EuiModalBody,
  EuiModalFooter,
  EuiModalHeader,
  EuiModalHeaderTitle,
  EuiConfirmModal,
  EuiText, EuiToast
} from '@elastic/eui';

import {resetPassword as resetPasswordAction} from "../../store/actions/userManagementActions";
import RegisterFrom from "../authentication/RegisterFrom";
import {EP_RESET_USER_PASSWORD} from "../../constants";
import {USER_MANAGEMENT_ERROR} from "../../store/actions/actions";



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
    const [loading, setLoading] = React.useState(false)
    const [password, setPassword] = React.useState(null)
    const [error, setError] = React.useState(null)

    React.useEffect(() => {
        setShow(props.show)
        setIsPasswordReset(false)
        setError(null)
    }, [props.show])


    /**
     * Reset password handler
     */
    const resetPassword = () => {

        setLoading(true)
        resetPasswordAction(props.userId).then(response => {
                let result = response.data.result
                setPassword(result)
                setIsPasswordReset(true)
                setLoading(false)
        }).catch(error => {
            console.log('HERE')
            setError(`Error while resetting password: ${error.response.data.message ? 
                error.response.data.message : 'unexpected error please contact system provider'}`)
            setLoading(false)
        })
    }

    if(show){
         return (
                <React.Fragment>
                    <EuiModal onClose={props.onClose}>
                        <EuiModalHeader>
                        <EuiModalHeaderTitle>Reset Password?</EuiModalHeaderTitle>
                        </EuiModalHeader>
                        <EuiModalBody>

                            {isPasswordReset ? (
                                    <EuiCallOut
                                        title="Password has been changed!"
                                        color="success"
                                        iconType="alert"
                                        onClose={() => setError(null)}
                                        isExpandable={true}
                                      >
                                         <EuiText>
                                            Password has been changed for user {password.email} as <b>" {password.password} "</b>
                                         </EuiText>
                                         <EuiText>
                                             <b>IMPORTANT: Please make sure you saved the password before closing this window.</b>
                                         </EuiText>
                                    </EuiCallOut>

                                ) : error ? (

                                    <EuiCallOut
                                        title="Opps!"
                                        color="danger"
                                        iconType="alert"
                                        onClose={() => setError(null)}
                                        isExpandable={true}
                                      >
                                     <p>{error}</p>
                                    </EuiCallOut>
                                ) :
                                "Reset password will generate a random password for user. Do you want " +
                                "to reset the password?"}

                        </EuiModalBody>
                        <EuiModalFooter>
                            {!isPasswordReset &&
                                <EuiButton onClick={resetPassword} fill isLoading={loading}>
                                    {loading ? 'Resetting password' : error ? "Try Again" : "Reset Password"}
                            </EuiButton>}
                            <EuiButton onClick={props.onClose} fill>
                                Close
                            </EuiButton>
                        </EuiModalFooter>
                    </EuiModal>

                </React.Fragment>
            )
    }else{
        return null
    }

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
