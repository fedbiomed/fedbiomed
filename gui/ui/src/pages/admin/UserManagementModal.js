import React, { useState } from 'react';
import {
  EuiButton,
    EuiCallOut,
  EuiModal,
  EuiModalBody,
  EuiModalFooter,
  EuiModalHeader,
  EuiModalHeaderTitle,
  EuiConfirmModal,
  EuiText
} from '@elastic/eui';

import {resetPassword as resetPasswordAction} from "../../store/actions/userManagementActions";
import RegisterFrom from "../authentication/RegisterForm";



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
    const [copied, setCopied] = React.useState(false)


    React.useEffect(() => {
        setShow(props.show)
        setIsPasswordReset(false)
        setError(null)
        setCopied(false)
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
            setError(`Error while resetting password: ${error.response.data.message ? 
                error.response.data.message : 'unexpected error please contact system provider'}`)
            setLoading(false)
        })
    }

    /**
     * Copy auto generated password
     */
    const copyPassword = () => {
        navigator.clipboard.writeText(password.password).then(function() {
              setCopied(true)
        });
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
                                      >
                                     <p>{error}</p>
                                    </EuiCallOut>
                                ) :
                                "Reset password will generate a random password for user. Do you want " +
                                "to reset the password?"}

                        </EuiModalBody>
                        <EuiModalFooter>
                            {!isPasswordReset ? (

                                <EuiButton onClick={resetPassword} fill isLoading={loading}>
                                    {loading ? 'Resetting password' : error ? "Try Again" : "Reset Password"}
                                </EuiButton>


                                ) : (
                                    <EuiButton onClick={copyPassword} disabled={copied ? true : false} iconType={copied ? 'checkInCircleFilled' : 'copyClipboard'} isLoading={loading}>
                                        {copied ? 'Copied' : "Copy Password"}
                                    </EuiButton>
                                )
                                }
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


// this module should be use to update Modal module content
const AccountRequestManagementModal = (props) => {
    const [show, setShow] = useState(props.show)

    React.useEffect(() => {
        // update local state
        setShow(props.show)
    }, [props.show])


    const closeModal = () => {
        setShow(false)
        props.onClose()
    }

    const confirmModal = () => {
        props.onConfirmAccountRequestModal()
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
                        buttonColor={props.coor}
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


export  {
    UserManagementConfirmation,
    UserPasswordResetManagement,
    UserAccountCreation,
    AccountRequestManagementModal
};
