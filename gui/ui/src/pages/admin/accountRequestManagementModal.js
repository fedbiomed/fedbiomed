import React, { useState } from 'react';
import {
  EuiConfirmModal,
} from '@elastic/eui';


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


export  {AccountRequestManagementModal};