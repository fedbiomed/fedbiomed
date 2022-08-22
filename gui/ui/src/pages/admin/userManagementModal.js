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
  EuiConfirmModal
} from '@elastic/eui';


const UserManagementModal = (props) => {
    const [show, setShow] = React.useState(props.show)
    //const [isModalVisible, setIsModalVisible] = useState(false);


    React.useEffect(() => {
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
             defaultFocusedButton="cancel"/>

        </div>
        </React.Fragment>
    )
}

export default UserManagementModal;