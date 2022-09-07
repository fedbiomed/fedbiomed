import React, {useState} from 'react';
import {EuiModal, EuiModalHeader, EuiModalHeaderTitle, EuiModalBody, EuiButtonEmpty, EuiButton, EuiModalFooter }
    from "@elastic/eui"
import {useDispatch, useSelector} from "react-redux";


const ResultModal = (props) => {

    const [isModalVisible, setIsModalVisible] = useState(props.show);
    const result = useSelector((state) => state.list)

    const closeModal = () => setIsModalVisible(false);



    return (
      <EuiModal onClose={closeModal} initialFocus="[name=popswitch]">
        <EuiModalHeader>
          <EuiModalHeaderTitle>
            <h1>{props.title}</h1>
          </EuiModalHeaderTitle>
        </EuiModalHeader>
        <EuiModalBody>{props.body}</EuiModalBody>
        <EuiModalFooter>
          <EuiButton  onClick={closeModal} fill>
            Close
          </EuiButton>
        </EuiModalFooter>
      </EuiModal>
    );
};

export default ErrorModel;