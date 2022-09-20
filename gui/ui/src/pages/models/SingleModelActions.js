import React from 'react';
import Button, {ButtonsWrapper} from "../../components/common/Button";
import {connect} from "react-redux"
import {useNavigate} from "react-router-dom";
import Modal from "../../components/common/Modal"
import {approve_model, delete_model, reject_model} from "../../store/actions/modelsActions";
import {Label, TextArea} from "../../components/common/Inputs";

const SingleModelActions = (props) => {

    const {single_model} = props
    const [modal , setModal] = React.useState({show: false, displayNotes:false,
        header: null, approveAction:null, cancelAction:null, content:null})
    const [notes, setNotes] = React.useState(single_model.notes)
    const navigator = useNavigate()

    /**
     * Globally handles approval/rejection operation
     * @param action
     */
    const handleModalApprove = (action) => {

        setModal({...modal, show:false})
        action({model_id: single_model.model_id, notes: notes}, navigator)
    }

    /**
     * Update `notes` state each time user types
     * @param e
     */
    const onNotestChange = (e) => {
        setNotes(e.target.value)
    }

    /**
     * Handles approve operation
     */
    const onApprove = () => {
        setModal({
            show: true,
            approveAction : {text: "Approve Model", type: "positive", action:props.approve},
            cancelAction: {text: "Cancel", type: "negative"},
            header: "Selected model will be approved.",
            displayNotes: true
            }
        )
    }

    /**
     * Handles reject operation
     */
    const onReject = () => {
        setModal({
            show: true,
            approveAction : {text: "Reject Model", type: "negative", action:props.reject},
            cancelAction: {text: "Cancel", type: "positive"},
            header: "Selected model will be approved.",
            displayNotes: true}
        )

    }

    /**
     * Handles delete operation
     */
    const onDelete = () => {
        setModal({
            show: true,
            approveAction: {text: "Yes Delete", type:"negative", action:props._delete},
            cancelAction: {text: "Cancel", type: "positive"},
            header: "Selected model will be deleted are you sure?",
            displayNotes: false}
        )
    }


    /**
     * Handles modal window close action
     */
    const handleClose = () => {
        setModal({...modal, show:false})
    }


    return (
        <div style={props.style}>
            <ButtonsWrapper alignment={'right'}>
                {single_model.model_status === "Pending" ? (
                    <React.Fragment>
                        <Button type={"positive"} onClick={onApprove}>Approve</Button>
                        <Button type={"attention"} onClick={onReject}>Reject</Button>
                    </React.Fragment>
                ) : single_model.model_status === "Approved" ? (
                    <React.Fragment>
                        <Button type={"attention"} onClick={onReject}>Change to Reject</Button>
                    </React.Fragment>
                ) : single_model.model_status === "Rejected" ? (
                    <React.Fragment>
                        <Button type={"positive"} onClick={onApprove}>Change to Approve</Button>
                    </React.Fragment>
                ) : null}
                <Button type={"negative"} disable={single_model.model_type === "default" ? true : false} onClick={onDelete}>Delete</Button>
            </ButtonsWrapper>
            <Modal show={modal.show} width="35%" onModalClose={handleClose} >
                <Modal.Header>
                    {modal.header}
                </Modal.Header>
                <Modal.Content>
                    {modal.displayNotes ? (
                        <React.Fragment>
                            <Label>Please enter or update the notes related to this model</Label>
                            <TextArea
                                style={{width:"inherit"}}
                                name={"notes"}
                                value={notes}
                                onChange={onNotestChange} />
                        </React.Fragment>
                    ): null}
                </Modal.Content>
                <Modal.Footer>
                    <Button type={modal.approveAction?.type} onClick={() => handleModalApprove(modal.approveAction?.action)}>
                        {modal.approveAction?.text}
                    </Button>
                    <Button type={modal.cancelAction?.type} onClick={handleClose}>
                        {modal.cancelAction?.text}
                    </Button>
                    {modal.footer}
                </Modal.Footer>
            </Modal>
        </div>
    );

};

const mapDispatchToPros = (dispatch) => {
    return{
        approve : (data, n) => dispatch(approve_model(data, n)),
        reject : (data, n) => dispatch(reject_model(data, n)),
        _delete : (data, n) => dispatch(delete_model(data, n))
    }
}



export default connect(null, mapDispatchToPros)(SingleModelActions);

