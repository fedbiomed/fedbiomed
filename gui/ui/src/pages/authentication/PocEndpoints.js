import { useState } from 'react';
import { EP_PROTECTED, EP_ADMIN } from '../../constants';
import axios from 'axios';
import Modal from '../../components/common/Modal';
import Button from "../../components/common/Button";

const PocEndpoints = (props) => {

    const [message, SetMessage] = useState({show: false, header:"", msg: ""})
    const token = props.accessToken; // sessionStorage.getItem('accessToken');
    const get_protected_data = () => {
        axios.get(EP_PROTECTED, {
            // TODO: Find a way to add headers automatically to all requests
            headers: {
                'Authorization': `Bearer ${token}` 
            }
        })
            .then( response => {
                console.log(response) 
            })
            .catch( (error) => {
                if (error.response) {
                    SetMessage({show:true, header: 'An error occured', msg: error.response.data.message})
                  }else{
                    SetMessage({show:true, header: 'Unexpected Error', msg:error.toString()})
                  }
            })
    }

    const get_admin_data = () => {
        axios.get(EP_ADMIN, {
            headers: {
                'Authorization': `Bearer ${token}` 
            }
        })
            .then( response => {
                console.log(response) 
            })
            .catch( (error) => {
                if (error.response) {
                    SetMessage({show:true, header: 'An error occured', msg: error.response.data.message})
                  }else{
                    SetMessage({show:true, header: 'Unexpected Error', msg:error.toString()})
                  }
            })
    }

    const handleClose = () => {
        SetMessage({show:false, header: '', msg:''})
    }

    return (
        <div>
            <div>
                <button onClick={get_protected_data}>Get Protected Data</button>
            </div>
            <div>
                <button onClick={get_admin_data}>Get Admin Data</button>
            </div>
            <Modal show={message.show} onModalClose={handleClose}>
                <Modal.Header>
                {message.header}
                </Modal.Header>
                <Modal.Content>
                    {/* If first connection, please register.
                    Please contact your local administrator for further details */}
                    {message.msg}
                </Modal.Content>
                <Modal.Footer>
                    <Button onClick={handleClose}>
                        close
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
      );
}

export default PocEndpoints;