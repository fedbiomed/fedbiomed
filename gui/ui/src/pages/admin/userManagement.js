import axios from "axios";
import {EP_ADMIN} from '../../constants';
import React, { useState } from 'react';


const UserManagement = (props) => {

    const [isadmin, setIsAdmin] = useState(false)
    const isAccessGranted = () => {
        axios.get(EP_ADMIN).then((response) => {
            
            setIsAdmin(true)
        }).catch((error) => {
            alert(error.response.data.message)
            
        })
        
    }
    return (
        
        <div>
            {isAccessGranted()}
            {isadmin?<h1>
                This is User Management webpage. Only admin should be able to reach this page</h1>:
                <h1>You are simple user. you cannot access this page</h1>}
        </div>
    )
}

export default UserManagement;