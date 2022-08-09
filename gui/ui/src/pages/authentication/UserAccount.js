import React, {useState} from 'react';
import  {TableInfo, TableCol, TableRow} from "../../components/common/Tables";
import { readToken } from '../../store/actions/tokenFunc';
import { EP_UPDATE_PASSWORD } from '../../constants';
import Button from '../../components/common/Button';
import Modal from '../../components/common/Modal';
import {TextArea, Text, Label} from '../../components/common/Inputs';
import {connect, useDispatch} from 'react-redux';
import axios from 'axios';

const UserAccount = () => {
    // displays webpage for user account management 

    const [newPassword, setNewPassword] = useState({show: false})
    //const [text, setText] = useState("")
    const [userInfo, setUserInfo] = useState(readToken());


    var initial_passwords = {'password1': '',
                             'password2': ''}

    const [password, setPassword] = useState(initial_passwords);

    const dispatch = useDispatch();

    const onChangePassword = (key, target_value) => {
        // displays the Modal window for changing password

        setNewPassword({show:true})
        // console.log(event.target.name)
        // console.log(event.target.value)
        setPassword({...password, [key]: target_value})
        //setText(target_value)
        
    }

    const onSubmitNewPassword = () => {

        // check if both password matches, otherwise displays error message
        if (password['password1'] === password['password2']){
            
            sendPassword(password['password1'] , EP_UPDATE_PASSWORD)

        } else {
            dispatch({type :'ERROR_MODAL', payload:'Passwords do not match! Please try again'})
            setNewPassword({...newPassword, ['show']:true})
        }

    }

    const sendPassword = (new_password, url) => {
        // sends HTTP POST for updating password
        let user_email  = userInfo['username'];
        console.log(userInfo)
        console.log(new_password)
        axios.post(url, {email: user_email, 
                         password: new_password})
             .then((response) => {
                dispatch({type :'SUCCESS_MODAL', payload:'New Password changed!'})
                setNewPassword({...newPassword, ['show']:false})
                // TODO: should we Send request or info to administrator about this password change?
             })
             .catch((error) => {
                alert(error.toString())

                if (error.response) {
                    dispatch({type :'ERROR_MODAL', payload: error.response.data.message})
                }else{
                dispatch({type :'ERROR_MODAL', payload:'Incorrect password' + error.toString()})
                }
            })

    }

    const onCloseModal = () => {
        // Closes modal window (change password window)
        setNewPassword({show:false})
        setPassword(initial_passwords)
    }

    const feedUserData = () =>{
        // collects access token info about user and adds them into a dictionary.
        // Adds a field password mapping a `Button` component
        const user_info = userInfo;
        user_info['password'] = <Button onClick={onChangePassword}>Change</Button>
        return user_info

    }
    console.log("password")
    console.log(password)

    //setText("")

    return (
        <React.Fragment> 
            <div>
                <h2>
                    User account
                </h2>
                <h4>
                    This page contains user account info
                </h4>
                
                <TableRow>
                    <TableCol>
                        <TableInfo info={feedUserData()} mode={false}/>
                        
                    </TableCol>
                </TableRow>
                <h4>To change your username, please contact your administrator</h4>
            </div>
            <div >
                <Modal show={newPassword.show} onModalClose={onCloseModal}>
                    <Modal.Header><h2>Change Password</h2></Modal.Header>
                    <Modal.Content>
                        <Label>Submit new password</Label>
                        <Text placeholder={"Enter password"}  type="password" value={password.password1} onChange={(e) => onChangePassword("password1", e.target.value)} minlength='10' required>
                            {"Enter password"}
                        </Text>
                        <Label>write again new password (both of them should match)</Label>
                        <Text placeholder={"Enter password"} type="password"  value={password.password2} onChange={(e) => onChangePassword("password2", e.target.value)}  required minlength='10'>
                            {"Enter password"}
                        </Text >
                        <Button onClick={onSubmitNewPassword}>Confirm</Button>
                        <Button onClick={onCloseModal}>Cancel</Button>
                    </Modal.Content>
                </Modal>
            </div>
        </React.Fragment> 
        
    )
}

export default UserAccount