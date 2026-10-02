import React, {useState} from 'react';
import {
    EuiButton,
    EuiFieldPassword,
    EuiTitle,
    EuiFlexGroup,
    EuiFlexItem,
    EuiForm,
    EuiFormRow,
    EuiSpacer,
    EuiToast,
    EuiToolTip,
} from "@elastic/eui";
import {EP_UPDATE_PASSWORD} from "../../constants";
import axios from "axios";
import {useDispatch, useSelector} from 'react-redux'
import {useNavigate} from 'react-router-dom'
import {removeToken} from '../../store/actions/authActions'

const initialPassForm = {old_password: '', password: '', confirm: ''}
const initialNotif = {show : false, message : '', title: ''}


const PasswordChange = (props) => {

    const [passForm, setPassForm] = useState(initialPassForm)
    const [notif, setNotif] = useState(initialNotif)
    const notClose = () => setNotif(initialNotif)
    const handleChange = (e) => setPassForm({...passForm, [e.target.name] : e.target.value})
    const resetForm = () => setPassForm(initialPassForm)
    const user = useSelector((state) => state.auth)
    const dispatch = useDispatch()
    const navigate = useNavigate()
    /**
     * On password form is submitted
     * @param e
     */
    const onSubmitNewPassword = (e) => {
        e.preventDefault()

        // check if both password matches, otherwise displays error message
        if (passForm.password === passForm.confirm){
            sendPassword(passForm.old_password, passForm.password , EP_UPDATE_PASSWORD)
        } else {
            setNotif({show:true,  title: "Error", message: 'Please make sure passwords are same.'})
        }
    }

    /**
     * API call for updating password
     * @param old_password
     * @param new_password
     * @param url
     */
    const sendPassword = (old_password, new_password, url) => {
        let data = {email: user.email, password: new_password, old_password: old_password}
        axios.post(url, data)
             .then((response) => {
                 if (props.requiredChange) {
                     dispatch({type: 'LOGOUT'})
                     removeToken(navigate)
                     dispatch({type: 'SUCCESS_MODAL', payload: 'Password changed. Please log in with your new password.'})
                     return
                 }
                 setNotif({show:true,  title: "Success", message: 'Password has been successfully changed.'})
                 resetForm()
                 setTimeout(() => [
                     setNotif(initialNotif)
                 ], 7000)
             })
             .catch((error) => {
                if (error.response) {
                    setNotif({show:true,  title: "Error", message: error.response.data.message })
                }else{
                    setNotif({show:true,  title: "Error", message: 'Incorrect password' + error.toString() })
                }
            })
    }


    return (
        <React.Fragment>
            <EuiTitle>
                <h2>Change/Update Password</h2>
            </EuiTitle>
            {props.requiredChange && <p>You must change your initial or reset password before continuing.</p>}
            <EuiForm component="form"  onSubmit={onSubmitNewPassword} >
                 <EuiFlexGroup direction="column" >
                     <EuiFlexItem grow={false}>
                         {notif.show ? (
                             <React.Fragment>
                                 <EuiSpacer size="l" />
                                 <EuiToast
                                        title={notif.title}
                                        color={notif.title === "Error" ? "danger" : "success"}
                                        iconType="alert"
                                        onClose={notClose}
                                        style={{width:400}}
                                      >
                                     <p>{notif.message}</p>
                                 </EuiToast>
                             </React.Fragment>

                         ) : null}
                     </EuiFlexItem>
                     <EuiFlexItem grow={false}>
                          <EuiFormRow label={props.isAdmin?"Admin password":"Old Password"} hasEmptyLabelSpace>
                             <EuiFieldPassword
                                    type='dual'
                                    name={props.isAdmin?"admin_password":"old_password"}
                                    value={passForm.old_password}
                                    onChange={handleChange}
                              />
                         </EuiFormRow>
                     </EuiFlexItem>
                     <EuiFlexItem grow={false}>
                          <EuiFormRow label={"Password"} hasEmptyLabelSpace>
                            <EuiToolTip
                                  display={"block"}
                                  position="right"
                                  title={"Attention!"}
                                  content="Password must have at least 8 characters,
                                  including an uppercase letter, a lowercase letter and a number"
                            >
                                 <EuiFieldPassword
                                        type='dual'
                                        name={"password"}
                                        value={passForm.password}
                                        onChange={handleChange}
                                  />
                            </EuiToolTip>
                         </EuiFormRow>
                     </EuiFlexItem>
                     <EuiFlexItem grow={false}>
                          <EuiFormRow label={"Confirm Password"} hasEmptyLabelSpace>
                            <EuiToolTip
                                  display={"block"}
                                  position="right"
                                  title={"Attention!"}
                                  content="Password must have at least 8 characters,
                                  including an uppercase letter, a lowercase letter and a number"
                            >
                                 <EuiFieldPassword
                                        type='dual'
                                        name={"confirm"}
                                        value={passForm.confirm}
                                        onChange={handleChange}
                                  />
                            </EuiToolTip>
                         </EuiFormRow>
                     </EuiFlexItem>
                     <EuiFlexItem grow={false} >
                         <EuiFormRow display="center">
                            <EuiButton type="submit" fill>
                                Update Password
                            </EuiButton>
                         </EuiFormRow>
                     </EuiFlexItem>
                 </EuiFlexGroup>
            </EuiForm>
        </React.Fragment>
    );
};

export default PasswordChange;
