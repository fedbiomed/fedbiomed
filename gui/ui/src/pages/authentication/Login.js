import React, { useState } from 'react';
import {Link, useNavigate} from "react-router-dom";
import axios from 'axios';
import {EP_LOGIN} from '../../constants';
import logo from "../../assets/img/fedbiomed-logo-small.png"
import styles from "./Auth.module.css"
import {useDispatch} from 'react-redux';
import AuthLayout from "./AuthLayout";
import {setToken, decodeToken, setUser} from "../../store/actions/authActions";
import {
    EuiTitle,
    EuiButton,
    EuiFlexItem,
    EuiFlexGroup,
    EuiForm,
    EuiFormRow,
    EuiFieldPassword,
    EuiFieldText,
    EuiSpacer,
    EuiIcon, EuiToast,
} from '@elastic/eui';
import {SET_LOADING} from "../../store/actions/actions";

const initialLoginForm = {email: '', password: ''}

const Login = (props) => {
    const navigate = useNavigate();
    const [loginForm, setLoginForm] = useState(initialLoginForm)

    // React redux dispatch hook
    const dispatch = useDispatch();

    const [error, setError] = useState({show : false, message : ''})
    const errorClose = () => setError({show:false, message:''})

    /**
    * Handle login form change events
    * @param event
    */
    const handleChange = (e) => setLoginForm({ ...loginForm, [e.target.name]: e.target.value})


    /**
     * Login action to send login request to server
     * @param event
     * @param url
     * @param action
     * */
    const login = (event) => {


        event.preventDefault()

        dispatch({type: SET_LOADING, payload: {status: true, text: 'Login....'}})
        let data = { email: loginForm.email, password: loginForm.password}
        axios.post(EP_LOGIN, data).then((response) => {
            let {access_token, refresh_token} = response.data.result

            // Sets token to session store
            setToken(access_token, refresh_token)

            // Saves user info into global state
            dispatch(setUser(decodeToken()))

            // Navigate to home page
            navigate('/')
            dispatch({type: SET_LOADING, payload: {status: false, text: null}})

          }).catch((error) => {
                if (error.response) {
                    setError({show:true, message: error.response.data.message})
                }else{
                    setError({show:true, message: error.toString()})
                }
                dispatch({type: SET_LOADING, payload: {status: false, text:null}})
          })
    }

    return (
            <AuthLayout>
                <div className={styles.centered}>
                    <img className={styles.imgLogo} alt="fedbiomed-logo" src={logo}/>
                    <EuiTitle>
                        <h1>Fed-BioMed</h1>
                    </EuiTitle>
                    <EuiSpacer size="m" />
                    <p>Please login to access Node management panel </p>
                </div>
                <EuiFlexGroup justifyContent="spaceAround">
                    <EuiFlexItem style={{width: 400}}>
                        <EuiForm component="form"  onSubmit={login} >
                             <EuiFlexGroup direction="column" >
                                 <EuiFlexItem grow={false}>
                                 {error.show ? (
                                     <React.Fragment>
                                         <EuiSpacer size="l" />
                                         <EuiToast
                                                title="Error"
                                                color="danger"
                                                iconType="alert"
                                                onClose={errorClose}
                                                style={{width:400}}
                                              >
                                             <p>{error.message}</p>
                                         </EuiToast>
                                     </React.Fragment>

                                 ) : null}
                                 </EuiFlexItem>
                                 <EuiFlexItem grow={false}>
                                    <EuiFormRow label={"Enter your e-mail"} hasEmptyLabelSpace>
                                          <EuiFieldText
                                              onChange={handleChange}
                                              text={loginForm.email}
                                              name='email'
                                              prepend={<EuiIcon type="email" />}
                                              value={loginForm.email}
                                           />
                                    </EuiFormRow>
                                 </EuiFlexItem >
                                 <EuiFlexItem grow={false}>
                                      <EuiFormRow label={"Password"} hasEmptyLabelSpace>
                                         <EuiFieldPassword
                                                type='dual'
                                                name={"password"}
                                                value={loginForm.password}
                                                onChange={handleChange}
                                          />
                                     </EuiFormRow>
                                 </EuiFlexItem>
                                 <EuiFlexItem grow={false} >
                                     <EuiFormRow display="center">
                                        <EuiButton type="submit" fill>
                                            Login
                                        </EuiButton>
                                     </EuiFormRow>
                                 </EuiFlexItem>
                             </EuiFlexGroup>
                        </EuiForm>
                    </EuiFlexItem>
                </EuiFlexGroup>
                <EuiFlexGroup direction={'column'}>
                    <EuiFlexItem>
                        <p>Don't you have an account? <Link to={"/register"}>Create one!</Link></p>
                    </EuiFlexItem>
                </EuiFlexGroup>
            </AuthLayout>
    );
}

export default Login;
