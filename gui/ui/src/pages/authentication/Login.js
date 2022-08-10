import React, { useState, useEffect } from 'react';
import {Link, useNavigate} from "react-router-dom";
import axios from 'axios';
import {EP_LOGIN} from '../../constants';
import logo from "../../assets/img/fedbiomed-logo-small.png"
import styles from "./Auth.module.css"
import {useDispatch} from 'react-redux';
import AuthLayout from "./AuthLayout";
import {EuiTitle,
        EuiButton,
        EuiFlexItem,
        EuiFlexGroup,
        EuiForm,
        EuiFormRow,
        EuiFieldPassword,
        EuiFieldText,
        EuiSpacer,
        EuiIcon,
} from '@elastic/eui';

const initialLoginForm = {email: '', password: ''}

const Login = (props) => {

    const navigate = useNavigate();
    const [loginForm, setLoginForm] = useState(initialLoginForm)
    const dispatch = useDispatch();

  /**
   * Login action to send login request to server
   * @param event
   * @param url
   * @param action
   */
    const login = (event) => {

      // Prevent form submit
      event.preventDefault()

      let data = { email: loginForm.email, password: loginForm.password}
      axios.post(EP_LOGIN, data).then((response) => {
          let {access_token, refresh_token} = response.data.result
          props.setToken(access_token, refresh_token)
          navigate('/')
      }).catch((error) => {
        if (error.response) {
          dispatch({type :'ERROR_MODAL', payload: error.response.data.message})
        }else{
          dispatch({type :'ERROR_MODAL', payload: error.toString()})
        }
      })
    }

  /**
   * Handle login form change events
   * @param event
   */
  const handleChange = (event) => {

      let {value, name} = event.target
      setLoginForm(prevNote => ({ ...prevNote, [name]: value}))

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
                                 <EuiFlexItem grow={false} justifyContent="spaceAround">
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
                                 <EuiFlexItem grow={false} justifyContent="spaceAround">
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
