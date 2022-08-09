import React, { useState, useEffect } from 'react';
import { useNavigate } from "react-router-dom";
import axios from 'axios';
import { EP_LOGIN, EP_REGISTER, LOGIN, REGISTER } from '../../constants';
import logo from "../../assets/img/fedbiomed-logo-small.png"
import styles from "./Login.module.css"
import {useDispatch} from 'react-redux';
import {
  EuiFlexItem,
  EuiFlexGroup,
  EuiPanel,
  EuiPageHeader,
  EuiCode,
  EuiText,
} from '@elastic/eui';

const initialLoginForm = {email: '', password: ''}

const Login = (props) => {

  const navigate = useNavigate();
    const [loginForm, setLoginForm] = useState(initialLoginForm)
    const dispatch = useDispatch();
    const [message, SetMessage] = useState("Unknown error")
    const [side_nav, setSideNav] = useState(null)

  /**
   * Login action to send login request to server
   * @param event
   * @param url
   * @param action
   */
    const logMeIn = (event, url, action) => {

      // Prevent form submit
      event.preventDefault()

      let data = { email: loginForm.email, password: loginForm.password}
      axios.post(url, data).then((response) => {
        if (action === LOGIN && response.status === 200) {
          props.setToken(response.data.result.access_token, response.data.result.refresh_token)
        } else if (action === REGISTER && response.status === 201) {
          dispatch({type :'SUCCESS_MODAL', payload:'Successfully registered You can now log in !'})
        }
        // Redirect to home
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
      setLoginForm(prevNote => ({
          ...prevNote, [name]: value})

      )}


  const Header = (
          <div className={styles.headerWrapper}>
                <img className={styles.imgLogo} alt="fedbiomed-logo" src={logo}/>
                <span>Fed-BioMed Node Management Panel</span>
           </div>
  )

    return (
        <React.Fragment>
            <EuiFlexGroup justifyContent="spaceAround">
                 <EuiFlexItem grow={false}>
                     <EuiPageHeader
                            bottomBorder
                            className={styles.header}
                            paddingSize={"s"}
                            pageTitleProps={{size: "s"}}
                            pageTitle={Header}
                            description={"Welcome to Fed-BioMed Node application. " +
                                "In this application, you can manage your data files " +
                                "that are deployed in the node or load new datasets into the node"}
                     />
                     <EuiPanel>
                          <h1>Login</h1>
                            <form className='login' onSubmit={logMeIn}>
                            <div className='input-container'>
                              <label>Email</label>
                              <input
                                  onChange={handleChange}
                                  type='email'
                                  text={loginForm.email}
                                  name='email'
                                  placeholder='Email'
                                  value={loginForm.email}
                                  required
                              />
                            </div>
                            <div className='input-container'>
                              <label>Password</label>
                              <input
                                  onChange={handleChange}
                                  type='password'
                                  text={loginForm.password}
                                  name='password'
                                  placeholder='Password'
                                  value={loginForm.password}
                                  required
                              />
                            </div>
                            <div className='button-container'>
                              <button type="submit" onClick={(e) => {
                                logMeIn(e, EP_LOGIN, LOGIN);
                                }}
                              >Sign in</button>
                            </div>
                            <div className='button-container'>
                              <p>Not Yet registered ? Please sign up !</p>
                              <button type="submit" onClick={(e) => {
                                logMeIn(e, EP_REGISTER, REGISTER);
                                }}
                              >Sign up</button>
                            </div>
                          </form>
                        <div>
                            <p>
                                Forgot your password? <a href="there">Please reach here</a>
                            </p>

                        </div>
                     </EuiPanel>
                 </EuiFlexItem>
             </EuiFlexGroup>
        </React.Fragment>
    );
}

export default Login;
