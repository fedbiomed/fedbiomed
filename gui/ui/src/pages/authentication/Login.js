import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { EP_LOGIN, EP_REGISTER, LOGIN, REGISTER } from '../../constants';
import Modal from '../../components/common/Modal';
import Button from "../../components/common/Button";


const Login = (props) => {

    const [loginForm, setloginForm] = useState({
      email: '',
      password: ''
    })
    const [message, SetMessage] = useState({show: false, header:"", msg: ""})
    const [side_nav, setSideNav] = useState(null)
    const logMeIn = (event, url, action) => {
      axios({
        method: 'POST',
        url: url,
        data:{
          email: loginForm.email,
          password: loginForm.password
         }
      })
      .then((response) => {
        console.log("recieved request")
        if (action === LOGIN && response.status === 200) {
          props.setToken(response.data.result.access_token, response.data.result.refresh_token)
          side_nav.style.display = "block"; // now display nav_bar if login is successful
        } else if (action === REGISTER && response.status === 201) {
          SetMessage({show:true, header: 'Successfully registered', msg: 'You can now log in !'})
        }
      }).catch((error) => {
        console.log("found error!")
        console.log(error)
        if (error.response) {
          SetMessage({show:true, header: 'An error occured', msg: error.response.data.message})
        }else{
          SetMessage({show:true, header: 'Unexpected Error', msg:error.toString()})
        }
      })

      setloginForm(({
        email: '',
        password: ''}))

      event.preventDefault()
    } 

    const handleChange = (event) => { 
      console.log("into handle change")
      const {value, name} = event.target
      setloginForm(prevNote => ({
          ...prevNote, [name]: value})
      )}

    /**
     * Handles modal window close action
     */
     const handleClose = () => {
      SetMessage({show:false, header: '', msg:''})
  }

  // ------------ getting side bar/nav_bar object ----------------
    const { my_id_html, data } = props;

    useEffect(() => {
        // we hide the navigation tab for the login page
        var my_id_html = document.getElementById("#my_id");  // get the side_nav through its id
        my_id_html.style.display = "none";  // hide side_nav bar when login in
        console.log("inside use effect")
        console.log(my_id_html)
        console.log(message)
        console.log(side_nav)
        setSideNav(my_id_html)  // save ref to object through a hook
        
    }, [data])

    return (
      <React.Fragment>
      <div>
          <h2>Fed-BioMed Node GUI</h2>
          <p>Welcome to Fed-BioMed Node application. In this application, you can manage your data
              files that are deployed in the node or load new datasets into the node.  </p>
          <p>Please Login using your username and password</p>
      </div>
      <div>
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
            {/* <input type='submit'/> */}
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
      </div>
      <div>
          <p> 
              Forgot your password? <a>Please reach here</a>
          </p>
        
      </div>
      {/* Error messsage in case of incorrect login (using Modal component) */}
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
      </React.Fragment>
    );
}

export default Login;
