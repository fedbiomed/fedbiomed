import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { EP_LOGIN } from '../../constants';


function Login(props) {

    const [loginForm, setloginForm] = useState({
      email: '',
      password: ''
    })

    function logMeIn(event) {
      axios({
        method: 'POST',
        url: EP_LOGIN,
        data:{
          email: loginForm.email,
          password: loginForm.password
         }
      })
      .then((response) => {
        props.setToken(response.data.result.access_token, response.data.result.refresh_token)
      }).catch((error) => {
        // How to display error message to user ?
        console.log(error)
      })

      setloginForm(({
        email: '',
        password: ''}))

      event.preventDefault()
    }

    function handleChange(event) { 
      const {value, name} = event.target
      setloginForm(prevNote => ({
          ...prevNote, [name]: value})
      )}

    const handleSubmit = (event) => {
        // Prevent page reload
        event.preventDefault();
      };


    const { my_id_html, data } = props;

    useEffect(() => {
        // we hide the navigation tab for the login page
        var my_id_html = document.getElementById("#my_id");  // get the side_nav through its id
        my_id_html.style.display = "none";
        console.log("inside use effect")
        console.log(my_id_html)
        
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
            <input type='submit' />
          </div>
        </form>
      </div>
      <div>
          <p> 
              Forgot your password? <a>Please reach here</a>
          </p>
        </div>
      </React.Fragment>
    );
}

export default Login;



// const Login = (props) => {

//     const [errorMessages, setErrorMessages] = useState({});
//     const renderErrorMessage = (name) =>
//         { name === errorMessages.name && (
//             <div className='error'>{errorMessages.message}</div>)
//         };
//     console.log('is active')
//     const computedClassName = props.active ? 'active' : 'muted';
//     console.log(computedClassName)
//     const handleSubmit = (event) => {
//         // Prevent page reload
//         event.preventDefault();
//     };

//     const hideMainSideBar = () => {
//         return <div className='main-side-bar' style={{'display': 'none'}}/>


//     }
//     return (
//         <React.Fragment>
//             <div>
//                 <h2>Fed-BioMed Node GUI</h2>
//                 <p>Welcome to Fed-BioMed Node application. In this application, you can manage your data
//                     files that are deployed in the node or load new datasets into the node.  </p>
//             </div>
//             {hideMainSideBar}
//             <div className='form'>
//                     <form onSubmit={handleSubmit}>
//                         <div className='input-container'>
//                         <label>Username </label>
//                         <input type='text' name='username' required />
//                         {renderErrorMessage('username')}
//                         </div>
//                         <div className='input-container'>
//                         <label>Password </label>
//                         <input type='password' name='pass' required />
//                         {renderErrorMessage('pass')}
//                         </div>
//                         <div className='button-container'>
//                         <input type='submit' />
//                         </div>
//                     </form>
//             </div>

            
//         </React.Fragment>
//     )
// }



// export default Login