import React, { useState, useEffect, useRef } from 'react';


const Login = (props) => {


    const [errorMessages, setErrorMessages] = useState({});
    const renderErrorMessage = (name) =>
        { name === errorMessages.name && (
            <div className="error">{errorMessages.message}</div>)
        };
    console.log("is active")
    const computedClassName = props.active ? 'active' : 'muted';

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

            <div className="form">
                    <form onSubmit={handleSubmit}>
                        <div className="input-container">
                        <label>Username </label>
                        <input type="text" name="uname" required />
                        {renderErrorMessage("uname")}
                        </div>
                        <div className="input-container">
                        <label>Password </label>
                        <input type="password" name="pass" required />
                        {renderErrorMessage("pass")}
                        </div>
                        <div className="button-container">
                        <input type="submit" />
                        </div>
                    </form>
            </div>

            <div>
                <p> 
                    Forgot your password? <a>Please reach here</a>
                </p>
            </div>

            
        </React.Fragment>
    )
}


export default Login