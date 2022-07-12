import React from 'react';
import axios from 'axios';


// const [errorMessages, setErrorMessages] = useState({});
// const [isSubmitted, setIsSubmitted] = useState(false);

// const renderErrorMessage = (name) =>
//   name === errorMessages.name && (
//     <div className="error">{errorMessages.message}</div>
//   );

//   const handleSubmit = (event) => {
//     // Prevent page reload
//     event.preventDefault();
//   };

/*
const Login = (props) => {
    return (
        <React.Fragment>
            <div className="frame-header" style={{textAlign:'center'}}>
                <h2>Fed-BioMed Node GUI</h2>
                <p>Welcome to Fed-BioMed Node application. In this application, you can manage your data
                    files that are deployed in the node or load new datasets into the node. <br></br>Please login first </p>
            </div>
            <div>
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
            </div>
        </React.Fragment>
    )
}
*/
const Login = (props) => {


return (
    <React.Fragment>
        <div>
            <h2>Fed-BioMed Node GUI</h2>
            <p>Welcome to Fed-BioMed Node application. In this application, you can manage your data
                files that are deployed in the node or load new datasets into the node.  </p>
        </div>
        
    </React.Fragment>
)
}



export default Login