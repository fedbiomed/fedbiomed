import React from 'react';
import { useLocation } from 'react-router-dom';
import { get } from 'lodash';
import axios from 'axios';
import { getToken, checkIsTokenActive }  from './pages/authentication/tokenFunc';

// const ErrorHandler = ({ children }) => {
//   const location = useLocation();

//   switch (get(location.state, 'errorStatusCode')) {
//     case 404:
//       return <h1>error 404 bro!</h1>;
    
//     // ... cases for other types of errors
      
//     default:
//       return children
//   }
// };

// export default ErrorHandler;


//const { getToken, checkIsTokenActive } = useToken();
// Add a request interceptor
axios.interceptors.request.use(function (req) {
    // Do something before request is sent
    console.log("GOT RESPONSE DATA")
    return req;
  }, function (error) {
    // Do something with request error
    console.log("GOT ERROR")
    console.log(error)

    return Promise.reject(error);
  });

// Add a response interceptor
axios.interceptors.response.use(function (response) {
    // Any status code that lie within the range of 2xx cause this function to trigger
    // Do something with response data
    
    return response;
  }, function (error) {
    // Any status codes that falls outside the range of 2xx cause this function to trigger
    // Do something with response error
    console.log("GOT ERROR 2")
    console.log(error.message)
    console.log(error.request)
    console.log(error.request.status)

    switch (error.request.status){
      case 404:
        // should be handled by React's Router (see App.js)
      console.log("HTTP ERROR 404");
      break;

      case 401:
        // we should differentiate case where token has epxired with case "unsufficient privileged"
        console.log("UNAUTHORIZED");
        let token = getToken();
        let is_token_expired = checkIsTokenActive();

        // let s retrieve token (if any)
        if (token){
          alert("Error 401: sesssion expired, please login again")
          window.location.href = '/login';  // redirect to login page
        }
        break;
    }
    return Promise.reject(error);
  });

  export default axios;
  