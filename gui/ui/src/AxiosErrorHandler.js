import React from 'react';
import { useLocation, useNavigate} from 'react-router-dom';
import { get } from 'lodash';
import axios from 'axios';
import {EP_REFRESH} from './constants';
import { getToken, checkIsTokenActive, getRefreshToken }  from './store/actions/tokenFunc';
import { createBrowserHistory } from 'history';
import { configure } from '@testing-library/react';
import { store } from './index';
//import GetLastWebpageUrl from './utils';

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


  //let navigate = useNavigate();
  const handleTokenExpiration = (msg=null) =>
  {
    // logic for handling token expiration after a 401 HTTP error request (unauthorized)
    if (msg === null){
      alert("Error 401: sesssion expired, please login again")
    }else{

      alert(msg)
    }
    
    window.location.href = '/login';  // redirect to login page 
  }
  // Add a request interceptor
  axios.interceptors.request.use(function (req) {
      // Do something before request is sent
      console.log("GOT RESPONSE DATA")
      console.log(req.url)
      const token = getToken();
      if (token && req.url !== EP_REFRESH){
        // set headers as required by jst_extended library
        // (flask server side)
        req.headers.Authorization = `Bearer ${token}`;
        console.log(req.headers.Authorization)
      }
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
      // var my_id_html = document.getElementById("#my_id"); 
      // my_id_html.style.display = "block";
      return response;
    }, function (error) {
      // Any status codes that falls outside the range of 2xx cause this function to trigger
      // Do something with response error
      const history = new createBrowserHistory();
      console.log(error.response.data)

      //return new Promise((resolve, reject) => {
        return new Promise((resolve, reject) => {
        console.log(error.response)
        switch (error.request.status){
          case 404:
            // should be handled by React's Router (see App.js)
            reject(error)
            console.log("HTTP ERROR 404");
            break;
  
          case 401:
            // we should differentiate case where token has epxired with case "unsufficient privileged"
            store.dispatch({type: "LOGOUT"})
            console.log("UNAUTHORIZED");
            let access_token = getToken();
            let is_token_expired = checkIsTokenActive();
  
            // let s retrieve token (if any)
            if (access_token){
  
              if (is_token_expired){
                let refresh_token = getRefreshToken();
                console.log("refresh token")
                console.log(refresh_token)
                if (error.response.config.url !== EP_REFRESH){
                  const originalRequest = error.config;
                  axios.post(EP_REFRESH,{'hello': 'you'}, {headers: 
                    {
                      'Authorization':  `Bearer ${refresh_token}`
                    }
                  })
                  .then(res => {
                    console.log("working")
                    console.log(res)
                    let new_access_token = res.data.result.access_token;
                    let new_refresh_token = res.data.result.refresh_token;
                    sessionStorage.setItem('accessToken', new_access_token);
                    sessionStorage.setItem('refreshToken', new_refresh_token);
                    

                    store.dispatch({type: "LOGIN"})
                    // window.location.reload();
                    resolve(axios(originalRequest))
                  })
                  .catch(rf_error => {
                    
                    reject(rf_error);
                    // send a refresh token
                      alert(error.response.data.message)
                  })
                } else {
                  handleTokenExpiration(error.response.data.message)
                }
                
                
                
              }
              else{
                alert("Unsufficient privileges")
                //redirect to previous page
                history.back()
              }
            }else{
              let link = window.location.href.toString().split(window.location.host)[1];
              if ( (link !== '/login') && ( link !== '/login/')){
                handleTokenExpiration(error.response.data.message)
              }
  
            }
            break;
          default:
            reject(error)
            break;
        }
      })
    });


export default axios;

  