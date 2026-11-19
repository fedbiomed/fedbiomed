
import axios from 'axios';
import {EP_REFRESH, EP_LOGIN} from './constants';
import { getAccessToken, checkIsTokenActive, getRefreshToken }  from './store/actions/authActions';

// this handler wraps axios logic request:
// it does mainly 2 things:
// 1. formats request with correct header (for token auth)
// 2. contains logic for getting refresh tokens, and ensuring idle user are disconnected


// Add a request interceptor
axios.interceptors.request.use(function (req) {

  const token = getAccessToken();
  if (token && req.url !== EP_REFRESH){
    // set headers as required by jst_extended library (flask server side)
    req.headers.Authorization = `Bearer ${token}`;
  }
  return req;
}, function (error) {
  // Do something with request error
  return Promise.reject(error);
});


export const setupAxios = () => {

      // Add a response interceptor
    axios.interceptors.response.use(function (response) {
      // Any status code that lie within the range of 2xx cause this function to trigger
      // Do something with response data

      return response;
    }, function (error) {

      return new Promise((resolve, reject) => {
          switch (error.request.status){
              case 404:
                // should be handled by router (see App.js)
                reject(error)
                break;

              case 401:
                // If endpoint is login
                if(error.response.config.url === EP_LOGIN){
                  reject(error)
                  break;
                }

              case 422:
                // we should differentiate case where token has expired with case "insufficient privileged"
                let access_token = getAccessToken();
                let is_token_expired = checkIsTokenActive();
                // let s retrieve token (if any)
                if (access_token){
                  if (is_token_expired){
                    let refresh_token = getRefreshToken();
                    if (error.response.config.url !== EP_REFRESH){
                      const originalRequest = error.config;
                      axios.get(EP_REFRESH,
                       {headers:
                        {
                          'Authorization':  `Bearer ${refresh_token}`
                        }
                      })
                      .then(res => {

                        let new_access_token = res.data.result.access_token;
                        let new_refresh_token = res.data.result.refresh_token;
                        sessionStorage.setItem('accessToken', new_access_token);
                        sessionStorage.setItem('refreshToken', new_refresh_token);

                        resolve(axios(originalRequest))
                      })
                      .catch(rf_error => {
                        // at this point, refresh token should have expired (as well as access token)
                        reject(rf_error);
                        window.location.href = '/login'
                      })
                    } else {
                      // case where refresh token has expired
                      reject(error);
                    }
                  }else{
                    sessionStorage.removeItem('accessToken');
                    sessionStorage.removeItem('refreshToken');
                    window.location.href = '/login'
                  }
                }else{
                  sessionStorage.removeItem('accessToken');
                  sessionStorage.removeItem('refreshToken');
                  window.location.href = '/login'
                }
                break;
              default:
                reject(error)
                break;
            }
        })
    });
}


  