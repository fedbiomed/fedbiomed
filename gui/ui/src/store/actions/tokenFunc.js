import { isExpired, decodeToken } from "react-jwt";
import {ROLE} from '../../constants';

const getAccessToken = () => {
    const accessToken = sessionStorage.getItem('accessToken');
    // const refreshToken = sessionStorage.getItem('refreshToken');
    console.log("token")
    console.log(accessToken)
    return accessToken && accessToken
  };


const getRefreshToken = () => {
  const refreshToken = sessionStorage.getItem('refreshToken');
  return refreshToken && refreshToken
};


const checkIsTokenActive = () => {
    // checks if session token is still active (has not expired)
    // returns false if token has expired (meaning user should login again)

    const token = getAccessToken();
    
      
    let decoded_token = null;
    try {
       decoded_token = JSON.parse(atob(token.split(".")[1]));
    } catch (e) {
      decoded_token =  null;
    };
    let val = isExpired(token)
    console.log(val)
    return val
    
    // if ((decoded_token !== null) && (decoded_token.exp > Date.now())) {
    //   return true
    // } else {
    //   return false
    // }
  }


const readToken = () => {
  // reads token properties from the current access token

  let access_token = getAccessToken();
  let decoded_token = null;
    try {
       decoded_token = JSON.parse(atob(access_token.split(".")[1]));
    } catch (e) {
      decoded_token =  null;
    };
  console.log(decoded_token)
  const info_retirever = {'username': 'email',
                          'role': 'role'
                           }
  var user_info = {}
  if (decoded_token){
    for (const [key, value] of Object.entries(info_retirever)) {
    try{
      if (key === 'role'){
        user_info['role'] = ROLE[decoded_token['role']]
      }else{
        user_info[key] = decoded_token[value]

      }
      

    }catch(e){
      console.log(e)
    }
  }
  }
  
  return user_info

}
  export  {getAccessToken};
  export  {checkIsTokenActive};
  export {getRefreshToken};
  export {readToken};