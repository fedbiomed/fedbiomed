import { isExpired, decodeToken } from "react-jwt";

const getToken = () => {
    const accessToken = sessionStorage.getItem('accessToken');
    // const refreshToken = sessionStorage.getItem('refreshToken');
    console.log("token")
    console.log(accessToken)
    return accessToken && accessToken
  };

  const checkIsTokenActive = () => {
    // checks if session token is still active (has not expired)
    // returns false if token has expired (meaning user should login again)

    const token = getToken();
    
      
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

  export  {getToken};
  export  {checkIsTokenActive};