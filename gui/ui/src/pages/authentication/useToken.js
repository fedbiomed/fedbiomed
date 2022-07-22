import { useState } from 'react';

const useToken = () => {

  const getToken = () => {
    const accessToken = sessionStorage.getItem('accessToken');
    // const refreshToken = sessionStorage.getItem('refreshToken');
    console.log("token")
    console.log(accessToken)
    return accessToken && accessToken
  }

  const [accessToken, setToken] = useState(getToken());

  const saveToken = (accessToken, refreshToken) => {
    sessionStorage.setItem('accessToken', accessToken);
    sessionStorage.setItem('refreshToken', refreshToken);
    setToken(accessToken);
  };

  const  removeToken = () => {
    sessionStorage.removeItem('accessToken');
    sessionStorage.removeItem('refreshToken');
    setToken(null, null);
  }

  const checkIsTokenActive = () => {
    // checks if session token is still active (has not expired)
    // returns false if token has expired (meaning user should login again)

    const token = getToken();    
    const decoded_token = null;
    try {
       decoded_token = JSON.parse(atob(token.split(".")[1]));
    } catch (e) {
      decoded_token =  null;
    };
    
    if ((decoded_token !== null) && (decoded_token.exp * 1000 < Date.now())) {
      return true
    } else {
      return false
    }
  }

  return {
    setToken: saveToken,
    accessToken,
    removeToken,
    checkIsTokenActive
  }
}

export default useToken;