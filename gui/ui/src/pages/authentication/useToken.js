import { useState } from 'react';

import { getToken, checkIsTokenActive } from '../../store/actions/tokenFunc';

const useToken = () => {

  //const [accessToken, setToken] = useState(null);

  
  const [accessToken, setToken] = useState(getToken()); 

  const saveToken = (accessToken, refreshToken) => {
    sessionStorage.setItem('accessToken', accessToken);
    sessionStorage.setItem('refreshToken', refreshToken);
    console.log("token saved")
    setToken(accessToken);
  };

  const  removeToken = () => {
    sessionStorage.removeItem('accessToken');
    sessionStorage.removeItem('refreshToken');
    console.log("TOKEN REMOVED")
    setToken(null);
    alert("User deconnected")
  }
  

  return {
    setToken: saveToken,
    accessToken,
    removeToken,
    getToken,
    checkIsTokenActive
  }
}

export default useToken;