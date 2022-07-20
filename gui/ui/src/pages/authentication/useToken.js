import { useState } from 'react';

function useToken() {

  function getToken() {
    const accessToken = sessionStorage.getItem('accessToken');
    // const refreshToken = sessionStorage.getItem('refreshToken');
    return accessToken && accessToken
  }

  const [accessToken, setToken] = useState(getToken());

  function saveToken(accessToken, refreshToken) {
    sessionStorage.setItem('accessToken', accessToken);
    sessionStorage.setItem('refreshToken', refreshToken);
    setToken(accessToken);
  };

  function removeToken() {
    sessionStorage.removeItem('accessToken');
    sessionStorage.removeItem('refreshToken');
    setToken(null, null);
  }

  return {
    setToken: saveToken,
    accessToken,
    removeToken
  }
}

export default useToken;