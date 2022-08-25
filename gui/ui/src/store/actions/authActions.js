import { isExpired } from "react-jwt";
import {ROLE} from '../../constants';
import {LOGIN} from './actions'

/**
 * Sets user to global state
 * @param data
 * @returns {(function(*): void)|*}
 */
export const setUser = (data) => {
    return (dispatch) => {
        dispatch({type: LOGIN, payload : {
                user_name: data.name ? data.name : '',
                user_surname: data.surname ? data.surname : '',
                role : ROLE[data.role],
                email : data.email
            } })
    }
}

/**
 * Gets access token from session storage
 * @returns {""|string}
 */
export const getAccessToken = () => {
    const accessToken = sessionStorage.getItem('accessToken');
    return accessToken && accessToken
  };

/**
 * Gets refresh token from session store
 * @returns {""|string}
 */
export const getRefreshToken = () => {
  const refreshToken = sessionStorage.getItem('refreshToken');
  return refreshToken && refreshToken
};


/**
 * Sets token to local storage
 * @param accessToken
 * @param refreshToken
 */
export const setToken = (accessToken, refreshToken) => {
    sessionStorage.setItem('accessToken', accessToken);
    sessionStorage.setItem('refreshToken', refreshToken);
};

/**
 * Sets token to local storage
 * @param accessToken
 * @param refreshToken
 */
export const removeToken = () => {
    sessionStorage.removeItem('accessToken');
    sessionStorage.removeItem('refreshToken');
    console.log("token removed")
    alert("User disconnected")
}


/**
 * Checks if session token is still active (has not expired) returns false if token has expired
 * (meaning user should log in again)
 * @returns {boolean} Is token expired
 */
export const checkIsTokenActive = () => {
    const token = getAccessToken();
    let decoded_token

    try {
       decoded_token = JSON.parse(atob(token.split(".")[1]));
    } catch (e) {
       decoded_token = null
    }

    return isExpired(token)
}

/**
 * Reads token properties from the current access token
 * @returns {{object}} User Info
 */
export const decodeToken = () => {

    let access_token = getAccessToken();
    let decoded_token

    // Decode token --------------------------------------------------------
    try {
       decoded_token = JSON.parse(atob(access_token.split(".")[1]));
    } catch (e) {
       decoded_token =  null;
       console.error('Can not parse token!')
    }

    return decoded_token

}