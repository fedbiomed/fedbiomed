import axios from "axios"
import { isExpired } from "react-jwt";
import {EP_AUTH, ROLE} from '../../constants';
import {LOGIN, SET_LOADING} from './actions'



export const autoLogin = (navigate) => {

    return (dispatch) => {
        dispatch({type: SET_LOADING, payload: {status: true }})

        axios.get(EP_AUTH, {}).then(response => {
            dispatch(setUser(response.data.result))
            dispatch({type: SET_LOADING, payload: {status: false}})
        }).catch(error => {
            dispatch({type: SET_LOADING, payload: {status: false}})
            navigate('/login')
            dispatch({type: 'ERROR_MODAL', payload: "Your session is expired please login"})
        })
    }
}

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
    return accessToken
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
export const removeToken = (navigate) => {
    sessionStorage.removeItem('accessToken');
    sessionStorage.removeItem('refreshToken');
    navigate('/login')
}


/**
 * Checks if session token is still active (has not expired) returns false if token has expired
 * (meaning user should log in again)
 * @returns {boolean} Is token expired
 */
export const checkIsTokenActive = () => {
    const token = getAccessToken();
    return isExpired(token)
}

/**
 * Reads token properties from the current access token
 * @returns {{object}|null} User Info
 */
export const decodeToken = () => {

    let access_token = getAccessToken();
    let decoded_token

    if(access_token){
        // Decode token --------------------------------------------------------
        try {
           decoded_token = JSON.parse(atob(access_token.split(".")[1]));
        } catch (e) {
           decoded_token =  null;
           console.error('Can not parse token!')
        }
        return decoded_token
    }else{
        return null
    }

}