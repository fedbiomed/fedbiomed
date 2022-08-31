// Models Actions
export const LIST_MODELS = "LIST_MODELS"
export const APPROVE_MODEL = "APPROVE_MODEL"
export const SINGLE_MODEL = "SINGLE_MODEL"
export const RESET_SINGLE_MODEL = "SINGLE_MODEL"
export const LOGIN = "LOGIN"

export const LIST_USERS = 'LIST_USERS'
export const USER_MANAGEMENT_ERROR = 'USER_MANAGEMENT_ERROR'
export const LIST_USERS_LOADING = 'LIST_USERS_LOADING'
export const USER_MANAGEMENT_SUCCESS_MESSAGE = 'USER_MANAGEMENT_SUCCESS_MESSAGE'
export const GET_USER_REQUESTS = "GET_REQUESTS"
export const GET_USER_REQUESTS_ERROR = "GET_REQUESTS_ERROR"
export const GET_USER_REQUESTS_LOADING = "GET_REQUESTS_LOADING"

export const APPROVE_USER_REQUEST = "APPROVE_REQUEST"
export const APPROVE_USER_REQUEST_ERROR = "APPROVE_REQUEST_ERROR"

export const REJECT_USER_REQUEST = "REJECT_REQUEST"
export const REJECT_USER_REQUEST_ERROR = "REJECT_REQUEST_ERROR"


export const SET_LOADING = "SET_LOADING"

/**
 * Dispatch action the display global error modal window
 * @param error
 * @param message
 * @returns {(function(*): void)|*}
 */
export const displayError = (error, message = "") => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: false}})
        if(error.response){
            if(error.response.data.message){
                dispatch({type: 'ERROR_MODAL', payload: message + error.response.data.message})
            }else{
                dispatch({type: 'ERROR_MODAL', payload: 'Undefined error: ' + error.toString()})
            }
        }
    }
}