// Models Actions
export const LIST_MODELS = "LIST_MODELS"
export const APPROVE_MODEL = "APPROVE_MODEL"
export const SINGLE_MODEL = "SINGLE_MODEL"
export const RESET_SINGLE_MODEL = "SINGLE_MODEL"
export const LOGIN = "LOGIN"

export const LIST_USERS = 'LIST_USERS'
export const LIST_USERS_ERROR = 'LIST_USERS_ERROR'

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