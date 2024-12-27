import {
    APPROVE_USER_REQUEST,
    GET_USER_REQUESTS,
    GET_USER_REQUESTS_ERROR,
    GET_USER_REQUESTS_LOADING,
    REJECT_USER_REQUEST,
    USER_REQUESTS_ERROR_MESSAGE,
    USER_REQUESTS_SUCCESS_MESSAGE
} from "./actions/actions";

/**
 * Initial state for user account data format
 * @type {{list: null}}
 */
 const requestsInitialState = {
    requests: [],
    error : null,
    loading: false,
    success : null
}

/**
 * Reducer for account creation requests state
 * @param state
 * @param action
 * @returns {{list}}
 */
 export const accountRequestReducer = (state = requestsInitialState, action) => {

    switch (action.type){
        case GET_USER_REQUESTS:
            return { 
                ...state,
                requests: action.payload,
                error: null}

        case GET_USER_REQUESTS_ERROR:
            return {
                ...state,
                requests : [],
                error : action.payload
            }

        case GET_USER_REQUESTS_LOADING:
            return {
                ...state,
                loading : action.payload
            }

        case APPROVE_USER_REQUEST:
            return {
                ...state,
                requests: action.payload,
                error: null,
                loading: false
            }

        case REJECT_USER_REQUEST:
            return {
                ...state,
                requests: action.payload,
                error: null,
                loading: false,
            }

        case USER_REQUESTS_SUCCESS_MESSAGE:
            return {
                ...state,
                error : null,
                success: action.payload
            }
        case USER_REQUESTS_ERROR_MESSAGE:
            return {
                ...state,
                error: action.payload,
                success: null
        }

        default:
            return state
    }
}