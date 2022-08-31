import {APPROVE_USER_REQUEST, APPROVE_USER_REQUEST_ERROR, GET_USER_REQUESTS, GET_USER_REQUESTS_ERROR, 
    GET_USER_REQUESTS_LOADING, REJECT_USER_REQUEST, REJECT_USER_REQUEST_ERROR} from "./actions/actions";

/**
 * Initial state for user account data format
 * @type {{list: null}}
 */
 const requestsInitialState = {
    requests: [],
    error : null,
    loading: false,
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
                error: false,
                loading: false
            }

        case APPROVE_USER_REQUEST_ERROR:
            return {
                ...state,
                error: true
            }

        case REJECT_USER_REQUEST:
            return {
                ...state,
                requests: action.payload,
                error: false,
                loading: false
            }

        case REJECT_USER_REQUEST_ERROR:
            return {
                ...state,
                error: true
            }

        default:
            return state
    }
}