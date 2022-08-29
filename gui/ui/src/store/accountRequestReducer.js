import {GET_USER_REQUESTS, GET_USER_REQUESTS_ERROR, GET_USER_REQUESTS_LOADING} from "./actions/actions";

/**
 * Initial state for user account data format
 * @type {{list: null}}
 */
 const requestsInitialState = {
    requests: [],
    error : null,
    loading: false
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
        default:
            return state
    }
}