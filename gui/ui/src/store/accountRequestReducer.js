import {GET_USER_REQUESTS} from "./actions/actions";

/**
 * Initial state for user account data format
 * @type {{list: null}}
 */
 const requestsInitialState = {
    requests: [],
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
            return { requests: action.payload}
        default:
            return state
    }
}