/**
 * Initial state for user account data format
 * @type {{identifiers: {}, format: null, folder_path: null}}
 */
 const requestsInitialState = {
    user_requests_list: null,
}

/**
 * Reducer for account creation requests state
 * @param state
 * @param action
 * @returns {{single_model: null, list}|{single_model: null, list: null}|{single_model, list: null}}
 */
 export const accountRequestReducer = (state = requestsInitialState, action) => {

    switch (action.type){
        case 'GET_REQUESTS':
            return { ...state, user_requests_list: action.payload}
        default:
            return state
    }
}