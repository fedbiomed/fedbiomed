/**
 * Initial state for user account data format
 * @type {{list: null}}
 */
 const requestsInitialState = {
    list: null,
}

/**
 * Reducer for account creation requests state
 * @param state
 * @param action
 * @returns {{list}}
 */
 export const accountRequestReducer = (state = requestsInitialState, action) => {

    switch (action.type){
        case "GET_REQUESTS":
            console.log("requests")
            return { ...state, list: action.payload}
        default:
            return state
    }
}