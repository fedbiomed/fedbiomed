/**
 * Initial state for BIDS data format
 * @type {{identifiers: {}, format: null, folder_path: null}}
 */
const initialState = {
    folder_path: null,
    format: null,
    identifiers: {}
}


/**
 * BIDS format dataset creation global state management
 * @param state
 * @param action
 * @returns {{identifiers, format: null, folder_path: null}|{identifiers: {}, format: null,
 *            folder_path: null}|{identifiers: {}, format: null, folder_path}}
 */
export const bidsReducer = (state = initialState, action) => {


    switch (action.type){
        case "SET_FOLDER_PATH":
            return {
                ...state,
                folder_path: action.payload
            }

        case "SET_FORMAT":

            return {
                ...state
            }

        case "SET_IDENTIFIERS":
            return {
                ...state,
                identifiers: action.payload
            }

        default:
            break
    }
}