/**
 * Initial state for BIDS data format
 * @type {{identifiers: {}, format: null, folder_path: null}}
 */
const initialState = {
    data_path: null,
    patient_folders: null,
    format: null,
    reference_csv: null,
    identifiers: null,

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
                data_path: action.payload
            }
        case "PATIENT_FOLDERS":
            return {
               ...state,

            }

        case "SET_FORMAT":

            return {
                ...state,
                patient_folders: action.payload
            }

        case "SET_REFERENCE_CSV":
            return {
                ...state,
                reference_csv : action.payload
            }

        case "SET_IDENTIFIERS":
            return {
                ...state,
                identifiers: action.payload
            }

        default:
            return state
    }
}