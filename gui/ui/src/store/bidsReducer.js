/**
 * Initial state for BIDS data format
 * @type {{identifiers: {}, format: null, folder_path: null}}
 */
const initialState = {
    bids_root: null,
    patient_folders: null,
    modalities : null,
    bids_ref : {
        ref : {index: null, name:null},
        subjects: {
            available_subject : null,
            missing_entries: null,
            missing_folders: null
        }
    },
    reference_column_name:null,
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
        case "SET_BIDS_ROOT":
            return {
                ...state,
                bids_root: action.payload.root_path,
                modalities: action.payload.modalities,
            }
        case "RESET_BIDS_ROOT":
            return {
                ...state,
                bids_root: null,
                modalities: null,
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
        case "SET_BIDS_REF":
            return {
                ...state,
                bids_ref: {
                    ref : action.payload.ref,
                    subjects: {
                        available_subjects : action.payload.subjects.available_subjects,
                        missing_entries: action.payload.subjects.missing_entries,
                        missing_folders: action.payload.subjects.missing_folders
                    }
                }}

        case "RESET_BIDS_REF":
            return {
                ...state,
                bids_ref : initialState.bids_ref
            }
        default:
            return state
    }
}