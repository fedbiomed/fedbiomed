/**
 * Initial state for medical_folder_ref data format
 * @type {{identifiers: {}, format: null, folder_path: null}}
 */
const initialState = {
    medical_folder_root: null,
    patient_folders: null,
    modalities : null,
    medical_folder_ref : {
        ref : {index: null, name:null},
        subjects: {
            available_subject : null,
            missing_entries: null,
            missing_folders: null
        }
    },
    metadata : {
        name: null,
        tags: null,
        desc: null,
    },
    reference_csv: null,
    ignore_reference_csv: false,
}


/**
 * MedicalFolder format dataset creation global state management
 * @param state
 * @param action
 * @returns {{identifiers, format: null, folder_path: null}|{identifiers: {}, format: null,
 *            folder_path: null}|{identifiers: {}, format: null, folder_path}}
 */
export const medicalFolderReducer = (state = initialState, action) => {

    switch (action.type){
        case "SET_MEDICAL_FOLDER_ROOT":
            return {...state, medical_folder_root: action.payload.root_path, modalities: action.payload.modalities,
            }
        case "RESET_MEDICAL_FOLDER_ROOT":
            return {
                ...state,
                medical_folder_root: null,
                modalities: null,
            }
        case "PATIENT_FOLDERS":
            return {
               ...state,

            }
        case "SET_FORMAT":
            return { ...state, patient_folders: action.payload}

        case "SET_REFERENCE_CSV":
            return {
                ...state,
                reference_csv : action.payload
            }
        case "SET_MEDICAL_FOLDER_METADATA":
            return {
                ...state,
                metadata: {
                    ...state.metadata,
                    ...action.payload
                }
            }
        case "SET_IDENTIFIERS":
            return {
                ...state,
                identifiers: action.payload
            }
        case "SET_MEDICAL_FOLDER_REF":
            return {
                ...state,
                medical_folder_ref: {
                    ref : action.payload.ref,
                    subjects: {
                        available_subjects : action.payload.subjects.available_subjects,
                        missing_entries: action.payload.subjects.missing_entries,
                        missing_folders: action.payload.subjects.missing_folders
                    }
                }}
        case "RESET_MEDICAL_FOLDER_REF":
            return {
                ...state,
                medical_folder_ref : initialState.medical_folder_ref
            }
        case "RESET_MEDICAL_FOLDER_REFERENCE_CSV":
            return {
                ...state,
                medical_folder_ref : initialState.medical_folder_ref,
                reference_csv : null
            }
        case "SET_IGNORE_REFERENCE_CSV":
            return {
                ...state,
                ignore_reference_csv : action.payload
            }
        case "RESET_MEDICAL_FOLDER":
            return initialState
        default:
            return state
    }
}

const medicalFolderPreviewInitialState = {
    subject_table : null,
    modalities : null,
    dataset_id : null,
}

/**
 * State reducer for Medical Folder Dataset preview
 * @param state
 * @param action
 * @returns {{modalities: null, dataset_id: null, subject_table: null}|*}
 */
export const medicalFolderPreviewReducer = (state = medicalFolderPreviewInitialState, action) => {

    switch (action.type){
        case "SET_MEDICAL_FOLDER_PREVIEW":
            return action.payload
        default:
            return state
    }
}

