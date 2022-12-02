/**
 * Initial state for medical_folder_ref data format
 * @type {{identifiers: {}, format: null, folder_path: null}}
 */
const initialState = {
    medical_folder_root: null,
    modality_folders : null,
    default_modality_names: [],
    current_modality_names: [],
    use_custom_mod2fol: false,
    modalities_mapping: null,
    mod2fol_mapping: null,
    has_all_mappings: false,
    reference_csv: null,
    ignore_reference_csv: false,
    medical_folder_ref : {
        ref : {index: null, name: null},
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
            return {
                ...state,
                medical_folder_root: action.payload.root_path,
                modality_folders: action.payload.modality_folders,
            }
        case "RESET_MEDICAL_FOLDER_ROOT":
            return {
                ...state,
                medical_folder_root: initialState.medical_folder_root,
                modality_folders: initialState.modality_folders,
            }
        case "RESET_MEDICAL_CHANGE_USED_DLP":
            return {
                ...state,
                use_custom_mod2fol: initialState.use_custom_mod2fol,
                //default_modality_names: dont change
                current_modality_names: action.payload, //dont reset to default
                modalities_mapping: {},
                mod2fol_mapping: {},
                has_all_mappings: initialState.has_all_mappings,
                reference_csv: initialState.ref,
                ignore_reference_csv: initialState.ignore_reference_csv,
                medical_folder_ref : initialState.medical_folder_ref,
                metadata : initialState.metadata,
            }
        case "SET_MEDICAL_CHANGE_USED_DLP":
            return {
                ...state,
                use_custom_mod2fol: action.payload.use_custom_mod2fol,
                //default_modality_names: dont change
                current_modality_names: action.payload.current_modality_names, //dont reset to default
                modalities_mapping: action.payload.modalities_mapping,
                mod2fol_mapping: action.payload.mod2fol_mapping,
                has_all_mappings: action.payload.has_all_mappings,
                reference_csv: action.payload.reference_csv,
                ignore_reference_csv: action.payload.ignore_reference_csv,
                medical_folder_ref : initialState.medical_folder_ref,
                metadata : initialState.metadata,
            }

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
                reference_csv : initialState.reference_csv
            }
        case "SET_IGNORE_REFERENCE_CSV":
            return {
                ...state,
                ignore_reference_csv : action.payload
            }
        case "SET_CUSTOMIZE_MOD2FOL":
            return {
                ...state,
                use_custom_mod2fol : action.payload
            }
        case "SET_DEFAULT_MODALITY_NAMES":
            return {
                ...state,
                default_modality_names: action.payload
            }
        case "SET_CURRENT_MODALITY_NAMES":
            return {
                ...state,
                current_modality_names: action.payload
            }
        case "UPDATE_MODALITIES_MAPPING":
            let mapping = state.modalities_mapping
            mapping[action.payload.folder_name] = action.payload.modality_name
            return {
                ...state,
                modalities_mapping: mapping,
            }
        case "UPDATE_MOD2FOL_MAPPING":
            return {
                ...state,
                mod2fol_mapping: action.payload,
            }
        case "UPDATE_HAS_ALL_MAPPINGS":
            return {
                ...state,
                has_all_mappings: action.payload,
            }
        case "CLEAR_MODALITY_MAPPING":
            let mod_mapping = state.modalities_mapping
            delete mod_mapping[action.payload]
            return {
                ...state,
                modalities_mapping: mod_mapping,
            }
        case "RESET_MEDICAL_FOLDER":
            // need to ensure reinitialization of nested objects
            return {
                ...initialState,
                mod2fol_mapping: {},
                modalities_mapping: {}
            }
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

