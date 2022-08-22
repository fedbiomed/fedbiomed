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
    use_new_mod2fol_association: false,
    default_modality_names: [],
    modalities_mapping: {},
    mod2fol_mapping: {},
    has_all_mappings: false
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
        case "SET_CREATE_MOD2FOL":
            return {
                ...state,
                use_new_mod2fol_association : action.payload
            }
        case "SET_DEFAULT_MODALITY_NAMES":
            return {
                ...state,
                default_modality_names: action.payload
            }
        case "UPDATE_MODALITIES_MAPPING":
            let mapping = state.modalities_mapping
            mapping[action.payload.folder_name] = action.payload.modality_name

            let m2f = state.mod2fol_mapping
            if(action.payload.modality_name in m2f) {
                if(!m2f[action.payload.modality_name].includes(action.payload.folder_name)){
                    m2f[action.payload.modality_name].push(action.payload.folder_name)
                }
            } else {
                m2f[action.payload.modality_name] = [action.payload.folder_name]
            }

            let has_all_update = true
            for(const folder of state.modalities.values()) {
                if(!mapping[folder]) {
                    has_all_update = false
                    break
                }
            }

            return {
                ...state,
                modalities_mapping: mapping,
                mod2fol_mapping: m2f,
                has_all_mappings: has_all_update,
            }
        case "CLEAR_MODALITY_MAPPING":
            let mod_mapping = state.modalities_mapping
            delete mod_mapping[action.payload]

            let m2f_clear = state.mod2fol_mapping
            for(const modality in m2f_clear) {
                if(m2f_clear[modality].includes(action.payload)){
                    m2f_clear[modality] = m2f_clear[modality].filter(folder => folder != action.payload)
                }
            }

            return {
                ...state,
                modalities_mapping: mod_mapping,
                mod2fol_mapping: m2f_clear,
                has_all_mappings: false,
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

