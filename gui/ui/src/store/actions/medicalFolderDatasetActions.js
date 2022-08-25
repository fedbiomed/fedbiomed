import axios from "axios";
import {
    EP_REPOSITORY_LIST,
    EP_LOAD_CSV_DATA,
    EP_VALIDATE_MEDICAL_FOLDER_ROOT,
    EP_VALIDATE_REFERENCE_COLUMN,
    EP_ADD_MEDICAL_FOLDER_DATASET,
    EP_PREVIEW_MEDICAL_FOLDER_DATASET,
    EP_DEFAULT_MODALITY_NAMES,
} from "../../constants";
import {displayError} from "./actions";

/**
 * Sets Folder Path
 * @param path
 * @returns {(function(*): void)|*}
 */
export const setFolderPath = (path) => {
    return (dispatch) => {
        if(path.type !== "dir"){
            dispatch({type: 'ERROR_MODAL' , payload: "ROOT path for MedicalFolder dataset should be folder/directory"})
            return
        }
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Setting/validating MedicalFolder root folder path"}})
        axios.post(EP_VALIDATE_MEDICAL_FOLDER_ROOT, {medical_folder_root : path.path})
            .then(response => {
                let data = response.data.result
                if(data.valid){
                    dispatch({type:'RESET_DATA_LOADING_PLAN'})
                    dispatch({type:'RESET_MEDICAL_FOLDER'})
                    dispatch({type: "SET_MEDICAL_FOLDER_ROOT", payload: { root_path: path.path, modality_folders: data.modalities}})
                    dispatch({type:'SET_LOADING', payload: {status: false}})
                    dispatch(checkSubDirectories(path.path))
                }else{
                    dispatch({type:'SET_LOADING', payload: {status: false}})
                    dispatch({type:'RESET_MEDICAL_FOLDER'})
                    dispatch({type: 'ERROR_MODAL', payload: data.message})
                }
            }).catch(error => {
                dispatch({type:'SET_LOADING', payload: {status: false}})
                dispatch({type:'RESET_MEDICAL_FOLDER', payload: false})
                dispatch(displayError(error))
        })



    }
}

/**
 * Set reference column that corresponds patient folders
 * @param ref
 * @returns {(function(*): void)|*}
 */
export const setFolderRefColumn = (ref) => {
    return (dispatch, getState) => {
        // TODO: Validate selected column corresponds patient folders
        let state = getState()
        let reference_csv = state.medicalFolderDataset.reference_csv.path
        let root = state.medicalFolderDataset.medical_folder_root
        let validation_data =  {
            reference_csv_path: reference_csv,
            medical_folder_root: root,
            index_col: ref.index
        }

        dispatch({type:'SET_LOADING', payload: {status: true, text: "Setting/validating MedicalFolder subject reference column..."}})
        axios.post(EP_VALIDATE_REFERENCE_COLUMN, validation_data).then(response => {
            let data = response.data.result
            if (data.valid) {
                let payload = {
                    ref: ref,
                    subjects: { available_subjects: data.subjects.intersection,
                                missing_entries: data.subjects.missing_entries,
                                missing_folders: data.subjects.missing_folders
                    }
                }
                dispatch({type: "SET_MEDICAL_FOLDER_REF", payload: payload})
            }else{
                dispatch({type: "RESET_MEDICAL_FOLDER_REF"})
                dispatch({type: "ERROR_MODAL", payload: data.message})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Can not verify reference column for MedicalFolder dataset. Error message is: "))
        })
    }
}

/**
 * Sets reference csv file for MedicalFolder
 * @param path
 * @returns {(function(*): void)|*}
 */
export const setReferenceCSV = (path) => {
    return (dispatch) => {

        dispatch({type:'SET_LOADING', payload: {status: true, text: "Setting reference/demographics dataset for MedicalFolder..."}})
        axios.post(EP_LOAD_CSV_DATA, {path : path.path}).then( response => {
            if(response.status === 200){
                let data = response.data.result
                dispatch({type: "SET_REFERENCE_CSV", payload: { path: path.path, data: data}})
            }else{
                dispatch({type: 'ERROR_MODAL', payload: response.data.result.message})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while validating reference CSV file"))
        })
        return
    }
}

/**
 * Sets MedicalFolder dataset meta information dataset name, tags etc.
 * @param data
 * @returns {(function(*))|*}
 */
export const setMedicalFolderDatasetMetadata = (data) => {
    return (dispatch) => {
        dispatch({type: "SET_MEDICAL_FOLDER_METADATA", payload: data})
    }
}


export const setIgnoreReferenceCsv = (data) => {
    return (dispatch) => {
        dispatch({type: "SET_IGNORE_REFERENCE_CSV", payload: data})
    }
}

export const setCustomizeModalitiesToFolders = (value) => {
    return (dispatch) => {
        dispatch({type: 'SET_CUSTOMIZE_MOD2FOL', payload: (value.target.value === 'true' ? true : false)})
    }
}

//export const createModalitiesToFoldersBlock = (modalities_mapping) => {
//    return (dispatch) => {
//        dispatch({type: 'SET_DLP', payload: -1})
//        dispatch({type:'SET_LOADING', payload: {status: true, text: "Saving Association"}})
//        axios.post(EP_LOADING_BLOCK_MOD2FOL_CREATE, {mapping: modalities_mapping}).then(response => {
//            dispatch({type: 'ADD_PIPELINE',
//                      payload: {type_id: 'modalities_to_folders',
//                                serial_id: response.data.result.serial_id,
//                                module: response.data.result.module,
//                                qualname: response.data.result.qualname,
//                                }})
//            dispatch({type:'SET_LOADING', payload: {status: false}})
//        }).catch(error => {
//            dispatch({type:'SET_LOADING', payload: {status: false}})
//            dispatch(displayError(error, "Error while getting default modality names."))
//})
//    }
//}

export const initModalityNames = () => {
    return (dispatch, getState) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Loading default modality names."}})
        axios.get(EP_DEFAULT_MODALITY_NAMES).then(response => {
            dispatch({type: 'SET_DEFAULT_MODALITY_NAMES', payload: response.data.result.default_modalities})
            if(getState().medicalFolderDataset.current_modality_names.length === 0) {
                dispatch({type: 'SET_CURRENT_MODALITY_NAMES', payload: response.data.result.default_modalities})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while getting default modality names."))
        })
    }
}

export const updateModalitiesMapping = (data) => {
    return (dispatch, getState) => {
        dispatch({type: 'UPDATE_MODALITIES_MAPPING', payload: {folder_name: data.folder_name, modality_name: data.modality_name} })
        
        let medical_folder = getState().medicalFolderDataset
        let mapping = medical_folder.modalities_mapping
        let m2f = medical_folder.mod2fol_mapping
        if(data.modality_name in m2f) {
            if(!m2f[data.modality_name].includes(data.folder_name)){
                m2f[data.modality_name].push(data.folder_name)
            }
        } else {
            m2f[data.modality_name] = [data.folder_name]
        }
        dispatch({type: 'UPDATE_MOD2FOL_MAPPING', payload: m2f })

        let has_all_mappings = true
        for(const folder of medical_folder.modality_folders.values()) {
            if(!mapping[folder]) {
                has_all_mappings = false
                break
            }
        }
        dispatch({type: 'UPDATE_HAS_ALL_MAPPINGS', payload: has_all_mappings})

        let current_modalities = getState().medicalFolderDataset.current_modality_names
        let has_modality = false
        for(const mod of current_modalities.values()) {
            if(data.modality_name === mod['label']){
                has_modality = true
            }
        }
        if(!has_modality) {
            current_modalities.push({'label': data.modality_name, 'value': data.modality_name})
        }
        dispatch({type: 'SET_CURRENT_MODALITY_NAMES', payload: current_modalities})
    }
}

export const clearModalityMapping = (folder_name) => {
    return (dispatch, getState) => {
        dispatch({type: 'CLEAR_MODALITY_MAPPING', payload: folder_name})

        let medical_folder = getState().medicalFolderDataset
        let m2f = medical_folder.mod2fol_mapping
        for(const modality in m2f) {
            if(m2f[modality].includes(folder_name)){
                m2f[modality] = m2f[modality].filter(folder => folder !== folder_name)
                if(m2f[modality].length === 0){
                    delete m2f[modality]
                }
                break
            }
        }
        dispatch({type: 'CLEAR_MOD2FOL_MAPPING', payload: m2f})

        dispatch({type: 'UPDATE_HAS_ALL_MAPPINGS', payload: false})

        // TODO clear modality from current
    }
}

/**
 * Sends Medical Folder dataset add request and validate result
 * @returns {(function(*))|*}
 */
export const addMedicalFolderDataset = (navigator) => {
    return (dispatch, getState) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Adding MedicalFolder dataset by validating all the inputs..."}})
        let medical_folder = getState().medicalFolderDataset
        let dlp = getState().dataLoadingPlan
        let data = {
            medical_folder_root : medical_folder.medical_folder_root,
            name : medical_folder.metadata.name,
            desc : medical_folder.metadata.desc,
            tags : medical_folder.metadata.tags
        }

        if(!medical_folder.ignore_reference_csv){
            data = {
                ...data,
                index_col: medical_folder.medical_folder_ref.ref.index,
                reference_csv_path: medical_folder.reference_csv ? medical_folder.reference_csv.path : null,
            }
        }

        data['dlp_id'] = dlp.selected_dlp_index !== null ?
                            dlp.existing_dlps['data'][dlp.selected_dlp_index][1] : null
        data['dlp_loading_blocks'] = dlp.dlp_loading_blocks
        data['dlp_name'] = dlp.dlp_name

        axios.post(EP_ADD_MEDICAL_FOLDER_DATASET, data).then( response => {
                dispatch({type: 'SUCCESS_MODAL' , payload: "Dataset has been successfully added"})
                dispatch({type:'SET_LOADING', payload: {status: false}})
                navigator('/datasets')
                dispatch({type:'RESET_MEDICAL_FOLDER'})
                dispatch({type:'RESET_DATA_LOADING_PLAN'})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while adding MedicalFolder dataset: "))
        })

    }
}

/**
 *
 * @param dataset_id
 * @returns {(function(*))|*}
 */
export const getMedicalFolderPreview = (dataset_id) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Loading MedicalFolder dataset subject table..."}})
        axios.post(EP_PREVIEW_MEDICAL_FOLDER_DATASET, {dataset_id: dataset_id}).then(response => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch({type: "SET_MEDICAL_FOLDER_PREVIEW", payload : {...response.data.result, dataset_id : dataset_id}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while getting preview of MedicalFolder dataset."))
        })
    }
}

/*
 * API call to check subdirectories in MedicalFolderDataset root folder to verify they can be parsed
 * @param path
 * @returns {(function(*): void)|*}
 */
const checkSubDirectories = (path) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Checkinh sub directories for MedicalFolder..."}})
        axios.post(EP_REPOSITORY_LIST, {path: path}).then(response => {
            if(response.status !== 200){
                dispatch({type: 'ERROR_MODAL' , payload: response.data.result.message})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while checking sub-directories of root MedicalFolderDataset folder."))
        })
    }
}

