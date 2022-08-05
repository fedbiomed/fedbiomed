import axios from "axios";
import {
    EP_REPOSITORY_LIST,
    EP_LOAD_CSV_DATA,
    EP_VALIDATE_MEDICAL_FOLDER_ROOT,
    EP_VALIDATE_REFERENCE_COLUMN,
    EP_ADD_MEDICAL_FOLDER_DATASET,
    EP_PREVIEW_MEDICAL_FOLDER_DATASET,
    EP_DLP_LIST,
    EP_DEFAULT_MODALITY_NAMES,
    EP_DP_MOD2FOL_CREATE,
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
                    dispatch({type: "SET_MEDICAL_FOLDER_ROOT", payload: { root_path: path.path, modalities: data.modalities}})
                    dispatch({type: "SET_FOLDER_PATH", payload: path.path})
                    dispatch({type:'SET_LOADING', payload: {status: false}})
                    dispatch({type:'RESET_MEDICAL_FOLDER_REFERENCE_CSV'})
                    dispatch(getSubDirectories(path.path))
                }else{
                    dispatch({type:'RESET_MEDICAL_FOLDER_ROOT'})
                    dispatch({type: 'ERROR_MODAL', payload: data.message})
                }
            }).catch(error => {
                dispatch({type:'RESET_MEDICAL_FOLDER_ROOT', payload: false})
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

        validation_data['dlp_id'] = state.medicalFolderDataset.selected_dlp_index !== null ?
            state.medicalFolderDataset.existing_dlps['data'][state.medicalFolderDataset.selected_dlp_index][1] : null
        validation_data['dlp_pipelines'] = state.medicalFolderDataset.dlp_pipelines

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

export const setUsePreExistingDlp = (data) => {
    return (dispatch) => {
        dispatch({type: "SET_USE_PRE_EXISTING_DLP", payload: data})
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Fetching existing Data Loading Plans..."}})
        axios.get(EP_DLP_LIST).then(response => {
            dispatch({type: "SET_EXISTING_DLPS", payload: response.data.result})
            dispatch({type:'SET_LOADING', payload: {status: false}})
        })
    }
}

export const setDLP = (index) => {
    return (dispatch) => {
        dispatch({type: 'CLEAR_PIPELINES', payload: {}})
        dispatch({type: 'SET_DLP', payload: index})
    }
}

export const setCreateModalitiesToFoldersPipeline = (value) => {
    return (dispatch) => {
        dispatch({type: 'SET_CREATE_DLP', payload: value})
    }
}

export const CreateModalitiesToFoldersPipeline = (modalities_mapping) => {
    return (dispatch) => {
        dispatch({type: 'SET_DLP', payload: -1})
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Saving Association"}})
        axios.post(EP_DP_MOD2FOL_CREATE, {mapping: modalities_mapping}).then(response => {
            dispatch({type: 'ADD_PIPELINE',
                      payload: {type_id: response.data.result.type_id,
                                serial_id: response.data.result.serial_id}})
            dispatch({type:'SET_LOADING', payload: {status: false}})
        })
    }
}

export const getDefaultModalityNames = () => {
    return (dispatch) => {
        axios.get(EP_DEFAULT_MODALITY_NAMES).then(response => {
            dispatch({type: 'SET_DEFAULT_MODALITY_NAMES', payload: response.data.result.default_modalities})
        })
    }
}

export const updateModalitiesMapping = (data) => {
    return (dispatch) => {
        dispatch({type: 'UPDATE_MODALITIES_MAPPING', payload: {folder_name: data.folder_name, modality_name: data.modality_name} })
    }
}

export const clearModalityMapping = (folder_name) => {
    return (dispatch) => {
        dispatch({type: 'CLEAR_MODALITY_MAPPING', payload: folder_name})
    }
}

export const setDLPName = (data) => {
    return (dispatch) => {
        dispatch({type: 'SET_DLP_NAME', payload: data})
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

        data['dlp_id'] = medical_folder.selected_dlp_index !== null ?
                            medical_folder.existing_dlps['data'][medical_folder.selected_dlp_index][1] : null
        data['dlp_pipelines'] = medical_folder.dlp_pipelines
        data['dlp_name'] = medical_folder.dlp_name

        axios.post(EP_ADD_MEDICAL_FOLDER_DATASET, data).then( response => {
                dispatch({type: 'SUCCESS_MODAL' , payload: "Dataset has been successfully added"})
                dispatch({type:'SET_LOADING', payload: {status: false}})
                navigator('/datasets')
                dispatch({type:'RESET_MEDICAL_FOLDER'})
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
 * API call to get subdirectories in MedicalFolderDataset root folder
 * @param path
 * @returns {(function(*): void)|*}
 */
const getSubDirectories = (path) => {
    return (dispatch) => {
        axios.post(EP_REPOSITORY_LIST, {path: path}).then(response => {
            dispatch({type:'SET_LOADING', payload: {status: true, text: "Getting sub directories for MedicalFolder..."}})
            if(response.status === 200){
                let data = response.data.result
                dispatch({type: "PATIENT_FOLDERS", payload:data.path})
            }else{
                dispatch({type: 'ERROR_MODAL' , payload: response.data.result.message})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while getting sub-directories of root MedicalFolderDataset folder."))
        })
    }
}

