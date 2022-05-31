import axios from "axios";
import {
    EP_REPOSITORY_LIST,
    EP_LOAD_CSV_DATA,
    EP_VALIDATE_BIDS_ROOT,
    EP_VALIDATE_REFERENCE_COLUMN,
    EP_ADD_BIDS_DATASET,
    EP_PREVIEW_BIDS_DATASET
} from "../../constants";

/**
 * Sets Folder Path
 * @param path
 * @returns {(function(*): void)|*}
 */
export const setFolderPath = (path) => {
    return (dispatch) => {
        if(path.type !== "dir"){
            dispatch({type: 'ERROR_MODAL' , payload: "ROOT path for BIDS dataset should be folder/directory"})
            return
        }
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Setting/validating BIDS root folder path"}})
        axios.post(EP_VALIDATE_BIDS_ROOT, {bids_root : path.path})
            .then(response => {
                let data = response.data.result
                if(data.valid){
                    dispatch({type: "SET_BIDS_ROOT", payload: { root_path: path.path, modalities: data.modalities}})
                    dispatch({type: "SET_FOLDER_PATH", payload: path.path})
                    dispatch({type:'SET_LOADING', payload: {status: false}})
                    dispatch({type:'RESET_BIDS_REFERENCE_CSV'})
                    dispatch(getSubDirectories(path.path))
                }else{
                    dispatch({type:'RESET_BIDS_ROOT'})
                    dispatch({type: 'ERROR_MODAL', payload: data.message})
                }
            }).catch(error => {
                dispatch({type:'RESET_BIDS_ROOT', payload: false})
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
        let reference_csv = state.bidsDataset.reference_csv.path
        let root = state.bidsDataset.bids_root

        dispatch({type:'SET_LOADING', payload: {status: true, text: "Setting/validating BIDS subject reference column..."}})
        axios.post(EP_VALIDATE_REFERENCE_COLUMN, {
            reference_csv_path: reference_csv,
            bids_root: root,
            index_col: ref.index
        }).then(response => {
            let data = response.data.result
            if (data.valid) {
                let payload = {
                    ref: ref,
                    subjects: { available_subjects: data.subjects.intersection,
                                missing_entries: data.subjects.missing_entries,
                                missing_folders: data.subjects.missing_folders
                    }
                }
                dispatch({type: "SET_BIDS_REF", payload: payload})
            }else{
                dispatch({type: "RESET_BIDS_REF"})
                dispatch({type: "ERROR_MODAL", payload: data.message})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch(displayError(error, "Can not verify reference column for BIDS dataset. Error message is: "))
        })
    }
}

/**
 * Sets reference csv file for BIDS
 * @param path
 * @returns {(function(*): void)|*}
 */
export const setReferenceCSV = (path) => {
    return (dispatch) => {

        dispatch({type:'SET_LOADING', payload: {status: true, text: "Setting reference/demographics dataset for BIDS..."}})
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
 * Sets BIDS dataset meta information dataset name, tags etc.
 * @param data
 * @returns {(function(*))|*}
 */
export const setBIDSDatasetMetadata = (data) => {
    return (dispatch) => {
        dispatch({type: "SET_BIDS_METADATA", payload: data})
    }
}


export const setIgnoreReferenceCsv = (data) => {
    return (dispatch) => {
        dispatch({type: "SET_IGNORE_REFERENCE_CSV", payload: data})
    }
}

/**
 * Sends BIDS dataset add request and validate result
 * @returns {(function(*))|*}
 */
export const addBIDSDataset = (navigator) => {
    return (dispatch, getState) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Adding BIDS dataset by validating all the inputs..."}})
        let bids = getState().bidsDataset
        let data = {
            bids_root : bids.bids_root,
            name : bids.metadata.name,
            desc : bids.metadata.desc,
            tags : bids.metadata.tags
        }

        if(!bids.ignore_reference_csv){
            data = {
                ...data,
                index_col: bids.bids_ref.ref.index,
                reference_csv_path: bids.reference_csv ? bids.reference_csv.path : null,
            }
        }
        axios.post(EP_ADD_BIDS_DATASET, data).then( response => {
                dispatch({type: 'SUCCESS_MODAL' , payload: "Dataset has been successfully added"})
                dispatch({type:'SET_LOADING', payload: {status: false}})
                navigator('/datasets')
                dispatch({type:'RESET_BIDS'})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while adding BIDS dataset: "))
        })
    }
}

/**
 *
 * @param dataset_id
 * @returns {(function(*))|*}
 */
export const getBIDSPreview = (dataset_id) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Loading BIDS dataset subject table..."}})
        axios.post(EP_PREVIEW_BIDS_DATASET, {dataset_id: dataset_id}).then(response => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch({type: "SET_BIDS_PREVIEW", payload : {...response.data.result, dataset_id : dataset_id}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while getting preview of BIDS"))
        })
    }
}

/**
 * API call to get subdirectories in BIDS root folder
 * @param path
 * @returns {(function(*): void)|*}
 */
const getSubDirectories = (path) => {
    return (dispatch) => {
        axios.post(EP_REPOSITORY_LIST, {path: path}).then(response => {
            dispatch({type:'SET_LOADING', payload: {status: true, text: "Getting sub directories for BIDS..."}})
            if(response.status === 200){
                let data = response.data.result
                dispatch({type: "PATIENT_FOLDERS", payload:data.path})
            }else{
                dispatch({type: 'ERROR_MODAL' , payload: response.data.result.message})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while getting sub-directories of root BIDS folder."))
        })
    }
}


/**
 * Dispatch action the display global error modal window
 * @param error
 * @param message
 * @returns {(function(*): void)|*}
 */
const displayError = (error, message = "") => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: false}})
        if(error.response){
            if(error.response.data.message){
                dispatch({type: 'ERROR_MODAL', payload: message + error.response.data.message})
            }else{
                dispatch({type: 'ERROR_MODAL', payload: 'Undefined error: ' + error.toString()})
            }
        }
    }
}