import axios from "axios";
import {EP_REPOSITORY_LIST,
    EP_LOAD_CSV_DATA,
    EP_VALIDATE_BIDS_ROOT,
    EP_VALIDATE_REFERENCE_COLUMN} from "../../constants";

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
        dispatch({type:'SET_LOADING', payload: true})
        axios.post(EP_VALIDATE_BIDS_ROOT, {bids_root : path.path})
            .then(response => {
                let data = response.data.result
                if(data.valid){
                    dispatch({type: "SET_BIDS_ROOT", payload: { root_path: path.path, modalities: data.modalities}})
                    dispatch({type: "SET_FOLDER_PATH", payload: path.path})
                    dispatch({type:'SET_LOADING', payload: false})
                    dispatch(getSubDirectories(path.path))
                }else{
                    dispatch({type:'RESET_BIDS_ROOT', payload: false})
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

        dispatch({type:'SET_LOADING', payload: true})
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
            dispatch({type:'SET_LOADING', payload: false})
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

        dispatch({type:'SET_LOADING', payload: true})
        axios.post(EP_LOAD_CSV_DATA, {path : path.path}).then( response => {
            if(response.status === 200){
                let data = response.data.result
                dispatch({type: "SET_REFERENCE_CSV", payload: { path: path.path, data: data}})
            }else{
                dispatch({type: 'ERROR_MODAL', payload: response.data.result.message})
            }
            dispatch({type:'SET_LOADING', payload: false})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: false})
            dispatch(displayError(error, "Error while validating reference CSV file"))
        })
        return
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
            dispatch({type:'SET_LOADING', payload: true})
            if(response.status === 200){
                let data = response.data.result
                dispatch({type: "PATIENT_FOLDERS", payload:data.path})
                console.log(data)
            }else{
                dispatch({type: 'ERROR_MODAL' , payload: response.data.result.message})
            }
            dispatch({type:'SET_LOADING', payload: false})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: false})
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
        dispatch({type:'SET_LOADING', payload: false})
        if(error.response.data.message){
            dispatch({type: 'ERROR_MODAL', payload: message + error.response.data.message})
        }else{
            dispatch({type: 'ERROR_MODAL', payload: 'Undefined error: ' + error.toString()})
        }
    }
}