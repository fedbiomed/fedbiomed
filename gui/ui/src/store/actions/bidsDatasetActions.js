import axios from "axios";
import {EP_REPOSITORY_LIST} from "../../constants";

/**
 * Sets Folder Path
 * @param path
 * @returns {(function(*): void)|*}
 */
export const setFolderPath = (path) => {
    return (dispatch) => {
        dispatch({type: "SET_FOLDER_PATH", payload: path})
        dispatch(getSubDirectories(path.path))
    }
}

/**
 * Sets reference csv file for BIDS
 * @param path
 * @returns {(function(*): void)|*}
 */
export const setReferenceCSV = (path) => {
    return (dispatch) => {
        dispatch({type: "SET_REFERENCE_CSV", payload: path})
    }
}

/**
 * API call to get sub directories in BIDS root folder
 * @param path
 * @returns {(function(*): void)|*}
 */
const getSubDirectories = (path) => {
    return (dispatch) => {
        axios.post(EP_REPOSITORY_LIST, {path: path}).then(response => {
            dispatch({type:'SET_LOADING', payload: true})
            if(response.status === 200){
                let data = response.data.result
                dispatch({type: "PATIENT_FOLDERS", payload:data})
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
const displayError = (error, message) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: false})
        if(error.response){
            dispatch({type: 'ERROR_MODAL', payload: message + error.response.data.message})
        }else{
            dispatch({type: 'ERROR_MODAL', payload: 'Unexpected Error:' + error.toString()})
        }
    }
}