import axios from "axios";
import {
    EP_LIST_DATA_LOADING_PLANS,
} from "../../constants";
import {displayError} from "./actions";
import {setChangeDlpMedicalFolderDataset} from "./medicalFolderDatasetActions"


export const setUsePreExistingDlp = (data) => {
    return (dispatch, getState) => {
        let state = getState()
        let use_dlp = data.target.value === 'true' ? true : false

        dispatch({type: "SET_USE_PRE_EXISTING_DLP", payload: use_dlp})
        if(data.target.value === 'true' && state.dataLoadingPlan.existing_dlps === null) {
            dispatch({type:'SET_LOADING', payload: {status: true, text: "Fetching existing Data Loading Plans..."}})
            axios.post(EP_LIST_DATA_LOADING_PLANS, {'target_dataset_type': 'medical-folder'}).then(response => {
                dispatch({type: "SET_EXISTING_DLPS", payload: response.data.result})
                dispatch({type:'SET_LOADING', payload: {status: false}})
                dispatch(setChangeDlpMedicalFolderDataset(use_dlp, state))
            }).catch(error => {
                dispatch({type:'SET_LOADING', payload: {status: false}})
                dispatch(displayError(error, "Error while fetching existing Data Loading Plans."))
            })
        } else {
            dispatch(setChangeDlpMedicalFolderDataset(use_dlp, state))
        }
    }
}

export const setDLPIndex = (event) => {
    return (dispatch, getState) => {
        dispatch({type: 'SET_DLP', payload: event.target.value})
        let state = getState() // needs to be done after setting DLP
        dispatch(setChangeDlpMedicalFolderDataset((event.target.value === '-1' ? false : true), state))
    }
}

export const setDLPDesc = (data) => {
    return (dispatch) => {
        dispatch({type: 'SET_DLP_NAME', payload: data})
    }
}

