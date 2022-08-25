import axios from "axios";
import {
    EP_DLP_LIST,
} from "../../constants";
import {displayError} from "./actions";


export const setChangeDlpMedicalFolderDataset = (use_dlp, state) => {
    return (dispatch) => {
        if(!use_dlp || state.dataLoadingPlan.selected_dlp_index === null) {
            dispatch({type: "RESET_MEDICAL_CHANGE_USED_DLP"})
        } else {
            //TODO: replace with values of the new used DLP
            let dlp = {
                use_custom_mod2fol: false,
                default_modality_names: [],
                current_modality_names: [],
                modalities_mapping: {}, // careful, not the initial null
                mod2fol_mapping: {}, // careful, not the initial nul
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
            dispatch({type: "SET_MEDICAL_CHANGE_USED_DLP", payload: dlp})
        }
    }
}

export const setUsePreExistingDlp = (data) => {
    return (dispatch, getState) => {
        let state = getState()
        let use_dlp = data.target.value === 'true' ? true : false

        dispatch({type: "SET_USE_PRE_EXISTING_DLP", payload: use_dlp})
        if(data.target.value === 'true') {
            dispatch({type:'SET_LOADING', payload: {status: true, text: "Fetching existing Data Loading Plans..."}})
            axios.get(EP_DLP_LIST).then(response => {
                dispatch({type: "SET_EXISTING_DLPS", payload: response.data.result})
                dispatch({type:'SET_LOADING', payload: {status: false}})
            }).catch(error => {
                dispatch({type:'SET_LOADING', payload: {status: false}})
                dispatch(displayError(error, "Error while fetching existing Data Loading Plans."))
            })
        } 
        dispatch(setChangeDlpMedicalFolderDataset(use_dlp, state))
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

