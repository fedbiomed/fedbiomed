import axios from "axios";
import {
    EP_DLP_LIST,
} from "../../constants";
import {displayError} from "./actions";


export const setUsePreExistingDlp = (data) => {
    return (dispatch) => {
        dispatch({type: "SET_USE_PRE_EXISTING_DLP", payload: (data.target.value === 'true' ? true : false)})
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
    }
}

export const setDLPIndex = (event) => {
    return (dispatch) => {
        dispatch({type: 'CLEAR_PIPELINES', payload: {}})
        dispatch({type: 'SET_DLP', payload: event.target.value})
    }
}

export const setDLPDesc = (data) => {
    return (dispatch) => {
        dispatch({type: 'SET_DLP_NAME', payload: data})
    }
}

