import axios from "axios";
import {
    EP_DLP_LIST,
} from "../../constants";


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

export const setDLPIndex = (event) => {
    let index = event.target.value
    return (dispatch) => {
        dispatch({type: 'CLEAR_PIPELINES', payload: {}})
        dispatch({type: 'SET_DLP', payload: index})
    }
}

export const setDLPDesc = (data) => {
    return (dispatch) => {
        dispatch({type: 'SET_DLP_NAME', payload: data})
    }
}

