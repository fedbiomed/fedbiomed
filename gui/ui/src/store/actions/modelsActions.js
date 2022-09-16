import {displayError, LIST_MODELS, RESET_SINGLE_MODEL, SINGLE_MODEL} from "./actions";
import {EP_APPROVE_MODEL, EP_DELETE_MODEL, EP_LIST_MODELS, EP_PREVIEW_MODEL, EP_REJECT_MODEL} from "../../constants";
import axios from "axios";
/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const list_models = (data = {}) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Listing all models..."}})
        axios.post(EP_LIST_MODELS, data).then(response => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            let data = response.data.result
            dispatch({type: LIST_MODELS, payload: data})
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while getting all the models: "))
        })
    }
}

/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const get_single_model = (data, navigator) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Listing all models..."}})
        axios.post(EP_PREVIEW_MODEL, data).then(response => {
            dispatch({type: SINGLE_MODEL , payload : response.data.result})
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            navigator('/models')
            dispatch(displayError(error, "Error while displaying model: "))
        })
    }
}


export const reset_single_model = () => {
    return (dispatch) => {
        dispatch({type: RESET_SINGLE_MODEL})
    }
}

/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const approve_model = (data, navigator) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Rejecting Model"}})
        axios.post(EP_APPROVE_MODEL, data).then(response => {
            navigator('/models')
            dispatch({type: 'SUCCESS_MODAL' , payload: "Model has been approved"})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while displaying model: "))
        })
    }
}


/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const reject_model = (data, navigator) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Rejecting Model"}})
        axios.post(EP_REJECT_MODEL, data).then(response => {
            navigator('/models')
            dispatch({type: 'SUCCESS_MODAL' , payload: "Model has been rejected"})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while displaying model: "))
        })
    }
}

/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const delete_model = (data, navigator) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Rejecting Model"}})
        axios.post(EP_DELETE_MODEL, data).then(response => {
            navigator('/models')
            dispatch({type: 'SUCCESS_MODAL' , payload: "Dataset has been successfully deleted"})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while displaying model: "))
        })
    }
}