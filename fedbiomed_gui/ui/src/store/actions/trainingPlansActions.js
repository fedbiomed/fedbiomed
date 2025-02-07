import {displayError, LIST_TRAINING_PLANS, RESET_SINGLE_TRAINING_PLAN, SINGLE_TRAINING_PLAN} from "./actions";
import {EP_APPROVE_TRAINING_PLAN, EP_DELETE_TRAINING_PLAN, EP_LIST_TRAINING_PLANS, EP_PREVIEW_TRAINING_PLAN, EP_REJECT_TRAINING_PLAN} from "../../constants";
import axios from "axios";
/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const list_training_plans = (data = {}) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Listing all training plans..."}})
        axios.post(EP_LIST_TRAINING_PLANS, data).then(response => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            let data = response.data.result
            dispatch({type: LIST_TRAINING_PLANS, payload: data})
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while getting training plans: "))
        })
    }
}

/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const get_single_training_plan = (data, navigator) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Listing all training plans..."}})
        axios.post(EP_PREVIEW_TRAINING_PLAN, data).then(response => {
            dispatch({type: SINGLE_TRAINING_PLAN , payload : response.data.result})
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            navigator('/training-plans')
            dispatch(displayError(error, "Error while displaying training plan: "))
        })
    }
}


export const reset_single_training_plan = () => {
    return (dispatch) => {
        dispatch({type: RESET_SINGLE_TRAINING_PLAN})
    }
}

/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const approve_training_plan = (data, navigator) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Rejecting training plan"}})
        axios.post(EP_APPROVE_TRAINING_PLAN, data).then(response => {
            navigator('/training-plans')
            dispatch({type: 'SUCCESS_MODAL' , payload: "Training plan has been approved"})
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
export const reject_training_plan = (data, navigator) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Rejecting training plan"}})
        axios.post(EP_REJECT_TRAINING_PLAN, data).then(response => {
            navigator('/training-plans')
            dispatch({type: 'SUCCESS_MODAL' , payload: "Training plan has been rejected"})
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
export const delete_training_plan = (data, navigator) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Rejecting training plan"}})
        axios.post(EP_DELETE_TRAINING_PLAN, data).then(response => {
            navigator('/training-plans')
            dispatch({type: 'SUCCESS_MODAL' , payload: "Dataset has been successfully deleted"})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while displaying model: "))
        })
    }
}