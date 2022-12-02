import axios from "axios"
import {
    APPROVE_USER_REQUEST, USER_REQUESTS_ERROR_MESSAGE, GET_USER_REQUESTS, GET_USER_REQUESTS_ERROR,
    GET_USER_REQUESTS_LOADING, REJECT_USER_REQUEST, USER_REQUESTS_SUCCESS_MESSAGE
} from "./actions";
import {EP_REQUESTS_LIST ,
        EP_REQUEST_APPROVE,
        EP_REQUEST_REJECT,
        REJECTED_REQUEST
    } from "../../constants";

/**
 * Request action for listing all account creation requests
 * @returns {dispatch}
 */
 export const listAccountRequests = () => {

    return (dispatch) => {
        dispatch({type: GET_USER_REQUESTS_LOADING, payload: true})
        axios.get(EP_REQUESTS_LIST, {})
             .then( response => {
                    dispatch({type: GET_USER_REQUESTS, payload: response.data.result})
                    dispatch({ type: GET_USER_REQUESTS_LOADING, payload: false})
                })
            .catch( error => {
                dispatch({type: GET_USER_REQUESTS_ERROR, payload: `An error occurred while listing platform 
                users ${error.response.data.message ? error.response.data.message : 'undefined error. Please' +
                        'contact system manager.'}`})
                dispatch({type: GET_USER_REQUESTS_LOADING, payload : false})
            })
    }
}


/**
 * Request action for approving an account creation request
 * @param data Object that has request_id
 * @returns {(function(*, *): void)|*}
 */
export const approveAccountRequest = (data) => {
    return (dispatch, getState) => {

        let user_requests = getState().user_requests.requests

        dispatch({type: GET_USER_REQUESTS_LOADING, payload: true})
        axios.post(EP_REQUEST_APPROVE, {request_id : data.request_id})
             .then(res => {
                    let index = user_requests.findIndex( (element) => element.request_id  === data.request_id)
                    user_requests.splice(index, 1);
                    dispatch({type: APPROVE_USER_REQUEST, payload: user_requests})
                    dispatch({
                        type: USER_REQUESTS_SUCCESS_MESSAGE,
                        payload: `The register request of user "${res.data.result.user_email}" has \
                        been  approved successfully` })
             })
             .catch(error => {
                 dispatch({type: GET_USER_REQUESTS_LOADING, payload: false})
                if(error.response){
                    dispatch({type: USER_REQUESTS_ERROR_MESSAGE, payload: 'Error while approving user request: ' + error.response.data.message})
                }else{
                    dispatch({type: USER_REQUESTS_ERROR_MESSAGE, payload: 'Unexpected error:' + error.toString()})
                }
             })
    }
}


/**
 * Request action for rejecting an account creation request
 * @param data Object that has request_id
 * @returns {(function(*, *): void)|*}
 */
export const rejectAccountRequest = (data) => {
    return (dispatch, getState) => {
        let user_requests = getState().user_requests.requests

        dispatch({type: GET_USER_REQUESTS_LOADING, payload: true})
        axios.post(EP_REQUEST_REJECT, {request_id : data.request_id})
             .then(res => {
                    let index = user_requests.findIndex( (element) => element.request_id  === data.request_id)
                    user_requests[index].request_status = REJECTED_REQUEST
                    dispatch({ type: REJECT_USER_REQUEST, payload: user_requests})
                    dispatch({type: GET_USER_REQUESTS_LOADING, payload: false})
                    dispatch({
                        type: USER_REQUESTS_SUCCESS_MESSAGE,
                        payload: `The register request of user "${res.data.result.user_email}" has been rejected.` })
             })
             .catch(error => {
                 dispatch({type: GET_USER_REQUESTS_LOADING, payload: false})
                if(error.response){
                    dispatch({type: USER_REQUESTS_ERROR_MESSAGE, payload: 'Error while rejecting user request: ' + error.response.data.message})
                }else{
                    dispatch({type: USER_REQUESTS_ERROR_MESSAGE, payload: 'Unexpected error:' + error.toString()})
                }
             })
    }
}