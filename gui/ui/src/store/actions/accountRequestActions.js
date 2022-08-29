import axios from "axios"
import {APPROVE_USER_REQUEST, APPROVE_USER_REQUEST_ERROR, APPROVE_USER_REQUEST_LOADING, 
    GET_USER_REQUESTS_ERROR, GET_USER_REQUESTS_LOADING, REJECT_USER_REQUEST, REJECT_USER_REQUEST_ERROR, REJECT_USER_REQUEST_LOADING} from "./actions";
import {EP_REQUESTS_LIST ,
        EP_REQUEST_APPROVE,
        EP_REQUEST_REJECT
    } from "../../constants";
import {GET_USER_REQUESTS} from "./actions";
import { store } from "../../index"

/**
 * Request action for listing all account creation requests
 * @returns {dispatch}
 */
 export const listAccountRequests = () => {

    return (dispatch) => {
        dispatch({type: GET_USER_REQUESTS_LOADING, payload: {status: true, text: "Listing available account creation requests..."}})
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
    return (dispatch) => {
        dispatch({type:APPROVE_USER_REQUEST_LOADING, payload: {status: true, text: "Approving user request..."}})
        // Get current user requests state
        let user_requests = store.getState().user_requests
        axios.post(EP_REQUEST_APPROVE, {request_id : data.request_id})
             .then(res => {
                if(res.status === 201){
                    let index = user_requests.requests.map(function(e) {
                        return e.request_id;
                    }).indexOf(data.request_id);
                    if (index > -1) {
                        user_requests.requests.splice(index, 1);
                        dispatch({ type: APPROVE_USER_REQUEST, payload: user_requests.requests})
                        dispatch({type: APPROVE_USER_REQUEST_LOADING, payload: {status: false}})
                    }
                }else{
                    dispatch({type:APPROVE_USER_REQUEST_LOADING, payload: {status: false}})
                    dispatch({type: APPROVE_USER_REQUEST_ERROR, payload: res.data.message})
                }
             })
             .catch(error => {
                 dispatch({type: APPROVE_USER_REQUEST_LOADING, payload: {status: false}})
                if(error.response){
                    dispatch({type: APPROVE_USER_REQUEST_ERROR, payload: 'Error while approving user request: ' + error.response.data.message})
                }else{
                    dispatch({type: APPROVE_USER_REQUEST_ERROR, payload: 'Unexpected error:' + error.toString()})
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
    return (dispatch) => {
        dispatch({type:REJECT_USER_REQUEST_LOADING, payload: {status: true, text: "Rejecting user request..."}})
        // Get current user requests state
        let user_requests = store.getState().user_requests
        axios.post(EP_REQUEST_REJECT, {request_id : data.request_id})
             .then(res => {
                if(res.status === 200){
                    let index = user_requests.requests.map(function(e) {
                        return e.request_id;
                    }).indexOf(data.request_id);
                    if (index > -1) {
                        user_requests.requests[index] = res.data.result
                        console.log(user_requests.requests)
                        dispatch({ type: REJECT_USER_REQUEST, payload: user_requests.requests})
                        dispatch({type: REJECT_USER_REQUEST_LOADING, payload: {status: false}})
                    }
                }else{
                    dispatch({type:REJECT_USER_REQUEST_LOADING, payload: {status: false}})
                    dispatch({type: REJECT_USER_REQUEST_ERROR, payload: res.data.message})
                }
             })
             .catch(error => {
                 dispatch({type: REJECT_USER_REQUEST_LOADING, payload: {status: false}})
                if(error.response){
                    dispatch({type: REJECT_USER_REQUEST_ERROR, payload: 'Error while rejecting user request: ' + error.response.data.message})
                }else{
                    dispatch({type: REJECT_USER_REQUEST_ERROR, payload: 'Unexpected error:' + error.toString()})
                }
             })
    }
}