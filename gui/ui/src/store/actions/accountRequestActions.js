import axios from "axios"
import {displayError, GET_USER_REQUESTS_ERROR, GET_USER_REQUESTS_LOADING, SET_LOADING} from "./actions";
import {EP_REQUESTS_LIST ,
        EP_REQUESTS_APPROVE,
        EP_REQUESTS_REJECT
    } from "../../constants";
import { authHeader } from "../../store/user.service"
import {GET_USER_REQUESTS} from "./actions";

/**
 * Request action for listing all account creation requests
 * @returns {dispatch}
 */
 export const listAccountRequests = () => {

    return (dispatch) => {
        dispatch({type: GET_USER_REQUESTS_LOADING, payload: {status: true, text: "Listing available account creation requests..."}})
        axios.get(EP_REQUESTS_LIST, {}, { headers: authHeader() })
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