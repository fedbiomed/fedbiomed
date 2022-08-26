import axios from "axios"
import {displayError} from "./actions";
import {EP_REQUESTS_LIST ,
        EP_REQUESTS_APPROVE,
        EP_REQUESTS_REJECT
    } from "../../constants";
import { authHeader } from "../../store/user.service"


/**
 * Request action for listing all account creation requests
 * @returns {dispatch}
 */
 export const listAccountRequests = (data) => {

    return (dispatch, getState) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Listing available account creation requests..."}})
        axios.get(EP_REQUESTS_LIST, { headers: authHeader() })
             .then( response => {
                if (response.status === 200){
                    dispatch({type:'SET_LOADING', payload: {status: false, text: ""}})
                    dispatch({ type : "GET_REQUESTS", payload: response.data.result})

                }else{
                    dispatch({type:'SET_LOADING', payload: {status: false, text: ""}})
                    alert(response.data.message)
                }
             })
             .catch( error => {
                 dispatch({type:'SET_LOADING', payload: {status: false, text: ""}})
                 dispatch(displayError(error, "Error while getting all the requests: "))
             })
    }
}