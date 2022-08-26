import axios from 'axios'
import {EP_LIST_USERS, EP_REMOVE_USER} from "../../constants";
import {LIST_USERS, LIST_USERS_ERROR, LIST_USERS_LOADING} from "./actions";


/**
 * Action for listing users
 * @returns {(function(*): void)|*}
 */
export const listUsers = () => {
    return (dispatch) => {

        dispatch({type: LIST_USERS_LOADING, payload : true})
        axios.get(EP_LIST_USERS).then(response => {
            dispatch({type: LIST_USERS, payload: response.data.result})
            dispatch({type: LIST_USERS_LOADING, payload : false})
        }).catch( error => {
            dispatch({type: LIST_USERS_ERROR, payload: `An error occurred while listing platform 
            users ${error.response.data.message ? error.response.data.message : 'undefined error. Please' +
                    'contact system manager.'}`})
            dispatch({type: LIST_USERS_LOADING, payload : false})
        })
    }
}
