import axios from 'axios'
import {EP_LIST_USERS, EP_REMOVE_USER} from "../../constants";
import {LIST_USERS} from "./actions";


/**
 * Action for listing users
 * @returns {(function(*): void)|*}
 */
export const listUsers = () => {
    return (dispatch) => {
        axios.get(EP_LIST_USERS).then(response => {
            dispatch({type: LIST_USERS, payload: response.data.result})
        }).catch( error => {
            dispatch({type: LIST_USERS, payload: `An error occurred while listing platform 
            users ${error.response.data.message ? error.response.data.message : 'undefined error. Please' +
                    'contact system manager.'}`})
        })
    }
}
