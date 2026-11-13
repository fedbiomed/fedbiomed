import axios from 'axios'
import {EP_CHANGE_USER_ROLE, EP_LIST_USERS, EP_REMOVE_USER, EP_RESET_USER_PASSWORD} from "../../constants";
import {LIST_USERS, LIST_USERS_LOADING, USER_MANAGEMENT_ERROR, USER_MANAGEMENT_SUCCESS_MESSAGE} from "./actions";


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
            dispatch({type: USER_MANAGEMENT_SUCCESS_MESSAGE, payload : response.data.message})
        }).catch( error => {
            dispatch(
                display_user_management_error(error, 'An error occurred while listing platform users')
            )
            dispatch({type: LIST_USERS_LOADING, payload : false})
        })
    }
}

/**
 *
 * @param id
 * @returns {(function(*, *): void)|*}
 */
export const deleteUser = (id) => {
    return (dispatch, getState) => {

        let users = getState().users.user_list

        dispatch({type: LIST_USERS_LOADING, payload : true})
        axios.delete(EP_REMOVE_USER, {data : {user_id: id}}).then(response => {
             let deleted_user_id = response.data.result.user_id

             let index_user = users.findIndex( (element) => element.user_id  === deleted_user_id)
             users.splice(index_user, 1)
             dispatch({type: LIST_USERS, payload: users })
             dispatch({type: LIST_USERS_LOADING, payload : false})
             dispatch({type: USER_MANAGEMENT_SUCCESS_MESSAGE, payload : response.data.message})

        }).catch(error => {
            dispatch(
                display_user_management_error(error, `An error occurred while removing user with id ${id}`)
            )
            dispatch({type: LIST_USERS_LOADING, payload : false})
        })

    }
}

/**
 * Update password action that returns promise.
 * PROMISE SHOULD BE HANDLED WHERE THIS METHOD IS CALLED
 * @param user_id
 * @returns {*}
 */
export const resetPassword = (user_id) => {
    return axios.patch(EP_RESET_USER_PASSWORD, {user_id: user_id})
}


export const changeUserRole = (user_id, role) => {
    return axios.patch(EP_CHANGE_USER_ROLE, {user_id: user_id, role: role})
}

/**
 * Helper for displaying error messages on user management panel.
 * This error will be displayed above the table
 * @param error
 * @param message
 * @returns {(function(*): void)|*}
 */
const display_user_management_error = (error, message) => {
    return ( dispatch) => {
        dispatch({type: USER_MANAGEMENT_ERROR, payload: `${message} ${error.response.data.message ? 
                error.response.data.message : 'undefined error. Please' +
                    'contact system manager.'}`})
    }
}