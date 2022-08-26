import {LIST_USERS, LIST_USERS_ERROR } from "./actions/actions";

const initialUsers =  {
    user_list: [],
    error : null
}

export const usersReducer = (state = initialUsers , action) => {
    switch (action.type){
        case LIST_USERS:
            return {
                user_list : action.payload,
                error : null
            }
        case LIST_USERS_ERROR:
            return {
                user_list : [],
                error : action.payload
            }
        default:
            return state
    }
}