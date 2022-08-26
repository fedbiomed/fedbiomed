import {LIST_USERS, LIST_USERS_ERROR, LIST_USERS_LOADING } from "./actions/actions";

const initialUsers =  {
    user_list: [],
    error : null,
    loading: false
}

export const usersReducer = (state = initialUsers , action) => {
    switch (action.type){
        case LIST_USERS:
            return {
                ...state,
                user_list : action.payload,
                error : null
            }
        case LIST_USERS_ERROR:
            return {
                ...state,
                user_list : [],
                error : action.payload
            }

        case LIST_USERS_LOADING:
            return {
                ...state,
                loading : action.payload
            }
        default:
            return state
    }
}