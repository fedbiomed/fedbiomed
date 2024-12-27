import {
    LIST_USERS,
    LIST_USERS_LOADING,
    USER_MANAGEMENT_ERROR,
    USER_MANAGEMENT_SUCCESS_MESSAGE
} from "./actions/actions";

const initialUsers =  {
    user_list: [],
    error : null,
    loading: false,
    success: null
}

export const usersReducer = (state = initialUsers , action) => {
    switch (action.type){
        case LIST_USERS:
            return {
                ...state,
                user_list : action.payload,
                error : null
            }
        case USER_MANAGEMENT_ERROR:
            return {
                ...state,
                error : action.payload,
                success: null
            }
        case USER_MANAGEMENT_SUCCESS_MESSAGE: {
            return {
                ...state,
                success: action.payload,
                error: null
            }
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