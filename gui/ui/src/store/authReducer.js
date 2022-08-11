const initialState = {
    'is_auth': false,
    'role': null,
    'email': null,
    'user_name': '',
    'user_surname': '',
}

/**
 * Reducer for authentication state
 * @param state
 * @param action
 * @returns {(*&{is_auth: boolean})|{role: null, user_name: string, is_auth: boolean, user_surname: string, email: null}}
 * */
export const authReducer = (state= initialState, action) => {
    switch (action.type){
        case 'LOGIN':
            return {...action.payload, is_auth : true}
        case 'LOGOUT':
            return {...initialState}
        default:
            return state    
    }
} 