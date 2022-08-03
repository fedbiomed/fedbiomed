const initialState = {
    'role': null,
    'is_auth': false,
    'email': null
}

export const authReducer = (state= initialState, action) => {
    switch (action.type){
        case 'LOGIN':
            return {
                ...state,
                is_auth : true
            }
        case 'LOGOUT':
            return {...state,
            is_auth : false}
        default:
            return state    
    }
} 