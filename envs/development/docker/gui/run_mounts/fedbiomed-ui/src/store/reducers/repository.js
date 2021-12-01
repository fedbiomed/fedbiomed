const initialState = {
    files : [],
    error : false,
    message : null
}


export const repositoryReducer = (state = initialState , action) => {


    switch (action.type) {
        case "LIST_REPOSITORY":

            return {
                ...state,
                files : action.payload.files,
                error : false,
                message : null
            }
        case "ERROR":

            return {
                files : [],
                error : true,
                message : action.payload.message
            }
            
        default: 
            
            return state
    }
}