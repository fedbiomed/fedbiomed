const initialStateRepository = {
    files : {},
    error : false,
    message : null
}


/**
 * Reducer for listing repository. 
 * @param {object} state 
 * @param {object} action 
 * @returns {object} 
 */
export const repositoryReducer = (state = initialStateRepository , action) => {


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
                message : action.payload
            }
            
        default: 
            
            return state
    }
}


const initialStateDataSets = {
    datasets : [],
    error : false,
    message : null
}


/**
 * Reducer for listing datasets 
 * @param {object} state 
 * @param {object} action 
 * @returns {object}
 */
export const datasetsreducer = (state = initialStateDataSets, action) => {

    switch (action.type) {

        case "GET_DATASETS":
            return {
                ...state,
                error : false,
                datasets : action.payload
            }
        
        case "UPDATE_DATASETS":

            return {
                ...state,
                error:false,
                datasets : action.payload
            } 
        
        case "DATASET_ERROR":
            return {
                ...state,
                error : true,
                datasets : [],
                message : action.payload       
            }
        default:
            return state

    }

}


const initialStateDataSetPreview = {
    data : null,
    error : false,
    message : null
}

export const datasetPreviewReducer = (state = initialStateDataSetPreview, action) => {
    switch (action.type) {

        case 'DATASET_PREVIEW':
            return {
                ...state,
                error: false,
                data : action.payload
            } 
        case 'DATASET_PREVIEW_ERROR':
            return {
                ...state,
                error: true,
                data : null,
                message : action.payload
            } 
        default:
            return state
    }   
}