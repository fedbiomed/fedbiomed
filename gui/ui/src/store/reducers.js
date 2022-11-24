// --------------------------------------------------------------------------------------------------------------------
const resultState = {
    error : false,
    success : false,
    message : null,
    show : false,
    loading:  false,
    launcher: null,
    text: ""

}
export const resultReducer = ( state = resultState, action) => {

    switch (action.type){
        case "SET_LOADING":
            if( state.launcher && state.loading !== action.payload.status &&
                state.launcher !== action.payload.launcher ){
                return state
            }else{
                return {
                    ...state,
                    loading: action.payload.status,
                    text : action.payload.text ? action.payload.text : "",
                    launcher: action.payload.launcher && action.payload.status === false ? null : action.payload.launcher
                }
            }


        case "ERROR_MODAL":
            return {
                error:true,
                show: true,
                success: false,
                message : action.payload,
                loading:false
            }
        case "SUCCESS_MODAL":
            return{
                error:false,
                show: true,
                success:true,
                message:action.payload,
                loading:false
            }
        case "RESET_GLOBAL_MODAL":
            return resultState
        default:
            return state
    }
}


// --------------------------------------------------------------------------------------------------------------------
const initialStateRepository = {
    files : {},
    folders: {},
    current: [],
    error : false,
    level: 0,
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
                folders:action.payload.folders,
                level: action.payload.level,
                current: action.payload.current,
                error : false,
                message : null
            }
        default: 
            
            return state
    }
}


// --------------------------------------------------------------------------------------------------------------------
const initialStateDataSets = {
    datasets : [],
    new_dataset : [],
    add_dataset : {
        success : null,
        message : null,
        result : null,
        waiting: false,
    },
    default_dataset: {
        success : null,
        result : null,
        waiting : false
    },
    remove_dataset : {
        success:false,
        waiting:false,
    },
    search : [],
    error : false,
    message : null
}

/**
 * Reducer for listing datasets 
 * @param {object} state 
 * @param {object} action 
 * @returns {object}
 */
export const datasetsReducer = (state = initialStateDataSets, action) => {

    switch (action.type) {

        case "GET_DATASETS":
            return {
                ...state,
                error : false,
                datasets : action.payload
            }
        case "SEARCH_DATASET_RESULT":
            return {
                ...state,
                search : action.payload
            }

        case "NEW_DATASET_TO_ADD":
            return {
                ...state,
                new_dataset : {
                    ...state.new_dataset,
                    path : action.payload.path,
                    extension: action.payload.extension
                },
                error : false,

            }

        case "RESET_NEW_DATASET":
            return {
                ...state,
                new_dataset : {
                    path : null,
                    extension: null,
                },
                error : false,

            }
        case "RESET_ADD_DATASET_RESULT": {
           return{
                ...state,
                add_dataset : {
                    error : null,
                    success : null,
                    message : null,
                    waiting: false,
                    result : null
                }
            }
        }

        case "ADD_DATASET_RESULT":
            return {
                ...state,
                datasets: [...state.datasets, action.payload],
                add_dataset : {
                         success:true,
                         waiting:false
                }
            }
        case "DEFAULT_DATASET_ADD_REQUEST":
            return {
                ...state,
                default_dataset: {
                    ...state.default_dataset,
                    waiting: true,
                }
            }

        case "DEFAULT_DATASET_ADD_SUCCESS":
            return {
                ...state,
                default_dataset: {
                    ...state.default_dataset,
                    success : true,
                    waiting: false,
                    result: action.payload
                }
            }
        case "RESET_DEFAULT_DATASET_RESULT":
            return {
                ...state,
                default_dataset: initialStateDataSets.default_dataset
            }
        case "REMOVE_DATASET_ERROR":
            return {
                ...state,
                remove_dataset : {
                    error:true,
                    success:false,
                    result: action.payload
                }
            }

        case "REMOVE_DATASET_SUCCESS":
            return {
                ...state,
                remove_dataset : {
                    success:true,
                    error:false,
                    result: action.payload
                }
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

// ---------------------------------------------------------------------------------------------------------------------
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


//----------------------------------------------------------
// changing password

// const initialPasswordState = {
//     new_pwd:"",
//     error: false,
//     message: ""
// }


// export const setNewPasswordWindow = (state = initialPasswordState, action) => {
//     switch(action.type){
//         case 'SETTING_PWD':
//             return {
//                 ...state,

//             }

//     }
// }