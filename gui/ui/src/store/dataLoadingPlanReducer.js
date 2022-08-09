const initialState = {
    use_preexisting_dlp: false,
    selected_dlp_index: null,
    existing_dlps: null,
    dlp_pipelines: {},
    dlp_name: ""
}


export const dataLoadingPlanReducer = (state = initialState, action) => {
    switch (action.type){
        case "SET_USE_PRE_EXISTING_DLP":
            return {
                ...state,
                use_preexisting_dlp : action.payload
            }
        case "SET_EXISTING_DLPS":
            return {
                ...state,
                existing_dlps : action.payload
            }
        case "SET_DLP":
            if (action.payload === -1){
                return  {
                    ...state,
                    selected_dlp_index : null
                }
            }
            return {
                ...state,
                selected_dlp_index : action.payload
            }
        case "SET_DLP_NAME":
            return {
                ...state,
                dlp_name : action.payload
            }
        case "ADD_PIPELINE":
            let pipelines = state.dlp_pipelines
            pipelines[action.payload.type_id] = action.payload.serial_id
            return {
                ...state,
                dlp_pipelines: pipelines
            }
        case "CLEAR_PIPELINES":
            return {
                ...state,
                dlp_pipelines: {}
            }
        case "RESET_DATA_LOADING_PLAN":
            return initialState
        default:
            return state
    }
}
