const initialState = {
    use_preexisting_dlp: false,
    selected_dlp_index: null,
    existing_dlps: null,
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
            if (parseInt(action.payload) === -1){
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
        // TODO: remove after completing DLP save
        //case "ADD_PIPELINE":
        //    let data_loading_blocks = state.dlp_loading_blocks
        //    data_loading_blocks[action.payload.type_id] = {
        //        serial_id: action.payload.serial_id,
        //        module: action.payload.module,
        //        qualname: action.payload.qualname,
        //    }
        //    return {
        //        ...state,
        //        dlp_loading_blocks: data_loading_blocks
        //    }
        //case "CLEAR_PIPELINES":
        //    return {
        //        ...state,
        //        dlp_loading_blocks: {}
        //    }
        case "RESET_DLP":
            return initialState
        default:
            return state
    }
}
