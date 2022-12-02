const initialState = {
    use_preexisting_dlp: false,
    preexisting_dlp : {
        dlp_id : null,
        mod2fol_mapping : null,
    },
    same_as_preexisting_dlp : true,
    selected_dlp_index: null,
    existing_dlps: null,
    dlp_name: "",
}


export const dataLoadingPlanReducer = (state = initialState, action) => {
    switch (action.type){
        case "SET_USE_PRE_EXISTING_DLP":
            return {
                ...state,
                use_preexisting_dlp : action.payload,
                dlp_name: initialState.dlp_name,
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
                    selected_dlp_index : initialState.selected_dlp_index,
                    dlp_name: initialState.dlp_name,
                }
            }
            return {
                ...state,
                selected_dlp_index : action.payload,
                dlp_name: initialState.dlp_name,
            }
        case "SET_DLP_NAME":
            return {
                ...state,
                dlp_name : action.payload
            }
        case "RESET_DLP":
            return initialState
        case "SET_DLP_SAME_AS_PREEXISTING":
            return {
                ...state,
                same_as_preexisting_dlp : action.payload
            }
        case "SET_DLP_PREEXISTING":
            return {
                ...state,
                preexisting_dlp: {
                    dlp_id : action.payload.dlp_id,
                    mod2fol_mapping : action.payload.mod2fol_mapping,
                },
                same_as_preexisting_dlp : initialState.same_as_preexisting_dlp,
            } 
        case "RESET_DLP_PREEXISTING":
            return {
                ...state,
                preexisting_dlp : initialState.preexisting_dlp,
                same_as_preexisting_dlp : initialState.same_as_preexisting_dlp,
            }
        default:
            return state
    }
}
