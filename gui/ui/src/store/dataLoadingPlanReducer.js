const initialState = {
    use_preexisting_dlp: false,
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
                    selected_dlp_index : null,
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
        default:
            return state
    }
}
