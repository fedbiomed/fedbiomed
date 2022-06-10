import {LIST_MODELS, RESET_SINGLE_MODEL, SINGLE_MODEL} from "./actions/actions"

/**
 * Initial state of models
 * @type {{single_model: null, list: null}}
 */
const modelsInitialState = {
    list : null,
    single_model : null
}


/**
 * Reducer for models state
 * @param state
 * @param action
 * @returns {{single_model: null, list}|{single_model: null, list: null}|{single_model, list: null}}
 */
export const modelsReducer = (state = modelsInitialState, action) => {

    switch (action.type){
        case LIST_MODELS:
            return { ...state, list: action.payload}
        case SINGLE_MODEL:
            return {...state, single_model: action.payload}
        case RESET_SINGLE_MODEL:
            return {...state, single_model : null}
        default:
            return state
    }
}