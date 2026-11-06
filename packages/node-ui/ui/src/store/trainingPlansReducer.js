import {LIST_TRAINING_PLANS, RESET_SINGLE_TRAINING_PLAN, SINGLE_TRAINING_PLAN} from "./actions/actions"

/**
 * Initial state of models
 * @type {{single_model: null, list: null}}
 */
const trainingPlansInitialState = {
    list : null,
    single_training_plan : null
}


/**
 * Reducer for models state
 * @param state
 * @param action
 * @returns {{single_model: null, list}|{single_model: null, list: null}|{single_model, list: null}}
 */
export const trainingPlansReducer = (state = trainingPlansInitialState, action) => {

    switch (action.type){
        case LIST_TRAINING_PLANS:
            return { ...state, list: action.payload}
        case SINGLE_TRAINING_PLAN:
            return {...state, single_training_plan: action.payload}
        case RESET_SINGLE_TRAINING_PLAN:
            return {...state, single_training_plan : null}
        default:
            return state
    }
}