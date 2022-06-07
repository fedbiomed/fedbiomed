import {LIST_MODELS} from "./actions";

/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const list_models = (data = {}) => {
    return (dispatch) => {

        let data = [
            {name: "HMMMM", id: "BIMMM", status: "ss"},
            {name: "HMMMM", id: "BIMMM", status: "ss"},
            {name: "HMMMM", id: "BIMMM", status: "ss"},
            {name: "HMMMM", id: "BIMMM", status: "ss"},
            {name: "HMMMM", id: "BIMMM", status: "ss"}
        ]

        dispatch({type: LIST_MODELS, payload: data})
    }


}


/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const approve_model = () => {
    return (dispatch) => {
    }
}


/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const reject_model = () => {
    return (dispatch) => {
    }
}

/**
 *
 * @param data
 * @returns {(function(*): void)|*}
 */
export const delete_model = () => {
    return (dispatch) => {
    }
}