import axios from "axios"
import {EP_DATASET_ADD ,
        EP_DEFAULT_DATASET_ADD,
        EP_DATASETS_LIST,
        EP_DATASET_REMOVE
    } from "../../constants";


/**
 * Action for adding new dataset
 * @param data
 * @returns {(function(*, *): void)|*}
 */
export const addNewDataset = (data) => {
    return (dispatch, getState) => {
        dispatch({type:'SET_LOADING', payload: true})

        axios.post(EP_DATASET_ADD, {
            ...data
        }).then(res => {
            dispatch({type:'SET_LOADING', payload: false})
            if(res.status === 200){
                dispatch({type: 'ADD_DATASET_RESULT' , payload: res.data.result})
                dispatch({type: 'SUCCESS_MODAL' , payload: "Dataset has been successfully added"})
            }else{
                dispatch({type: 'ERROR_MODAL' , payload:res.data.result.message})
            }
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: false})
            if(error.response){
                dispatch({type: 'ERROR_MODAL', payload: 'Error while adding new dataset: ' + error.response.data.message})
            }else{
                dispatch({type: 'ERROR_MODAL', payload: 'Unexpected Error:' + error.toString()})
            }
        })
    }
}


export const addDefaultDataset = (data) => {

    return (dispatch) => {

        dispatch({type:'SET_LOADING', payload: true})

        // Notify it is requested
        dispatch({type: 'DEFAULT_DATASET_ADD_REQUEST'})

        axios.post(EP_DEFAULT_DATASET_ADD, data).then( res => {
                dispatch({type:'SET_LOADING', payload: false})
                if(res.status === 200){
                    dispatch({type: 'SUCCESS_MODAL', payload: "Default dataset has been added" })
                    dispatch({type: 'DEFAULT_DATASET_ADD_SUCCESS', payload: res.data.result})
                }else{
                    dispatch({type: 'ERROR_MODAL' , payload: res.data.message })
                }

        }).catch( error => {
                dispatch({type:'SET_LOADING', payload: false})
                if(error.response){
                    dispatch({type: 'ERROR_MODAL', payload: 'Error while adding default dataset: ' + error.response.data.message})
                }else{
                    dispatch({type: 'ERROR_MODAL', payload: 'Unexpected error:' + error.toString()})
                }
        })
    }
}

/**
 * Request action for listing all datasets in the node
 * @param {object} data  Object that pass to POST
 * @returns {dispatch}
 */
export const listDatasets = (data) => {

    return (dispatch, getState) => {
        dispatch({type:'SET_LOADING', payload: true})
        axios.post(EP_DATASETS_LIST , data)
             .then( res => {
                if (res.status === 200){
                    dispatch({type:'SET_LOADING', payload: false})
                    dispatch({ type : "GET_DATASETS", payload: res.data.result})

                }else{
                    dispatch({type:'SET_LOADING', payload: false})
                    alert(res.data.message)
                }
             })
             .catch( error => {
                 dispatch({type:'SET_LOADING', payload: false})
                 alert(error)
             })

    }
}


/**
 * Search dataset function
 * @param data
 * @returns {(function(*, *): void)|*} dispatcher
 */
export const searchDataset = (data) => {
        return (dispatch, getState) => {
        dispatch({type:'SET_LOADING', payload: true})
        axios.post(EP_DATASETS_LIST , data)
             .then( res => {
                if (res.status === 200){
                    dispatch({type:'SET_LOADING', payload: false})
                    dispatch({type : "SEARCH_DATASET_RESULT", payload: res.data.result})
                }else{
                    dispatch({type:'SET_LOADING', payload: false})
                    alert(res.data.message)
                }
             })
             .catch( error => {
                 dispatch({type:'SET_LOADING', payload: false})
                 alert(error)
             })

    }
}

/**
 * Request action for removing single dataset from node
 * @param {object} data Object that has dataset_id
 * @returns {dispatch}
 */
export const removeDataset = (data) => {

    return (dispatch, getState) => {

        dispatch({type:'SET_LOADING', payload: true})
        // Get current datasets state
        let datasets = getState().datasets

        axios.post(EP_DATASET_REMOVE, {dataset_id : data.dataset_id})
             .then(res => {

                if(res.status === 200){

                    let index = datasets.datasets.map(function(e) {
                        return e.dataset_id;
                    }).indexOf(data.dataset_id);

                    if (index > -1) {
                        datasets.datasets.splice(index, 1);
                        dispatch({ type : "UPDATE_DATASETS", payload: datasets.datasets})
                         dispatch({type:'SET_LOADING', payload: false})
                        dispatch({type: "SUCCESS_MODAL", payload: "Dataset has been removed"})

                    }
                }else{
                    dispatch({type:'SET_LOADING', payload: false})
                    dispatch({type: "ERROR_MODAL", payload: res.data.message})
                }

             })
             .catch(error => {
                 dispatch({type:'SET_LOADING', payload: false})
                if(error.response){
                    dispatch({type: 'ERROR_MODAL', payload: 'Error while adding default dataset: ' + error.response.data.message})
                }else{
                    dispatch({type: 'ERROR_MODAL', payload: 'Unexpected error:' + error.toString()})
                }
             })

    }

}
