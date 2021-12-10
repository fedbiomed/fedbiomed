import axios from "axios"
import {EP_DATASET_ADD ,
        EP_DATASET_UPDATE,
        EP_DEFAULT_DATASET_ADD
    } from "../../constants";


/**
 * Action for adding new dataset
 * @param data
 * @returns {(function(*, *): void)|*}
 */
export const addNewDataset = (data) => {
    return (dispatch, getState) => {

        axios.post(EP_DATASET_ADD, {
            ...data
        }).then(res => {
            if(res.status === 200){
                dispatch({type: 'NEW_DATASET_ADD_SUCCESS' , payload: res.data.result})
            }else{
                dispatch({type: 'NEW_DATASET_ADD_ERROR' , payload:res.data.result.message})
            }
        }).catch(error => {
            if(error.response){
                dispatch({type: 'NEW_DATASET_ADD_ERROR', payload: 'Error while adding new dataset: ' + error.response.data.message})
            }else{
                dispatch({type: 'NEW_DATASET_ADD_ERROR', payload: 'Unexpected Error:' + error.toString()})
            }
        })
    }
}


export const addDefaultDataset = (data) => {

    return (dispatch) => {

        // Notify it is requested
        dispatch({type: 'DEFAULT_DATASET_ADD_REQUEST'})

        axios.post(EP_DEFAULT_DATASET_ADD, {
            name:'mnist'
        }).then( res => {
                if(res.status === 200){
                    dispatch({type: 'DEFAULT_DATASET_ADD_SUCCESS' , payload: res.data.result })
                }else{
                    dispatch({type: 'DEFAULT_DATASET_ADD_ERROR' , payload: res.data.message })
                }

        }).catch( error => {
                if(error.response){
                dispatch({type: 'DEFAULT_DATASET_ADD_ERROR', payload: 'Error while adding default dataset: ' + error.response.data.message})
                }else{
                    dispatch({type: 'DEFAULT_DATASET_ADD_ERROR', payload: 'Unexpected error:' + error.toString()})
                }
        })
    }
}

/**
 * Request action for listing all datasets in the node  
 * @param {object} data  Object that pass to POST
 * @returns {dispatch}
 */
export const listDatasets = () => {

    return (dispatch, getState) => {

        axios.post("/api/datasets/list" , {})
             .then( res => {

                if (res.status === 200){

                    dispatch({ type : "GET_DATASETS", payload: res.data.result}) 

                }else{
                    alert(res.data.message)
                }
             })
             .catch( error => {
                 alert(error)
             })

    }
}

/**
 * Reqeust action for removing single dataset from node
 * @param {object} data Object that has dataset_id 
 * @returns {dispatch}
 */
export const removeDataset = (data) => {

    return (dispatch, getState) => {

        // Get current datasets state 
        let datasets = getState().datasets

        axios.post('/api/datasets/remove', {dataset_id : data.dataset_id})
             .then(res => {

                if(res.status === 200){

                    let index = datasets.datasets.map(function(e) {
                        return e.dataset_id;
                    }).indexOf(data.dataset_id);
                    
                    if (index > -1) {
                        datasets.datasets.splice(index, 1);
                        dispatch({ type : "UPDATE_DASETS", payload: datasets.datasets})
                        alert('Dataset has been removed')
                    }
                    
                }else{
                    alert(res.data.message)
                }

             })
             .catch(error => {
                 if(error.status === 400){
                    alert(error.data.message)
                 }else{
                    alert(error.message)
                 }

                 
             })
            
    }

}