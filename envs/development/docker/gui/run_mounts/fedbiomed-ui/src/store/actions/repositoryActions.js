import axios from 'axios'
import {EP_REPOSITORY_LIST} from '../../constants'

/**
 * Action for getting files from repository based on 
 * given path value
 * @param {object} data 
 * @returns null 
 */
export  const getFilesFromRepository = (data) => {
    
    return (dispatch, getState) => {

        dispatch({type:'SET_LOADING', payload: true})

        let files = getState().repository.files
        let folders = getState().repository.folders
        let currentLevels = Object.keys(files).sort().map(item => parseInt(item))

        // If base level has been already called
        if(currentLevels.includes(0) && data.path.length === 0){
            dispatch({type:'SET_LOADING', payload: false})
            return null
        } else {      
            // Send post request to get list of files by given path array
            axios.post(EP_REPOSITORY_LIST , {
                path :  data.path
            }).then( (response) => {
                dispatch({type:'SET_LOADING', payload: false})
                if(response.status === 200){
                    let data = response.data
                    let level = data.level
                    
                    if (currentLevels.length === 0){
                        files[level] = data.files
                        folders[level] = { displays : data.displays, number:data.number }

                    }else{
                        if(level >= currentLevels[currentLevels.length - 1]){
                            files[level] = data.files
                            folders[level] = { displays : data.displays, number:data.number }
                        }else{
                            let levelsToRemove = currentLevels.slice(level+1)
                            levelsToRemove.forEach(element => {
                                delete files[element]
                                delete folders[element]

                            });
                            files[level] = data.files
                            folders[level] = { displays : data.displays, number:data.number }
                        }
                    }
                    dispatch({type:'LIST_REPOSITORY', payload : {files : files, folders:folders, base: data.base, message : null}})
                }else{
                    console.log(response)
                }
            }

            ).catch( (error) => {
                alert(error)
                dispatch({type:'SET_LOADING', payload: false})
            })
        }

        console.log(folders)

    }
}
