import axios from 'axios'
import {EP_REPOSITORY_LIST} from '../../constants'

/**
 * Action for getting files from repository based on 
 * given path value
 * @param {object} data 
 * @returns null 
 */
export  const getFilesFromRepository = (data, fresh = false) => {
    
    return (dispatch, getState) => {

        dispatch({type:'SET_LOADING', payload: true})
        let files
        let folders

        if(fresh === true){
            files = {}
            folders = {}
        }else{
            files = getState().repository.files
            folders = getState().repository.folders
        }
        let currentLevels = Object.keys(files).sort().map(item => parseInt(item))

        // Send post request to get list of files by given path array
        axios.post(EP_REPOSITORY_LIST , {
            path :  data.path
        }).then( (response) => {
            dispatch({type:'SET_LOADING', payload: false})
            if(response.status === 200){
                let data = response.data.result
                let level = data.level

                //Sorting Files
                let d = data.files.filter(file => file.type === 'dir')
                let f = data.files.filter(file => file.type === 'file')
                d.sort((a, b) => a.name.localeCompare(b.name))
                f.sort((a, b) => a.name.localeCompare(b.name))
                d.push(...f)
                data.files = d

                if (currentLevels.length === 0){
                    files[level] = data.files
                    folders[level] = { displays : data.displays, number:data.number, path: data.path}
                }else{
                    let levelsToRemove
                    if(level >= currentLevels[currentLevels.length - 1]){
                        files[level] = data.files
                        folders[level] = { displays : data.displays, number:data.number, path: data.path }
                    }else{
                        if(currentLevels[0] !== 0 ){
                            delete files[currentLevels.at(-1)]
                            delete folders[currentLevels.at(-1)]
                        }else{
                            levelsToRemove = currentLevels.slice(level+1)
                            levelsToRemove.forEach(element => {
                                delete files[element]
                                delete folders[element]
                            });
                        }

                        files[level] = data.files
                        folders[level] = { displays : data.displays, number:data.number, path: data.path}
                    }
                }

                dispatch({type:'LIST_REPOSITORY', payload : {files: files, folders: folders, base: data.base, level:level, message : null}})
            }else{
                dispatch({type: 'ERROR_MODAL' , payload: response.data.result.message})
            }
        }

        ).catch( (error) => {
            dispatch({type:'SET_LOADING', payload: false})
            if(error.response){
                dispatch({type: 'ERROR_MODAL', payload: 'Error while listing files: ' + error.response.data.message})
            }else{
                dispatch({type: 'ERROR_MODAL', payload: 'Unexpected Error:' + error.toString()})
            }
        })
    }
}
