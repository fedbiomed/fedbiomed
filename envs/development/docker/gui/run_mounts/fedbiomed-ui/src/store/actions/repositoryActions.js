import axios from 'axios'


/**
 * Action for getting files from repository based on 
 * given path value
 * @param {object} data 
 * @returns null 
 */
export  const getFilesFromRepository = (data) => {
    
    return (dispatch, getState) => {

        let files = getState().repository.files
        let currentLevels = Object.keys(files).sort().map(item => parseInt(item))

        // If base level has been already called
        if(currentLevels.length > 1 && data.path.length === 0){
            return null
        } else {      
            // Send post request to get list of files by given path array
            axios.post('/api/repository/list' , {
                path :  data.path
            }).then( (response) => {
                if(response.status === 200){
                    let data = response.data
                    let level = data.level
                    
                    if (currentLevels.length === 0){
                        files[level] = data.files
                    }else{
                        if(level >= currentLevels[currentLevels.length - 1]){
                            files[level] = data.files
                        }else{
                            let levelsToRemove = currentLevels.slice(level+1)
                            levelsToRemove.forEach(element => {
                                delete files[element] 
                            });
                            files[level] = data.files
                        }
                    }
                        
                    dispatch({type:'LIST_REPOSITORY', payload : {files : files, base:data.base, message : null}})
                }else{
                    console.log(response)
                }
            }

            ).catch( (error) => {
                alert(error)
            })
        }

    }
}