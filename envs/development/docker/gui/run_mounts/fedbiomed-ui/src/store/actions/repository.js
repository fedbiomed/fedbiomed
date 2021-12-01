import axios from 'axios'


/**
 * Action for getting files from repository based on 
 * given path value
 * @param {object} data 
 * @returns null 
 */
export  const getFilesFromRepository = (data) => {
    
    return (dispacth) => {


        axios.post('/api/repository/list' , {
            path :  data.path
        }).then( (response) => {
            if(response.status === 200){
                console.log(response)
            }else{
                console.log(response)
            }
        }

        ).catch( (error) => {
            console.log(error)
        })

        dispacth({type: 'LIST_REPOSITORY', payload : {list : []} })
    }
}