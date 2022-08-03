import React from 'react';
import axios from '../AxiosErrorHandler';
import  {TableInfo} from "../components/common/Tables"
import {EP_CONFIG_NODE_ENVIRON} from '../constants'


const Configuration = (props) => {

    const [config, setConfig] = React.useState(null)


    React.useEffect( () => {    
        get_node_config()
    }, [])

    /**
     * 
     * @param {string} dataset_id 
     */
    const get_node_config = () => {

        axios.get(EP_CONFIG_NODE_ENVIRON, {})
                .then( res => {
                    if (res.status === 200){
                        setConfig(res.data.result) 
                    }else{
                        alert(res.data.message)
                    }
                })
                .catch( (error) => {
                    console.log(error)
                    //alert(error.data.message)
                })
    }

    const parse_config = () => {

        let obj = {}
        Object.keys(config).forEach( item => {
            obj[item] = {value : config[item]}
        })

        return obj
    }
    return (
        <React.Fragment>
            {
                config ? (
                    <TableInfo info={parse_config(config)}/>
                ) : null
            }
        </React.Fragment>
    );
}

export default Configuration;