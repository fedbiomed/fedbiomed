import React from 'react';
import {Label, Text, Tag, TextArea, Select} from '../components/Inputs'
import Modal from '../components/Modal'
import {TableData, TableInfo} from "../components/Tables"
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';

export const DatasetPreview = (props) => {


    const [preview, setPreview] = React.useState(null)
    const { dataset_id } = useParams();
    const navigator = useNavigate();


    React.useEffect( ()=>{
        get_dataset_preview(dataset_id)
    },[])

    const prepare_info = (data) => {

        let info = {
            'Name': data.name,
            'Type' : data.data_type,
            'Tags' : data.tags.join(', '),
            'Dataset Path' : data.path, 
            "ID" : data.dataset_id          
        }

        return info
    }

    /**
     * 
     * @param {string} dataset_id 
     */
    const get_dataset_preview = (dataset_id) => {
    
        axios.post("/api/datasets/preview" , {dataset_id : dataset_id })
                .then( res => {
                    if (res.status === 200){
                        setPreview(res.data.result) 
                            console.log(res.data)
                    }else{
                        alert(res.data.message)
                        navigator('/datasets') 
                    }
                })
                .catch( (error, res) => {
                    alert(error.response.data.message)
                    navigator('/datasets') 
                })
    }

    return (
        <React.Fragment>
            <h4>Dataset Information</h4>
            {
                preview ? (
                    <TableInfo info={prepare_info(preview)}/>
                ) : null

            }
            <h4>Dataset Preview</h4>
            {
                preview &&  preview.data_preview && preview.data_type === "csv" ? (
                    <TableData table={preview.data_preview}/>
                ) : null
            }
        </React.Fragment>
    );
}

export default DatasetPreview;