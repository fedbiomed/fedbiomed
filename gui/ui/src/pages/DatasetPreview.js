import React from 'react';
import {TableData, TableInfo} from "../components/Tables"
import { useParams} from 'react-router-dom';
import {EP_DATASET_PREVIEW, DATA_NOTFOUND, EP_DATASET_UPDATE} from '../constants'
import {Text, Tag, TextArea} from '../components/Inputs'
import {Button, ButtonsWrapper} from '../components/Button'
import axios from 'axios';
import Repository from "./Repository";

export const DatasetPreview = (props) => {


    const [preview, setPreview] = React.useState(null)
    const [error, setError] = React.useState(null)
    const [loading, setLoading] = React.useState(true)
    const [edit , setEdit] = React.useState(false)
    const { dataset_id } = useParams();
    const setHeader = props.setHeader

    React.useEffect( ()=>{
        get_dataset_preview(dataset_id)
        if(setHeader){
            setHeader('Dataset Preview')
        }
    },[dataset_id, setHeader])

    /**
     * Activate edit view
     */
    const onEditClick = () => {
        setEdit(true)
    }

    /**
     * On edit view, when the input value has changed
     * @param e
     */
    const onEditChange = (e) => {
        let name = e.target.name
        setPreview({
            ...preview,
            [name] : e.target.value
        })
    }

    /**
     * When tags input value has changed
     * @param tags
     */
    const onTagsChange = (name, tags) => {
        setPreview({
            ...preview,
            tags : tags
        })
    }

    /**
     * Info object preparation for TableInfo
     * @param data
     * @returns {{
     * Type: {editable: boolean, value: *},
     * Description: {input: JSX.Element, editable: boolean, value},
     * "Dataset Path": {editable: boolean, value},
     * ID: {editable: boolean, value: (string|*)},
     * Tags: {input: JSX.Element, editable: boolean, value: string},
     * Name: {input: JSX.Element, editable: boolean, value}}}
     */
    const prepare_info = (data) => {

        let info = {
            'Name': { value : data.name,
                      editable:true,
                      input : <Text name="name" value={data.name} onChange={onEditChange}/>
               },
            'Type': { value : data.data_type ,
                      editable: false
                    },
            'Shape': { value : data.shape.join(' x '),
                      editable: false
                     },
            'Tags': { value : data.tags.join(', '),
                      editable : true,
                      input : <Tag tags={data.tags}  onTagsChange={onTagsChange}/>
                    },
            'Description': {value : data.description,
                            editable: true,
                            input : <TextArea name="description" onChange={onEditChange} value={data.description}/>
                            },
            'Dataset Path' : { value : data.path,
                               editable: false
                             },
            "ID" : {value : data.dataset_id,
                    editable: false
                }
        }

        return info
    }

    /**
     * This function gets dataset preview information
     * @param {string} dataset_id 
     */
    const get_dataset_preview = (dataset_id) => {
    
        axios.post(EP_DATASET_PREVIEW, {dataset_id : dataset_id })
                .then( res => {
                    if (res.status === 200){
                        setPreview(res.data.result)
                    }else{
                        setError(res.data.message)

                    }
                    setLoading(false)
                })
                .catch( (error) => {
                    if (error.response) {
                        setError(error.response.data.message)
                    }else{
                        setError('Unexpected Error : ' + error.toString())
                    }
                    setLoading(false)
                })
    }

    /**
     * Update request for dataset
     */
    const updateDataset = () => {

        axios.post(
            EP_DATASET_UPDATE,
            {
                name : preview.name,
                desc : preview.description,
                tags : preview.tags,
                dataset_id: preview.dataset_id

            }).then(res => {
                setEdit(false)
                setPreview( {
                    ...preview,
                    ...res.data.result
                })
            }).catch(error => {
                alert('Cant, update dataset')
                window.location.reload();
            })
    }

    /**
     * Cancels edit view
     */
    const editCanceled = () => {
        setEdit(false)
    }

    return (
        <div className="data-preview">
            {!loading ?
                    !error ? (
                    <>
                        <h4>Dataset Information</h4>
                        {
                            preview ? (
                                <TableInfo edit={edit} info={prepare_info(preview)}/>
                            ) : null

                        }
                         <ButtonsWrapper className={"float-right"}>
                            { edit ? (
                               <>
                                <Button type={"negative"} onClick={editCanceled}>Cancel</Button>
                                <Button onClick={updateDataset}>Update</Button>
                               </>
                            ) : (
                               <Button onClick={onEditClick}>Edit</Button>
                            )}
                        </ButtonsWrapper>
                        <hr/>
                        <h4>Dataset Preview</h4>
                        {
                            preview && preview.data_preview ?
                                preview.data_type === "csv" ? (
                                    <TableData table={preview.data_preview}/>
                                ) : (
                                    <div className={`repository`}>
                                        <Repository
                                            path={preview.data_preview}
                                            after={preview.data_preview.length}
                                            mode={`preview`}
                                        />
                                    </div>
                                ) : (
                                    <div className={`error-box`}>
                                        {DATA_NOTFOUND}
                                    </div>
                                )
                        }
                    </>
                    ) : (
                    <div className={`error-box`}>
                        {error}
                    </div>
                  ) : (
                    <div>Loading</div>
                )
            }
        </div>
    );
}

export default DatasetPreview;