import React from 'react';
import {connect} from "react-redux"
import {DataTableRows, DataTableHead, TableWrapper, TableData} from "../../../components/common/Tables";
import {getMedicalFolderPreview} from "../../../store/actions/medicalFolderDatasetActions";
import {ReactComponent as TickIcon} from '../../../assets/img/tick.svg'
import {ReactComponent as XIcon} from '../../../assets/img/x.svg'

const MedicalFolderDatasetPreview = (props) => {

    const {dataset_id, medical_folder_dataset, MedicalFolderPreview } = props

    React.useEffect(() => {
        if((dataset_id && !medical_folder_dataset.subject_table) ||
            (dataset_id && dataset_id !== medical_folder_dataset.dataset_id)){
            MedicalFolderPreview(dataset_id)
        }
    }, [dataset_id, medical_folder_dataset, MedicalFolderPreview])

    const transform = (text) => {
        if(text === false){
            return <XIcon style={{fill:"#f44949"}}/>
        }else if(text === true){
            return <TickIcon style={{fill:"#27ae60"}}/>
        }else{
            return text
        }
    }

    if(props.medical_folder_dataset.subject_table){
        return (
            <React.Fragment>
                <TableWrapper maxHeight={350}>
                    <TableData style={{textAlign:'center'}}>
                        <DataTableHead table={props.medical_folder_dataset.subject_table} showIndex={true} hoverColumns={false} indexName={"Subjects"}/>
                        <DataTableRows table={props.medical_folder_dataset.subject_table} showIndex={true} hoverColumns={false} transformation={transform}/>
                    </TableData>
                </TableWrapper>
            </React.Fragment>
        );
    }else{
        return null
    }
};


const mapStateToProps = (state) => {
    return {
        medical_folder_dataset : state.medicalFolderPreview
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        MedicalFolderPreview: (dataset_id) => dispatch(getMedicalFolderPreview(dataset_id))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(MedicalFolderDatasetPreview);