import React from 'react';
import {connect} from "react-redux"
import {Table, TableRows, TableHead } from "../../components/common/Tables";
import {getBIDSPreview} from "../../store/actions/bidsDatasetActions";
import {ReactComponent as TickIcon} from '../../assets/img/tick.svg'
import {ReactComponent as XIcon} from '../../assets/img/x.svg'
const BidsPreview = (props) => {

    const {dataset_id, bids, BIDSPreview } = props

    React.useEffect(() => {
        if((dataset_id && !bids.subject_table) ||
            (dataset_id && dataset_id !== bids.dataset_id)){
            BIDSPreview(dataset_id)
        }
    }, [dataset_id, bids, BIDSPreview])

    const transform = (text) => {
        if(text === false){
            return <XIcon style={{fill:"#f44949"}}/>
        }else if(text === true){
            return <TickIcon style={{fill:"#27ae60"}}/>
        }else{
            return text
        }
    }

    if(props.bids.subject_table){
        return (
            <React.Fragment>
                <Table style={{maxHeight:350,textAlign:'center'}}>
                    <TableHead table={props.bids.subject_table} hoverColumns={false}/>
                    <TableRows table={props.bids.subject_table} hoverColumns={false} transformation={transform}/>
                </Table>
            </React.Fragment>
        );
    }else{
        return null
    }
};


const mapStateToProps = (state) => {
    return {
        bids : state.bidsPreview
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        BIDSPreview: (dataset_id) => dispatch(getBIDSPreview(dataset_id))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(BidsPreview);