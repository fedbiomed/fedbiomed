import React from 'react';
import {connect} from 'react-redux';
import {ReactComponent as LaunchIcon} from '../../assets/img/launch.svg'
import {Link, useParams} from "react-router-dom";
import {TableHead, TableRow, TableCol, TableBody, EntryTable} from "../../components/common/Tables";
import {list_models} from "../../store/actions/modelsActions";
import styles from "./Models.module.css"

const ModelsList = (props) => {



    React.useEffect(() => {
        props.list_models()
    }, [])

    const renderStatus = (status) => {

        switch (status){
            case "Pending":
                return <span className={`${styles.tag} ${styles.tagPending}`}>Pending</span>
            case "Approved":
                return <span className={`${styles.tag} ${styles.tagApproved}`}>Approved</span>
            case "Rejected":
                return <span className={`${styles.tag} ${styles.tagDeclined}`}>Rejected</span>
        }
    }
    console.log(props.model_list)
    return (
        <React.Fragment>
            <EntryTable>
                <TableHead>
                    <TableRow>
                        <TableCol>Name</TableCol>
                        <TableCol>Type</TableCol>
                        <TableCol>Description</TableCol>
                        <TableCol>Status</TableCol>
                        <TableCol>Display</TableCol>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {props.model_list && props.model_list.map((item, key) => {
                        return(
                            <TableRow key={key}>
                                <TableCol>{item.name}</TableCol>
                                <TableCol>{item.model_type.toUpperCase()}</TableCol>
                                <TableCol>{item.description}</TableCol>
                                <TableCol transformation={renderStatus}>{item.model_status}</TableCol>
                                <TableCol>
                                    <Link to={{pathname: `preview/${item.model_id}`}}>
                                        <LaunchIcon/>
                                    </Link>
                                </TableCol>
                            </TableRow>
                        )
                    })}
                </TableBody>
            </EntryTable>
        </React.Fragment>
    );
};


const mapDispatchToPros = (dispatch) => {
    return{
        list_models : (data) => dispatch(list_models(data))
    }
}

const mapStateToProps = (state) => {
    return{
        model_list : state.models.list
    }
}

export default connect(mapStateToProps, mapDispatchToPros)(ModelsList);