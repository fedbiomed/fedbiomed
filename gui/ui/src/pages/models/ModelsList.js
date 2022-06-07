import React from 'react';
import {connect} from 'react-redux';
import {useParams} from "react-router-dom";
import {Table, TableHead, TableRow, TableCol, TableBody, EntryTable} from "../../components/common/Tables";
import {list_models} from "../../store/actions/modelsActions";

const ModelsList = (props) => {

    const params = useParams()

    React.useEffect(() => {
        list_models()
    }, [])

    return (
        <React.Fragment>
            <EntryTable>
                <TableHead>
                    <TableRow>
                        <TableCol>Name</TableCol>
                        <TableCol>Type</TableCol>
                        <TableCol>Date</TableCol>
                        <TableCol>Status</TableCol>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {props.model_list?.map((item, key) => {
                        return(
                            <TableRow>
                                <TableCol>{item.name}</TableCol>
                                <TableCol>{item.name}</TableCol>
                                <TableCol>{item.name}</TableCol>
                                <TableCol>{item.name}</TableCol>
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