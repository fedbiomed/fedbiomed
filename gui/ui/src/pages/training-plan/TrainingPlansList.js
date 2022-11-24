import React from 'react';
import {connect} from 'react-redux';
import {ReactComponent as LaunchIcon} from '../../assets/img/launch.svg'
import {Link} from "react-router-dom";
import {TableHead, TableRow, TableCol, TableBody, EntryTable} from "../../components/common/Tables";
import {list_training_plans} from "../../store/actions/trainingPlansActions";
import Moment from 'react-moment';
import styles from "./TrainingPlans.module.css"
import TableSearchBar from "../../components/common/TableSearchBar";

const TrainingPlansList = (props) => {

    const {list_training_plans} = props
    const [search, setLastSearch] = React.useState({text: null, by: null})
    const [sortBy, setSortBy] = React.useState(null)
    React.useEffect(() => {
        list_training_plans()
    }, [list_training_plans])

    /**
     * Renderer for status field of the table
     * @param status
     * @returns {JSX.Element|null}
     */
    const renderStatus = (status) => {

        switch (status){
            case "Pending":
                return <span className={`${styles.tag} ${styles.tagPending}`}>Pending</span>
            case "Approved":
                return <span className={`${styles.tag} ${styles.tagApproved}`}>Approved</span>
            case "Rejected":
                return <span className={`${styles.tag} ${styles.tagDeclined}`}>Rejected</span>
            default:
                return null
        }
    }

    /**
     * Handler for search typing
     * @param searchText
     * @param by
     */
    const onSearch = (searchText, by) => {
        if(searchText && searchText !== ""){
            list_training_plans({search: {by : by, text: searchText}, sort_by: sortBy})
            setLastSearch({text: searchText, by:by})
        }else if(search.text){
            list_training_plans()
            setLastSearch({text: searchText, by:by})
        }
    }

    const onSort = (sortBy) => {
        setSortBy(sortBy)
        list_training_plans({
            search: search.text ? {by : search.by, text: search.text} : null,
            sort_by : sortBy})
    }

    return (
        <React.Fragment>
            <TableSearchBar
                onSearch={onSearch}
                onSort={onSort}
                by={true}
                byOptions={[
                            {value : "name", name: "Name"},
                            {value : "researcher_id", name: "Researcher ID"},
                            {value : "training_plan_status", name: "Status"},
                            {value : "description", name: "Description"},
                            {value : "training_plan_type", name: "Model Type"},
                            {value : "date_last_action", name: "Last Action Date"},
                            {value : "date_registered", name: "Registration Date"},
                            ]}
                sortOptions = {[
                            {value : null , name: "None"},
                            {value : "name", name: "Name"},
                            {value : "researcher_id", name: "Researcher ID"},
                            {value : "training_plan_status", name: "Status"},
                            {value : "description", name: "Description"},
                            {value : "training_plan_type", name: "Model Type"},
                            {value : "date_last_action", name: "Last Action Date"},
                            {value : "date_registered", name: "Registration Date"},
                            ]}
            />
            <EntryTable>
                <TableHead>
                    <TableRow>
                        <TableCol>Name</TableCol>
                        <TableCol>Researcher</TableCol>
                        <TableCol>Type</TableCol>
                        <TableCol>Description</TableCol>
                        <TableCol>Last Action</TableCol>
                        <TableCol>Date Requested/Registered</TableCol>
                        <TableCol>Status</TableCol>
                        <TableCol>Display</TableCol>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {props.training_plan_list && props.training_plan_list.map((item, key) => {
                        return(
                            <TableRow key={key}>
                                <TableCol>{item.name}</TableCol>
                                <TableCol>{item.researcher_id}</TableCol>
                                <TableCol>{item.training_plan_type.toUpperCase()}</TableCol>
                                <TableCol>{item.description}</TableCol>
                                <TableCol>
                                {item.date_last_action ? (
                                    <Moment parse="DD-MM-YYYY hh:mm:ss" format="DD-MM-YYYY hh:mm">
                                        {item.date_last_action}
                                    </Moment>
                                ) : "-"}
                                    
                                </TableCol>
                                <TableCol>
                                    <Moment parse="DD-MM-YYYY hh:mm:ss" format="DD-MM-YYYY hh:mm">
                                        {item.date_registered}
                                    </Moment>
                                </TableCol>
                                <TableCol transformation={renderStatus}>{item.training_plan_status}</TableCol>
                                <TableCol>
                                    <Link to={{pathname: `preview/${item.training_plan_id}`}}>
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
        list_training_plans : (data) => dispatch(list_training_plans(data))
    }
}

const mapStateToProps = (state) => {
    return{
        training_plan_list : state.training_plans.list
    }
}

export default connect(mapStateToProps, mapDispatchToPros)(TrainingPlansList);