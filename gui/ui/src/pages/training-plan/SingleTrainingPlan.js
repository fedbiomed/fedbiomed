import React from 'react';
import {useParams, useNavigate} from "react-router-dom";
import {connect} from "react-redux"
import {get_single_training_plan as gsm,
        reset_single_training_plan as rsm } from "../../store/actions/trainingPlansActions";
import SyntaxHighlighter from 'react-syntax-highlighter';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import SingleTrainingPlanActions from "./SingleTrainingPlanActions";
import styles from "./TrainingPlans.module.css"


const SingleTrainingPlan = (props) => {

    const {training_plan_id} = useParams()
    const {single_training_plan, get_single_training_plan, reset_single_training_plan} = props
    const navigator = useNavigate()

    React.useEffect(() => {
        // Get single training plan
        get_single_training_plan({training_plan_id : training_plan_id}, navigator)
        return () => {
            reset_single_training_plan()
        }
    }, [training_plan_id, get_single_training_plan, navigator, reset_single_training_plan])

    if(single_training_plan){
         return (
            <React.Fragment>
                <div className="frame-header">
                    <div className={"row"} style={{alignItems: "center"}}>
                        <div className={"note"} style={{width: "70%"}}>
                            Displaying training plan <b>{single_training_plan?.name}</b>.
                            This training plan is in status
                             {single_training_plan?.training_plan_status === "Rejected" ? (
                                 <span className={`${styles.tag} ${styles.tagDeclined} ${styles.tagInline}`}>
                                     REJECTED
                                 </span>
                             ) : single_training_plan?.training_plan_status === "Approved" ? (
                                 <span className={`${styles.tag} ${styles.tagApproved} ${styles.tagInline}`}>
                                     APPROVED
                                 </span>
                             ) : single_training_plan?.training_plan_status === "Pending" ? (
                                 <span className={`${styles.tag} ${styles.tagPending} ${styles.tagInline}`}>
                                     PENDING
                                 </span>
                             ) : null
                             }
                        </div>
                        <SingleTrainingPlanActions style={{width: "30%"}} single_training_plan={single_training_plan}/>
                    </div>
                    <div className={"row"}>
                        <div>
                            <h4>Model information</h4>
                            <div className={styles.infoWrapper}>
                                <span className={`${styles.tag} ${styles.tagInfo}`}>Description</span>
                                <span className={`${styles.tag} ${styles.tagDesc}`}>{single_training_plan.description}</span>
                            </div>
                            <div className={styles.infoWrapper}>
                                <span className={`${styles.tag} ${styles.tagInfo}`}>Notes</span>
                                <span className={`${styles.tag} ${styles.tagDesc}`}>
                                    {single_training_plan.notes ? single_training_plan.notes : "-" }
                                </span>
                            </div>
                            <div className={styles.infoWrapper}>
                                <span className={`${styles.tag} ${styles.tagInfo}`}>Requested By</span>
                                <span className={`${styles.tag} ${styles.tagDesc}`}>
                                    {single_training_plan.researcher_id ? single_training_plan.researcher_id :
                                    `This training plan is "${single_training_plan.training_plan_type}"`} </span>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="frame-content">
                    <SyntaxHighlighter language="python" style={docco}>
                        {single_training_plan?.training_plan}
                    </SyntaxHighlighter>
                </div>
                <div className="frame-footer">
                </div>
            </React.Fragment>
        );
    }else{
        return null
    }
};



const mapDispatchToPros = (dispatch) => {
    return{
        get_single_training_plan : (data, navigator) => dispatch(gsm(data, navigator)),
        reset_single_training_plan : (data) => dispatch(rsm(data))
    }
}

const mapStateToProps = (state) => {
    return{
        single_training_plan : state.training_plans.single_training_plan
    }
}

export default connect(mapStateToProps, mapDispatchToPros)(SingleTrainingPlan);