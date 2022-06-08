import React from 'react';
import {useParams, useNavigate} from "react-router-dom";
import {connect} from "react-redux"
import {get_single_model as gsm,
        reset_single_model as rsm } from "../../store/actions/modelsActions";
import SyntaxHighlighter from 'react-syntax-highlighter';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import SingleModelActions from "./SingleModelActions";
import styles from "./Models.module.css"
const SingleModel = (props) => {

    const {model_id} = useParams()
    const {single_model, get_single_model, reset_single_model} = props
    const navigator = useNavigate()

    React.useEffect(() => {
        get_single_model({model_id : model_id}, navigator)

        return () => {
            reset_single_model()
        }
    }, [model_id, get_single_model, navigator, reset_single_model])

    if(single_model){
         return (
            <React.Fragment>
                <div className="frame-header">
                    <div className={"row"} style={{alignItems: "center"}}>
                        <div className={"note"} style={{width: "70%"}}>
                            Displaying model <b>{single_model?.name}</b>.
                            This model is in status
                             {single_model?.model_status === "Rejected" ? (
                                 <span className={`${styles.tag} ${styles.tagDeclined} ${styles.tagInline}`}>
                                     REJECTED
                                 </span>
                             ) : single_model?.model_status === "Approved" ? (
                                 <span className={`${styles.tag} ${styles.tagApproved} ${styles.tagInline}`}>
                                     APPROVED
                                 </span>
                             ) : single_model?.model_status === "Pending" ? (
                                 <span className={`${styles.tag} ${styles.tagPending} ${styles.tagInline}`}>
                                     PENDING
                                 </span>
                             ) : null
                             }
                        </div>
                        <SingleModelActions style={{width: "30%"}} single_model={single_model}/>
                    </div>
                    <div className={"row"}>
                        <div>
                            <h4>Model information</h4>
                            <div className={styles.infoWrapper}>
                                <span className={`${styles.tag} ${styles.tagInfo}`}>Description</span>
                                <span className={`${styles.tag} ${styles.tagDesc}`}>{single_model.description}</span>
                            </div>
                            <div className={styles.infoWrapper}>
                                <span className={`${styles.tag} ${styles.tagInfo}`}>Notes</span>
                                <span className={`${styles.tag} ${styles.tagDesc}`}>
                                    {single_model.notes ? single_model.notes : "-" }
                                </span>
                            </div>
                            <div className={styles.infoWrapper}>
                                <span className={`${styles.tag} ${styles.tagInfo}`}>Requested By</span>
                                <span className={`${styles.tag} ${styles.tagDesc}`}>
                                    {single_model.researcher_id ? single_model.researcher_id :
                                    `This model is "${single_model.model_type}"`} </span>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="frame-content">
                    <SyntaxHighlighter language="python" style={docco}>
                        {single_model?.content}
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
        get_single_model : (data, navigator) => dispatch(gsm(data, navigator)),
        reset_single_model : (data) => dispatch(rsm(data))
    }
}

const mapStateToProps = (state) => {
    return{
        single_model : state.models.single_model
    }
}

export default connect(mapStateToProps, mapDispatchToPros)(SingleModel);