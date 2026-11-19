import React from 'react';
import {Label, Tag, Text, TextArea} from "../../../components/common/Inputs";
import {connect} from "react-redux"
import {setMedicalFolderDatasetMetadata} from "../../../store/actions/medicalFolderDatasetActions";
import {setDLPDesc} from "../../../store/actions/dataLoadingPlanActions";
import {EuiCallOut} from '@elastic/eui'

const MedicalFolderMetadata = (props) => {

    const setMedicalFolderDatasetMetadata = (name, value) => {
        props.setMedicalFolderDatasetMetadata({[name] : value})
    }

    return (
        <div>
            <div className="row">
                    <div className="form-control" >
                        <Label>Dataset Name <span style={{fontSize:11}}>(min 4 character)</span>
                        </Label>
                        <Text
                            name={"name"}
                            type="text"
                            placeholder="Enter name for dataset"
                            onChange={(e) => { setMedicalFolderDatasetMetadata('name', e.target.value)}}
                            value={props.metadata.name ? props.metadata.name : ""}
                        />
                    </div>
                    <div className="form-control" >
                        <Label>Enter tags for dataset <span style={{fontSize:11}}>(Please press enter or space to register tag)</span></Label>
                        <Tag
                            name={"tags"}
                            type="text"
                            placeholder="Enter tags"
                            onChange={(e) => { setMedicalFolderDatasetMetadata('tags', e.target.value); }}
                            tags={props.metadata.tags ? props.metadata.tags : ""}
                        />

                    </div>
                </div>
                <div className={`row`}>
                    <div className="form-control">
                        <Label>Dataset description <span style={{fontSize:11}}>(min 4 character)</span> </Label>
                        <TextArea name="desc"
                                  type="text"
                                  placeholder="Please type a description for dataset"
                                  onChange={(e) => { setMedicalFolderDatasetMetadata('desc', e.target.value)}}
                                  value={props.metadata.desc ? props.metadata.desc : "" }
                        />
                    </div>
                </div>

                {props.use_custom_mod2fol ?
                    <React.Fragment>
                    {!props.use_preexisting_dlp || !props.same_as_preexisting_dlp ?
                        <div className={`row`}>
                            <div className="form-control">
                                <Label>Customization name <span style={{fontSize:11}}>(min 4 character)</span> </Label>
                                <Text name="desc"
                                          type="text"
                                          placeholder="Enter a name for the data loading customizations that you created."
                                          onChange={(e) => { props.setDLPDesc(e.target.value)}}
                                          value={props.dlp_name}
                                />
                            </div>
                        </div> : <div>

                              <EuiCallOut title="Customization Info" color="warning" iconType="help">
                                <p>
                                    Reusing unchanged existing customizations, don't need to save them again.
                                </p>
                              </EuiCallOut>

                        </div>
                    }

                    </React.Fragment>: null
                }
        </div>
    );
};

const mapStateToProps = (state) => {
    return {
        metadata : state.medicalFolderDataset.metadata,
        use_custom_mod2fol : state.medicalFolderDataset.use_custom_mod2fol,
        dlp_name : state.dataLoadingPlan.dlp_name,
        use_preexisting_dlp : state.dataLoadingPlan.use_preexisting_dlp,
        same_as_preexisting_dlp : state.dataLoadingPlan.same_as_preexisting_dlp,
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        setMedicalFolderDatasetMetadata : (data) => dispatch(setMedicalFolderDatasetMetadata(data)),
        setDLPDesc : (data) => dispatch(setDLPDesc(data))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(MedicalFolderMetadata);