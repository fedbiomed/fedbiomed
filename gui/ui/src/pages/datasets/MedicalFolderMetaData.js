import React from 'react';
import {Label, Tag, Text, TextArea} from "../../components/common/Inputs";
import {connect} from "react-redux"
import {setMedicalFolderDatasetMetadata, setDLPName} from "../../store/actions/medicalFolderDatasetActions";


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
                            onChange={(e) => { setMedicalFolderDatasetMetadata('tags', e.target.value)}}
                            tags={props.metadata.tags ? props.metadata.tags : ""}
                        />

                    </div>
                </div>
                <div className={`row`}>
                    <div className="form-control">
                        <Label>Description <span style={{fontSize:11}}>(min 4 character)</span> </Label>
                        <TextArea name="desc"
                                  type="text"
                                  placeholder="Please type a description for dataset"
                                  onChange={(e) => { setMedicalFolderDatasetMetadata('desc', e.target.value)}}
                                  value={props.metadata.desc ? props.metadata.desc : "" }
                        />
                    </div>
                </div>

                {props.use_new_mod2fol_association ?
                    <div className={`row`}>
                        <div className="form-control">
                            <TextArea name="desc"
                                      type="text"
                                      placeholder="Please enter an optional name for the data loading customizations that you created."
                                      onChange={(e) => { props.setDLPName(e.target.value)}}
                                      value={props.dlp_name}
                            />
                        </div>
                    </div> : null
                }
        </div>
    );
};

const mapStateToProps = (state) => {
    return {
        metadata : state.medicalFolderDataset.metadata,
        use_new_mod2fol_association : state.medicalFolderDataset.use_new_mod2fol_association,
        dlp_name : state.medicalFolderDataset.dlp_name,
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        setMedicalFolderDatasetMetadata : (data) => dispatch(setMedicalFolderDatasetMetadata(data)),
        setDLPName : (data) => dispatch(setDLPName(data))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(MedicalFolderMetadata);