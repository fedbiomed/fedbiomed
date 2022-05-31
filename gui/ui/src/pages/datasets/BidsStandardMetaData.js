import React from 'react';
import {Label, Tag, Text, TextArea} from "../../components/common/Inputs";
import {connect} from "react-redux"
import {setBIDSDatasetMetadata} from "../../store/actions/bidsDatasetActions";


const DatasetMetadata = (props) => {

    const setBIDSDatasetMetadata = (name, value) => {
        props.setBIDSDatasetMetadata({[name] : value})
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
                            onChange={(e) => { setBIDSDatasetMetadata('name', e.target.value)}}
                            value={props.metadata.name ? props.metadata.name : ""}
                        />
                    </div>
                    <div className="form-control" >
                        <Label>Enter tags for dataset <span style={{fontSize:11}}>(Please press enter or space to register tag)</span></Label>
                        <Tag
                            name={"tags"}
                            type="text"
                            placeholder="Enter tags"
                            onChange={(e) => { setBIDSDatasetMetadata('tags', e.target.value)}}
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
                                  onChange={(e) => { setBIDSDatasetMetadata('desc', e.target.value)}}
                                  value={props.metadata.desc ? props.metadata.desc : "" }
                        />
                    </div>
                </div>
        </div>
    );
};

const mapStateToProps = (state) => {
    return {
        metadata : state.bidsDataset.metadata
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        setBIDSDatasetMetadata : (data) => dispatch(setBIDSDatasetMetadata(data))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(DatasetMetadata);