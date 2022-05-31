import React from 'react';
import {Label, Tag, Text, TextArea} from "../../components/common/Inputs";

const DatasetMetadata = (props) => {

    const [metadata, setMetaData ] = React.useState({})

    const setDatasetMetadata = (name, value) => {
        setMetaData({...metadata, [name] : value})
        if(props.onMetadataChange ){
            props.onMetadataChange(metadata)
        }
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
                            onChange={(e) => { setDatasetMetadata('name', e.target.value)}}
                            value={metadata.name}
                        />
                    </div>
                    <div className="form-control" >
                        <Label>Enter tags for dataset <span style={{fontSize:11}}>(Please press enter or space to register tag)</span></Label>
                        <Tag
                            name={"tags"}
                            type="text"
                            placeholder="Enter tags"
                            onChange={(e) => { setDatasetMetadata('tags', e.target.value)}}
                            tags={metadata.tags}
                        />

                    </div>
                </div>
                <div className={`row`}>
                    <div className="form-control">
                        <Label>Description <span style={{fontSize:11}}>(min 4 character)</span> </Label>
                        <TextArea name="desc"
                                  type="text"
                                  placeholder="Please type a description for dataset"
                                  onChange={(e) => { setDatasetMetadata('desc', e.target.value)}}
                                  value={metadata.desc}
                        />
                    </div>
                </div>
        </div>
    );
};

export default DatasetMetadata;