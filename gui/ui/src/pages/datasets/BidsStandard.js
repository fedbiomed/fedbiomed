import React, {useState} from 'react';
import styles from "./AddDataset.module.css"
import Step from "../../components/layout/Step"
import {connect} from "react-redux"
import FileBrowser from "../../components/common/FileBrowser";
import {setFolderPath,
    setFolderRefColumn,
    setReferenceCSV,
    setBIDSDatasetMetadata,
    addBIDSDataset} from "../../store/actions/bidsDatasetActions"
import {SelectiveTable} from "../../components/common/Tables";
import BidsSubjectInformation from "./BidsSubjectInformation";
import Button, {ButtonsWrapper} from "../../components/common/Button";
import {useNavigate, useParams, useLocation} from "react-router-dom";
import DatasetMetadata from "./BidsStandardMetaData";


const withRouter = (Component) =>  {
    function ComponentWithRouterProp(props) {

      let location = useLocation();
      let navigate = useNavigate();
      let params = useParams();
      return (
        <Component
          {...props}
          router={{location, navigate, params}}
        />
      );
    }
    return ComponentWithRouterProp;
}


export class  BidsStandard extends React.Component {

    constructor(props) {
        super(props);
        this.state = {}
    }

    setDataPath = (path) => {
        this.props.setFolderPath(path)
    }

    setReferenceCSV = (path) => {
        if (this.props.bidsDataset.reference_csv) {
            this.props.setFolderRefColumn({name: null, index: null})
        }
        this.props.setReferenceCSV(path)
    }

    setReferenceFolderIDColumn = (index) => {
        this.props.setFolderRefColumn({
            index: index,
            name: this.props.bidsDataset.reference_csv.data.columns[index]
        })
    }
    setBIDSDatasetMetadata = (data) => {
        this.props.setBIDSDatasetMetadata(data)
    }

    addDataset = () => {
        this.props.addBIDSDataset(this.props.router.navigate)
    }


    render() {
        return (
            <div className={styles.main}>
                <Step key={1}
                      step={1}
                      desc={'Please select the root folder that contains BIDS Nifti format brain images. '}
                >
                   <FileBrowser
                        folderPath = {this.props.bids_root ? this.props.bids_root : null}
                        onSelect = {this.setDataPath}
                        buttonText = "Select Folder"
                        onlyFolders={true}
                   />
                    {this.props.bidsDataset.modalities ?
                        (<div className={''}>
                            <label>Modalities: </label>
                            {this.props.bidsDataset.modalities.map((item, key) => {
                                  return(
                                      <span className={styles.modalities} key={key}>{item}</span>
                                  )
                            })}
                        </div>) : null
                    }
                </Step>

                {this.props.bids_root ?(
                    <Step
                        key={2}
                        step={2}
                        desc={'Please select reference CSV file where al patient IDs are stored '}
                    >
                       <FileBrowser
                            folderPath = {this.props.bidsDataset.reference_csv ? this.props.bidsDataset.reference_csv.path : null}
                            onSelect = {this.setReferenceCSV}
                            onlyExtensions = {[".csv"]}
                            buttonText = "Select Data File"
                       />
                    </Step>
                    ) : null
                }

                {this.props.bids_root && this.props.bidsDataset.reference_csv != null ? (
                    <Step
                        key={3}
                        step={3}
                        desc={'Please select to ID column from reference csv'}
                    >
                        <SelectiveTable
                            style={{maxHeight:350}}
                            table={this.props.bidsDataset.reference_csv.data}
                            onSelect={this.setReferenceFolderIDColumn}
                            selectedLabel={"Folder Name"}
                            selectedColIndex={this.props.bidsDataset.bids_ref.ref.index}
                        />
                        <BidsSubjectInformation subjects={this.props.bidsDataset.bids_ref.subjects} />
                    </Step>
                ) : null }

                {this.props.bidsDataset.bids_ref.ref.name != null ? (
                    <Step
                        key={4}
                        step={4}
                        desc={'Please enter following information'}
                    >
                        <DatasetMetadata onMetadataChange={this.setBIDSDatasetMetadata}/>
                    </Step>
                ) : null }
                {this.props.metadata.name && this.props.metadata.tags && this.props.metadata.desc ? (
                    <Step
                        key={5}
                        step={5}
                        label="Add dataset"
                    >
                         <ButtonsWrapper>
                            <Button onClick={this.addDataset}>Add Dataset</Button>
                        </ButtonsWrapper>
                    </Step>
                ): null}
            </div>
        );
    }
}


/**
 * Map global bidsDataset of global state to local props.
 * @param state
 * @returns {{bidsDataset: ((function(*=, *): ({identifiers, format: null, folder_path: null} |
 *           {identifiers: {}, format: null, folder_path: null} |
 *           {identifiers: {}, format: null, folder_path}))|*)}}
 */
const mapStateToProps = (state) => {
    return {
        metadata : state.bidsDataset.metadata,
        bids_root : state.bidsDataset.bids_root,
        bidsDataset : state.bidsDataset
    }
}

/**
 * Dispatch actions to props
 * @param dispatch
 * @returns {{setFolderPath: (function(*): *)}}
 */
const mapDispatchToProps = (dispatch) => {
    return {
        setFolderPath : (data) => dispatch(setFolderPath(data)),
        setReferenceCSV : (data) => dispatch(setReferenceCSV(data)),
        setFolderRefColumn : (data) => dispatch(setFolderRefColumn(data)),
        setBIDSDatasetMetadata : (data) => dispatch(setBIDSDatasetMetadata(data)),
        addBIDSDataset : (navigate) => dispatch(addBIDSDataset(navigate))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(withRouter(BidsStandard));

