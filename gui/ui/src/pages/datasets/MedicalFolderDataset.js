import React from 'react';
import CreatableSelect from 'react-select/creatable';
import styles from "./AddDataset.module.css"
import Step from "../../components/layout/Step"
import {connect} from "react-redux"
import FileBrowser from "../../components/common/FileBrowser";
import {setFolderPath,
    setFolderRefColumn,
    setReferenceCSV,
    addMedicalFolderDataset,
    setIgnoreReferenceCsv,
    setUsePreExistingDlp,
    setDLP,
    setCreateNewDlp,
    getDefaultModalityNames,
    updateModalitiesMapping,
    clearModalityMapping,
    saveDlp,
    } from "../../store/actions/medicalFolderDatasetActions"
import {SelectiveTable} from "../../components/common/Tables";
import MedicalFolderSubjectInformation from "./MedicalFolderSubjectInformation";
import Button, {ButtonsWrapper} from "../../components/common/Button";
import {useNavigate, useParams, useLocation} from "react-router-dom";
import DatasetMetadata from "./MedicalFolderMetaData";
import {CheckBox} from "../../components/common/Inputs";


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


export class MedicalFolderDataset extends React.Component {
    componentDidMount(){
        this.getDefaultModalityNames()
    }

    setDataPath = (path) => {
        this.props.setFolderPath(path)
    }

    setReferenceCSV = (path) => {
        if (this.props.medicalFolderDataset.reference_csv) {
            this.props.setFolderRefColumn({name: null, index: null})
        }
        this.props.setReferenceCSV(path)
    }

    setReferenceFolderIDColumn = (index) => {
        this.props.setFolderRefColumn({
            index: index,
            name: this.props.medicalFolderDataset.reference_csv.data.columns[index]
        })
    }

    setDLPTableSelectedRow = (index) => {
        console.log('Selected DLP ', index)
        this.props.setDLP(index)
    }

    addDataset = () => {
        this.props.addMedicalFolderDataset(this.props.router.navigate)
    }

    ignoreReferenceCsv = (status) => {
        this.props.ignoreReferenceCsv(status)
    }

    getDefaultModalityNames = () => {
        this.props.getDefaultModalityNames()
    }

    usePreExistingDlp = (status) => {
        this.props.usePreExistingDlp(status)
    }

    saveDlp = async (event) => {
        // now need to invert the modalities_mapping to obtain a mapping of the form:
        // { modality_name : [folder_1, folder_2, ...] }
        let dlp = {}
        let mapping = this.props.modalities_mapping
        for(var key in mapping) {
            if(mapping[key] in dlp) {
                dlp[mapping[key]].push(key)
            } else {
                dlp[mapping[key]] = [key]
            }
        }
        this.props.saveDlp(dlp)

    }

    updateModalitiesMapping = (data, folder_name) => {
        if(data === null) {
            this.props.clearModalityMapping(folder_name)
        } else {
            data.modality_name = data.value
            data.folder_name = folder_name
            this.props.updateModalitiesMapping(data)
        }
    }

    render() {
        return (
            <div className={styles.main}>
                <Step key={1}
                      step={1}
                      desc={'Please select the root folder of MedicalFolder dataset.'}
                >
                   <FileBrowser
                        folderPath = {this.props.medical_folder_root ? this.props.medical_folder_root : null}
                        onSelect = {this.setDataPath}
                        buttonText = "Select Folder"
                        onlyFolders={true}
                   />
                    {this.props.medicalFolderDataset.modalities ?
                        (<div className={''}>
                            <label>Modalities: </label>
                            {this.props.medicalFolderDataset.modalities.map((item, key) => {
                                  return(
                                      <span className={styles.modalities} key={key}>{item}</span>
                                  )
                            })}
                        </div>) : null
                    }
                </Step>

                {this.props.medical_folder_root ?(
                    <React.Fragment>
                    <Step key={2}
                          step={2}
                          desc={'Would you like to use an existing DLP?'}
                    >
                           <CheckBox onChange={this.usePreExistingDlp}
                                     checked={this.props.use_preexisting_dlp}>
                                     Use an existing Data Loading Plan. A Data Loading Plan is a set of customizations to
                                     the way your data will be loaded and presented to the researcher during the federated
                                     training phase. For example, check this box if you wish to map your local folder names
                                     to more generic imaging modality names.
                           </CheckBox>
                            { !this.props.use_preexisting_dlp || this.props.existing_dlps === null ? null :
                            <SelectiveTable
                                maxHeight={350}
                                table={this.props.existing_dlps}
                                selectedLabel={"Folder Name"}
                                hoverColumns={false}
                                onSelect={this.props.setDLPTableSelectedRow}
                            />
                             }
                            { this.props.use_preexisting_dlp ? null :
                                <React.Fragment>
                                <CheckBox onChange={() => {console.log('Create new!')}}>
                                    Create a new Data Loading Plan.
                                </CheckBox>
                                  <form onSubmit={ (event) => {event.preventDefault(); this.saveDlp(event)}}>
                                  <div className={styles.dlp_modalities_container}>
                                  {this.props.medicalFolderDataset.modalities.map((item, key) => {
                                        return(
                                        <React.Fragment>
                                            <span className={styles.dlp_modalities} key={key}>{item}</span>
                                            <div className={styles.dlp_modality_selector}>
                                            <CreatableSelect
                                                isClearable
                                                onChange={event => {this.updateModalitiesMapping(event, item)}}
                                                options={this.props.default_modality_names}
                                            />
                                            </div>
                                        </React.Fragment>
                                        )})}
                                  </div>
                                  <button type='Submit'>Save Data Loading Plan</button>
                                  </form>
                                </React.Fragment>

                            }
                    </Step>

                    <Step
                        key={3}
                        step={3}
                        desc={'Please select reference/demographics CSV file where all subject folder names are stored'}
                    >
                       <CheckBox onChange={this.ignoreReferenceCsv}
                                 checked={this.props.ignore_reference_csv}>
                           Use only subject folders for MedicalFolder dataset. This option will allow you to loads MedicalFolder dataset
                           without declaring reference/demographics csv.
                       </CheckBox>
                        { !this.props.ignore_reference_csv ? (
                             <FileBrowser
                                folderPath = {this.props.medicalFolderDataset.reference_csv ? this.props.medicalFolderDataset.reference_csv.path : null}
                                onSelect = {this.setReferenceCSV}
                                onlyExtensions = {[".csv"]}
                                buttonText = "Select Data File"
                           />
                        ) : null}
                    </Step>
                    </React.Fragment>
                    ) : null
                }

                { !this.props.ignore_reference_csv && this.props.medical_folder_root && this.props.medicalFolderDataset.reference_csv != null ? (
                    <Step
                        key={4}
                        step={4}
                        desc={'Please select to column that represent subject folders in MedicalFolder root directory.'}
                    >
                        <SelectiveTable
                            maxHeight={350}
                            table={this.props.medicalFolderDataset.reference_csv.data}
                            onSelect={this.setReferenceFolderIDColumn}
                            selectedLabel={"Folder Name"}
                            selectedColIndex={this.props.medicalFolderDataset.medical_folder_ref.ref.index}
                        />
                        <MedicalFolderSubjectInformation subjects={this.props.medicalFolderDataset.medical_folder_ref.subjects} />
                    </Step>
                ) : null }

                {this.props.medicalFolderDataset.medical_folder_ref.ref.name != null || this.props.ignore_reference_csv ? (
                    <Step
                        key={5}
                        step={5}
                        desc={'Please enter following information'}
                    >
                        <DatasetMetadata/>
                    </Step>
                ) : null }
                {(this.props.metadata.name && this.props.metadata.tags && this.props.metadata.desc) &&
                    ((!this.props.ignore_reference_csv && this.props.medicalFolderDataset.medical_folder_ref.ref.name ) ||
                      this.props.ignore_reference_csv
                    )? (
                    <Step
                        key={6}
                        step={6}
                        label="Add/Register MedicalFolder Dataset"
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
 * Map global medicalFolderDataset of global state to local props.
 * @param state
 * @returns {{medicalFolderDataset: ((function(*=, *): ({identifiers, format: null, folder_path: null} |
 *           {identifiers: {}, format: null, folder_path: null} |
 *           {identifiers: {}, format: null, folder_path}))|*)}}
 */
const mapStateToProps = (state) => {
    return {
        metadata : state.medicalFolderDataset.metadata,
        medical_folder_root : state.medicalFolderDataset.medical_folder_root,
        medicalFolderDataset : state.medicalFolderDataset,
        ignore_reference_csv : state.medicalFolderDataset.ignore_reference_csv,
        use_preexisting_dlp  : state.medicalFolderDataset.use_preexisting_dlp,
        existing_dlps  : state.medicalFolderDataset.existing_dlps,
        default_modality_names : state.medicalFolderDataset.default_modality_names,
        modalities_mapping : state.medicalFolderDataset.modalities_mapping
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
        setDLPTableSelectedRow : (data) => dispatch(setDLP(data)),
        addMedicalFolderDataset : (navigate) => dispatch(addMedicalFolderDataset(navigate)),
        ignoreReferenceCsv : (data) => dispatch(setIgnoreReferenceCsv(data)),
        usePreExistingDlp : (data) => dispatch(setUsePreExistingDlp(data)),
        createNewDlp : (data) => dispatch(setCreateNewDlp(data)),
        getDefaultModalityNames : () => dispatch(getDefaultModalityNames()),
        updateModalitiesMapping : (data) => dispatch(updateModalitiesMapping(data)),
        clearModalityMapping : (data) => dispatch(clearModalityMapping(data)),
        saveDlp : (data) => dispatch(saveDlp(data)),
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(withRouter(MedicalFolderDataset));

