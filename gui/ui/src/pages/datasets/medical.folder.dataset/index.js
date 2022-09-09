import React from 'react';
import {useNavigate, useParams, useLocation} from "react-router-dom";
import {connect} from "react-redux"
import {setFolderPath,
    setFolderRefColumn,
    setReferenceCSV,
    addMedicalFolderDataset,
    setIgnoreReferenceCsv,
    } from "../../../store/actions/medicalFolderDatasetActions"
import styles from "../AddDataset.module.css"
import FileBrowser from "../../../components/common/FileBrowser";
import {Label} from '../../../components/common/Inputs'
import {SelectiveTable} from "../../../components/common/Tables";
import Button, {ButtonsWrapper} from "../../../components/common/Button";
import Step from "../../../components/layout/Step"
import {CheckBox} from "../../../components/common/Inputs";
import {EuiRadio, EuiSpacer, EuiSelect} from '@elastic/eui';
import MedicalFolderSubjectInformation from "./MedicalFolderSubjectInformation";
import DatasetMetadata from "./MedicalFolderMetaData";
import ModalitiesToFolders from "./ModalitiesToFolders";
import {
    setUsePreExistingDlp,
    setDLPIndex,
    } from "../../../store/actions/dataLoadingPlanActions"

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


export class Index extends React.Component {
    setDataPath = (path) => {
        this.props.setFolderPath(path)
    }

    setReferenceCSV = (path) => {
        this.props.setReferenceCSV(path)
    }

    setReferenceFolderIDColumn = (index) => {
        this.props.setFolderRefColumn({
            index: index,
            name: this.props.medicalFolderDataset.reference_csv.data.columns[index]
        })
    }

    addDataset = () => {
        this.props.addMedicalFolderDataset(this.props.router.navigate)
    }

    ignoreReferenceCsv = (status) => {
        this.props.setIgnoreReferenceCsv(status)
    }

    getDlpSelectOptions = () => {
        let options = [{value: '-1', text: 'Please select...'}]
        for(const index of this.props.existing_dlps.index) {
                options.push({value: index, text: this.props.existing_dlps.data[index][0]})
        }
        return(options)
    }

    getDlpSelectDefault = () => {
        return((this.props.selected_dlp_index === null ? '-1' : this.props.selected_dlp_index))
    }

    render() {
        return (
            <div className={styles.main}>
                <React.Fragment>
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
                    {this.props.medicalFolderDataset.modality_folders ?
                        (<div className={''}>
                            <label>Folder names: </label>
                            {this.props.medicalFolderDataset.modality_folders.map((item, key) => {
                                  return(
                                      <span className={styles.modalities} key={key}>{item}</span>
                                  )
                            })}
                        </div>) : null
                    }
                </Step>
                </React.Fragment>

                {this.props.medical_folder_root ?
                    <React.Fragment>
                    <Step key={2}
                          step={2}
                          desc={'Would you like to duplicate existing customizations for this dataset?'}
                    >
                        Customizations affect the way your data is loaded and presented to the researcher during
                        the federated training: modality to folder name associations, reference CSV name and column.
                        Choose if you wish to reuse previously defined customizations.

                        <EuiSpacer size="m" />
                        <EuiRadio
                            id="reuse-dlp-false"
                            label="Don't use existing customizations"
                            value="false"
                            checked={!this.props.use_preexisting_dlp}
                            onChange={this.props.usePreExistingDlp}
                        />
                        <EuiRadio
                            id="reuse-dlp-true"
                            label="Use, duplicate and edit an existing set of customizations"
                            value="true"
                            checked={this.props.use_preexisting_dlp}
                            onChange={this.props.usePreExistingDlp}
                        />
                        {this.props.use_preexisting_dlp && this.props.existing_dlps !== null ?
                            <React.Fragment>
                             <div className="form-control">
                                <Label>Please select one customization set.</Label>
                                <EuiSelect 
                                    options={this.getDlpSelectOptions()}
                                    value={this.getDlpSelectDefault()}
                                    onChange={this.props.setDLPIndex}
                                />
                            </div>
                            </React.Fragment> : null
                        }
                    </Step>
                        {!this.props.use_preexisting_dlp || this.props.selected_dlp_index != null ?
                            <Step key={3}
                                  step={3}
                                  desc={'Would you like to customize association of dataset folder names to imaging modality names?'}
                            >
                                <ModalitiesToFolders />
                            </Step>
                            : null
                         }
                     </React.Fragment>
                 : null
                }

                {this.props.medical_folder_root &&
                    (!this.props.use_preexisting_dlp || (this.props.selected_dlp_index != null)) &&
                    (!this.props.use_custom_mod2fol || this.props.has_all_mappings) ?
                    <React.Fragment>
                    <Step
                        key={4}
                        step={4}
                        desc={'Please select reference/demographics CSV file where all subject folder names are stored'}
                    >
                       <CheckBox onChange={this.ignoreReferenceCsv}
                                 checked={this.props.ignore_reference_csv}>
                           Use only subject folders for MedicalFolder dataset. This option will allow you to loads MedicalFolder dataset
                           without declaring reference/demographics csv.
                       </CheckBox>
                        { !this.props.ignore_reference_csv ? (
                             <FileBrowser
                                folderPath = {this.props.medicalFolderDataset.reference_csv ? this.props.medicalFolderDataset.reference_csv.path : []}
                                onSelect = {this.setReferenceCSV}
                                onlyExtensions = {[".csv"]}
                                buttonText = "Select Data File"
                           />
                        ) : null}
                    </Step>
                    </React.Fragment>
                 : null
                }

                {this.props.medical_folder_root &&
                    (!this.props.use_preexisting_dlp || (this.props.selected_dlp_index != null)) &&
                    (!this.props.use_custom_mod2fol || this.props.has_all_mappings) &&
                    !this.props.ignore_reference_csv && this.props.medicalFolderDataset.reference_csv != null ? (
                    <React.Fragment>
                    <Step
                        key={5}
                        step={5}
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
                    </React.Fragment>
                ) : null }

                {this.props.medical_folder_root &&
                    (!this.props.use_preexisting_dlp || (this.props.selected_dlp_index != null)) &&
                    (!this.props.use_custom_mod2fol || this.props.has_all_mappings) &&
                    (this.props.reference_csv_column != null || this.props.ignore_reference_csv) ? (
                    <React.Fragment>
                    <Step
                        key={6}
                        step={6}
                        desc={'Please enter following information'}
                    >
                        <DatasetMetadata/>
                    </Step>
                    </React.Fragment>
                ) : null }

                {this.props.medical_folder_root &&
                    (!this.props.use_preexisting_dlp || (this.props.selected_dlp_index != null)) &&
                    (!this.props.use_custom_mod2fol || this.props.has_all_mappings) &&
                    (this.props.reference_csv_column != null || this.props.ignore_reference_csv) &&
                    ( this.props.metadata.name && (this.props.metadata.name.length >= 4) &&
                    this.props.metadata.tags && (this.props.metadata.tags.length > 0) &&
                    this.props.metadata.desc && (this.props.metadata.desc.length >= 4)) ? (
                    <React.Fragment>
                    <Step
                        key={7}
                        step={7}
                        label="Add/Register MedicalFolder Dataset"
                    >
                         <ButtonsWrapper>
                            <Button onClick={this.addDataset}>Add Dataset</Button>
                        </ButtonsWrapper>
                    </Step>
                    </React.Fragment>
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
        use_preexisting_dlp  : state.dataLoadingPlan.use_preexisting_dlp,
        selected_dlp_index : state.dataLoadingPlan.selected_dlp_index,
        use_custom_mod2fol : state.medicalFolderDataset.use_custom_mod2fol,
        existing_dlps  : state.dataLoadingPlan.existing_dlps,
        medical_folder_root : state.medicalFolderDataset.medical_folder_root,
        medicalFolderDataset : state.medicalFolderDataset,
        ignore_reference_csv : state.medicalFolderDataset.ignore_reference_csv,
        has_all_mappings : state.medicalFolderDataset.has_all_mappings,
        reference_csv_column : state.medicalFolderDataset.medical_folder_ref.ref.name,
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
        addMedicalFolderDataset : (navigate) => dispatch(addMedicalFolderDataset(navigate)),
        setIgnoreReferenceCsv : (data) => dispatch(setIgnoreReferenceCsv(data)),
        usePreExistingDlp : (data) => dispatch(setUsePreExistingDlp(data)),
        setDLPIndex : (data) => dispatch(setDLPIndex(data)),
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(withRouter(Index));

