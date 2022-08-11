import React from 'react';
import CreatableSelect from 'react-select/creatable';
import {connect} from "react-redux"
import {SelectiveTable} from "../../components/common/Tables";
import Button from "../../components/common/Button";
import {CheckBox} from "../../components/common/Inputs";
import {useNavigate, useParams, useLocation} from "react-router-dom";
import styles from "./AddDataset.module.css"

import {
    setCreateModalitiesToFoldersPipeline,
    createModalitiesToFoldersPipeline,
    getDefaultModalityNames,
    updateModalitiesMapping,
    clearModalityMapping,
    } from "../../store/actions/medicalFolderDatasetActions"

import {
    setUsePreExistingDlp,
    setDLPIndex,
    } from "../../store/actions/dataLoadingPlanActions"


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


export class ModalitiesToFolders extends React.Component {
    componentDidMount(){
        this.props.getDefaultModalityNames()
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

    createModalitiesToFoldersPipeline = (event) => {
        // now need to invert the modalities_mapping to obtain a mapping of the form:
        // { modality_name : [folder_1, folder_2, ...] }
        let mod2fol = {}
        let mapping = this.props.modalities_mapping
        for(var key in mapping) {
            if(mapping[key] in mod2fol) {
                mod2fol[mapping[key]].push(key)
            } else {
                mod2fol[mapping[key]] = [key]
            }
        }
        this.props.createModalitiesToFoldersPipeline(mod2fol)
    }

    render() {
        return (
            <React.Fragment>
            { !this.props.use_new_dlp ?
                <CheckBox onChange={(status) => {this.props.usePreExistingDlp(status)}}
                checked={this.props.use_preexisting_dlp}
                >
                    Use an existing already-existing set of customizations.
                    The way your data is loaded and presented to the researcher during the federated
                    training phase will be affected by all the customizations in the plan selected below.
                    For example, check this box if you wish to map your local folder names
                    to more generic imaging modality names.
                </CheckBox> : null
            }
            { this.props.use_preexisting_dlp && this.props.existing_dlps !== null ?
                <React.Fragment>
                <p className={styles.dlp_selection_table_title}>Please select one customization from the table below.</p>
                <SelectiveTable
                    maxHeight={350}
                    table={this.props.existing_dlps}
                    hoverColumns={false}
                    onSelect={this.props.setDLPTableSelectedRow}
                    selectedRowIndex={this.props.selected_dlp_index}
                />
                </React.Fragment> : null
            }
            {!this.props.use_preexisting_dlp ?
                <React.Fragment>
                    <CheckBox onChange={(event) => {this.props.setCreateModalitiesToFoldersPipeline(event)}} >
                        Create a new customized association between imaging modality names and folder names
                        in your local file system.
                    </CheckBox>
                    { this.props.use_new_mod2fol_association ? (
                        <React.Fragment>
                            <div className={styles.dlp_modalities_container}>
                                {this.props.modalities.map((item, key) => {
                                    return(
                                        <React.Fragment key={`modfrag-${key}`}>
                                            <span className={styles.dlp_modalities} key={`modspan-${key}`}>{item}</span>
                                            <div className={styles.dlp_modality_selector} key={`modsel-${key}`}>
                                                <CreatableSelect
                                                    isClearable
                                                    onChange={event => {this.updateModalitiesMapping(event, item)}}
                                                    options={this.props.default_modality_names}
                                                    key={`modcreatsel-${key}`}
                                                />
                                            </div>
                                        </React.Fragment>
                                )})}
                            </div>
                            <div className={styles.saveAssociationsButtonContainer}>
                            <Button
                                onClick={(event) => {this.createModalitiesToFoldersPipeline(event)}}
                            >
                                Save association
                            </Button>
                            <button className={styles.helpCircle}>?</button>
                                <span className={styles.saveAssociationHelpText} >
                                    The last-saved association will be automatically linked to this dataset at the final
                                    step of this page. To fix a mistake, simply click the button again to save a new
                                    association and overwrite the previous one.
                                </span>
                            </div>
                        </React.Fragment>
                ) : null
                }
                </React.Fragment> : null
            }
            </React.Fragment>
        )
    }
}


const mapStateToProps = (state) => {
    return {
        modalities  : state.medicalFolderDataset.modalities,
        use_preexisting_dlp  : state.dataLoadingPlan.use_preexisting_dlp,
        use_new_mod2fol_association  : state.medicalFolderDataset.use_new_mod2fol_association,
        existing_dlps  : state.dataLoadingPlan.existing_dlps,
        default_modality_names : state.medicalFolderDataset.default_modality_names,
        modalities_mapping : state.medicalFolderDataset.modalities_mapping,
        dlp_loading_blocks : state.dataLoadingPlan.dlp_loading_blocks,
        selected_dlp_index : state.dataLoadingPlan.selected_dlp_index,
    }
}

/**
 * Dispatch actions to props
 * @param dispatch
 * @returns {{setFolderPath: (function(*): *)}}
 */
const mapDispatchToProps = (dispatch) => {
    return {
        setDLPTableSelectedRow : (data) => dispatch(setDLPIndex(data)),
        usePreExistingDlp : (data) => dispatch(setUsePreExistingDlp(data)),
        setCreateModalitiesToFoldersPipeline : (data) => dispatch(setCreateModalitiesToFoldersPipeline(data)),
        createModalitiesToFoldersPipeline : (data) => dispatch(createModalitiesToFoldersPipeline(data)),
        getDefaultModalityNames : () => dispatch(getDefaultModalityNames()),
        updateModalitiesMapping : (data) => dispatch(updateModalitiesMapping(data)),
        clearModalityMapping : (data) => dispatch(clearModalityMapping(data)),
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(withRouter(ModalitiesToFolders));

/*
*/