import React from 'react';
import CreatableSelect from 'react-select/creatable';
import {connect} from "react-redux"
import {EuiRadio, EuiSpacer} from '@elastic/eui';
import {useNavigate, useParams, useLocation} from "react-router-dom";
import styles from "./AddDataset.module.css"

import {
    setCustomizeModalitiesToFolders,
    initModalityNames,
    updateModalitiesMapping,
    clearModalityMapping,
    } from "../../store/actions/medicalFolderDatasetActions"

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
        this.props.initModalityNames()
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

    getModalityNameFromItem = (modality_tag) => {
        for(const modality of this.props.current_modality_names){
            if(modality.value === modality_tag){
                return(modality)
            }
        }
        return(Object.create({}))
    }

    render() {
        return (
            <React.Fragment>
            The way your data is loaded and presented to the researcher during the federated
            training is affected by associations below.
            Check this box if you wish to define mapping of local folder names
            to more generic imaging modality names.
            
            <EuiSpacer size="m" />
            <EuiRadio
                id="custom-modalities-false"
                label="Use folder names as modality names"
                value="false"
                checked={!this.props.use_custom_mod2fol}
                onChange={this.props.setCustomizeModalitiesToFolders}
            />
            <EuiRadio
                id="custom-modalities-true"
                label="Customize associations between folder names from the dataset and
                    imaging modality names"
                value="true"
                checked={this.props.use_custom_mod2fol}
                onChange={this.props.setCustomizeModalitiesToFolders}
            />
            { this.props.use_custom_mod2fol ? (
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
                                            options={this.props.current_modality_names}
                                            defaultValue={this.getModalityNameFromItem(this.props.modalities_mapping[item])}
                                            key={`modcreatsel-${key}`}
                                        />
                                    </div>
                                </React.Fragment>
                        )})}
                    </div>
                </React.Fragment>
            ) : null
            }

            </React.Fragment>
        )
    }
}


const mapStateToProps = (state) => {
    return {
        modalities  : state.medicalFolderDataset.modalities,
        use_custom_mod2fol  : state.medicalFolderDataset.use_custom_mod2fol,
        default_modality_names : state.medicalFolderDataset.default_modality_names,
        current_modality_names : state.medicalFolderDataset.current_modality_names,
        modalities_mapping : state.medicalFolderDataset.modalities_mapping,
        mod2fol_mapping : state.medicalFolderDataset.mod2fol_mapping,        
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
        setCustomizeModalitiesToFolders : (data) => dispatch(setCustomizeModalitiesToFolders(data)),
        initModalityNames : () => dispatch(initModalityNames()),
        updateModalitiesMapping : (data) => dispatch(updateModalitiesMapping(data)),
        clearModalityMapping : (data) => dispatch(clearModalityMapping(data)),
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(withRouter(ModalitiesToFolders));

/*
*/