import axios from "axios";
import {
    EP_REPOSITORY_LIST,
    EP_LOAD_CSV_DATA,
    EP_VALIDATE_SUBJECTS_ALL_MODALITIES,
    EP_VALIDATE_MEDICAL_FOLDER_ROOT,
    EP_VALIDATE_REFERENCE_COLUMN,
    EP_READ_DATA_LOADING_PLAN,
    EP_ADD_DATA_LOADING_PLAN,
    EP_DELETE_DATA_LOADING_PLAN,
    EP_ADD_MEDICAL_FOLDER_DATASET,
    EP_PREVIEW_MEDICAL_FOLDER_DATASET,
    EP_DEFAULT_MODALITY_NAMES,
} from "../../constants";
import {displayError} from "./actions";

/**
 * Sets Folder Path
 * @param path
 * @returns {(function(*): void)|*}
 */
export const setFolderPath = (path) => {
    return (dispatch) => {
        if(path.type !== "dir"){
            dispatch({type: 'ERROR_MODAL' , payload: "ROOT path for MedicalFolder dataset should be folder/directory"})
            return
        }
        dispatch({type:'SET_LOADING', payload: {status: true, launcher: "VALIDATE_ROOT", text: "Setting/validating MedicalFolder root folder path"}})
        axios.post(EP_VALIDATE_MEDICAL_FOLDER_ROOT, {medical_folder_root : path.path})
            .then(response => {
                let data = response.data.result
                if(data.valid){
                    dispatch({type:'RESET_MEDICAL_FOLDER'})
                    dispatch({type:'RESET_DLP'})
                    setTimeout(() => {
                        dispatch({type: "SET_MEDICAL_FOLDER_ROOT", payload: { root_path: path.path, modality_folders: data.modalities}})
                        dispatch(checkSubDirectories(path.path))
                        dispatch({type:'SET_LOADING', payload: {status: false, launcher: "VALIDATE_ROOT"}})
                    }, 200)

                }else{
                    dispatch({type:'RESET_MEDICAL_FOLDER'})
                    dispatch({type:'RESET_DLP'})
                    dispatch({type: 'ERROR_MODAL', payload: data.message})
                    dispatch({type:'SET_LOADING', payload: {status: false, launcher: "VALIDATE_ROOT"}})
                }
            }).catch(error => {
                dispatch({type:'SET_LOADING', payload: {status: false, launcher: "VALIDATE_ROOT"}})
                dispatch({type:'RESET_MEDICAL_FOLDER'})
                dispatch({type:'RESET_DLP'})
                dispatch(displayError(error))
        })



    }
}

/**
 * Set reference column that corresponds patient folders
 * @param ref
 * @returns {(function(*): void)|*}
 */
export const setFolderRefColumn = (ref) => {
    return (dispatch, getState) => {
        // TODO: Validate selected column corresponds patient folders
        let state = getState()
        let reference_csv = state.medicalFolderDataset.reference_csv.path
        let root = state.medicalFolderDataset.medical_folder_root
        let validation_data =  {
            reference_csv_path: reference_csv,
            medical_folder_root: root,
            index_col: ref.index
        }

        dispatch({type:'SET_LOADING', payload: {status: true, text: "Setting/validating MedicalFolder subject reference column..."}})
        axios.post(EP_VALIDATE_REFERENCE_COLUMN, validation_data).then(response => {
            let data = response.data.result
            if (data.valid) {
                let payload = {
                    ref: ref,
                    subjects: { available_subjects: data.subjects.intersection,
                                missing_entries: data.subjects.missing_entries,
                                missing_folders: data.subjects.missing_folders
                    }
                }
                dispatch({type: "SET_MEDICAL_FOLDER_REF", payload: payload})
            }else{
                dispatch({type: "RESET_MEDICAL_FOLDER_REF"})
                dispatch({type: "ERROR_MODAL", payload: data.message})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Can not verify reference column for MedicalFolder dataset. Error message is: "))
        })
    }
}

/**
 * Sets reference csv file for MedicalFolder
 * @param path
 * @returns {(function(*): void)|*}
 */
export const setReferenceCSV = (path) => {
    return (dispatch) => {

        dispatch({type:'SET_LOADING', payload: {status: true, text: "Setting reference/demographics dataset for MedicalFolder..."}})
        axios.post(EP_LOAD_CSV_DATA, {path : path.path}).then( response => {
            if(response.status === 200){
                let data = response.data.result
                dispatch({ type: "RESET_MEDICAL_FOLDER_REF"})
                dispatch({type: "SET_REFERENCE_CSV", payload: { path: path.path, data: data}})
            }else{
                dispatch({type: 'ERROR_MODAL', payload: response.data.result.message})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while validating reference CSV file"))
        })
        return
    }
}

/**
 * Sets MedicalFolder dataset meta information dataset name, tags etc.
 * @param data
 * @returns {(function(*))|*}
 */
export const setMedicalFolderDatasetMetadata = (data) => {
    return (dispatch) => {
        dispatch({type: "SET_MEDICAL_FOLDER_METADATA", payload: data})
    }
}

export const setIgnoreReferenceCsv = (data) => {
    return (dispatch) => {
        dispatch({type: "SET_IGNORE_REFERENCE_CSV", payload: data})
    }
}

export const setCustomizeModalitiesToFolders = (value) => {
    return (dispatch) => {
        dispatch({type: 'SET_CUSTOMIZE_MOD2FOL', payload: (value.target.value === 'true' ? true : false)})
    }
}

export const initModalityNames = () => {
    return (dispatch, getState) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Loading default modality names."}})
        axios.get(EP_DEFAULT_MODALITY_NAMES).then(response => {
            dispatch({type: 'SET_DEFAULT_MODALITY_NAMES', payload: response.data.result.default_modalities})
            if(getState().medicalFolderDataset.current_modality_names.length === 0) {
                dispatch({
                    type: 'SET_CURRENT_MODALITY_NAMES',
                    payload: JSON.parse(JSON.stringify(response.data.result.default_modalities))})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while getting default modality names."))
        })
    }
}

function isM2fEqual(m2f, m2f_init) {
    if(Object.keys(m2f).length !== Object.keys(m2f_init).length) {
        return false
    }

    for(const modality in m2f) {
        if(modality in m2f_init && m2f[modality].length === m2f_init[modality].length) {
            for(const folder of m2f[modality]) {
                if(!m2f_init[modality].includes(folder)) {
                    return false
                }
            }
        } else {
            return false
        }
    }
    return true
}

function isSameDlpContent(medical_folder, dlp) {
    let m2f = medical_folder.mod2fol_mapping
    let m2f_init = dlp.preexisting_dlp.mod2fol_mapping
    let dlp_is_same = true

    // can't be same DLP as loaded when not using a loaded DLP
    if(!dlp.use_preexisting_dlp) {
        return false
    }

    if(medical_folder.use_custom_mod2fol && dlp.preexisting_dlp.mod2fol_mapping !== null) {
        dlp_is_same = isM2fEqual(m2f, m2f_init)
    } else if (medical_folder.use_custom_mod2fol !== (dlp.preexisting_dlp.mod2fol_mapping !== null)) {
        // not the same DLP if only one uses custom mod2fol
        dlp_is_same = false
    }

    // add here other comparisons to do when extending the DLP content 
    // (eg: check if CSV file is used & is the same)
    return dlp_is_same
}

export const updateModalitiesMapping = (data) => {
    return (dispatch, getState) => {
        dispatch({type: 'UPDATE_MODALITIES_MAPPING', payload: {folder_name: data.folder_name, modality_name: data.modality_name} })
        
        let state = getState()
        let medical_folder = state.medicalFolderDataset
        let mapping = medical_folder.modalities_mapping
        let m2f = medical_folder.mod2fol_mapping
        for(const modality in m2f) {
            if(m2f[modality].includes(data.folder_name)){
                m2f[modality] = m2f[modality].filter(folder => folder !== data.folder_name)
                if(m2f[modality].length === 0){
                    delete m2f[modality]
                }
                break
            }
        }
        if(data.modality_name in m2f) {
            if(!m2f[data.modality_name].includes(data.folder_name)){
                m2f[data.modality_name].push(data.folder_name)
            }
        } else {
            m2f[data.modality_name] = [data.folder_name]
        }
        dispatch({type: 'UPDATE_MOD2FOL_MAPPING', payload: m2f })

        let dlp_is_same = isSameDlpContent(medical_folder, state.dataLoadingPlan)
        dispatch({type: 'SET_DLP_SAME_AS_PREEXISTING', payload: dlp_is_same})

        let has_all_mappings = true
        for(const folder of medical_folder.modality_folders.values()) {
            if(!mapping[folder]) {
                has_all_mappings = false
                break
            }
        }
        dispatch({type: 'UPDATE_HAS_ALL_MAPPINGS', payload: has_all_mappings})

        let current_modalities = getState().medicalFolderDataset.current_modality_names
        let has_modality = false
        for(const mod of current_modalities.values()) {
            if(data.modality_name === mod['label']){
                has_modality = true
            }
        }
        if(!has_modality) {
            current_modalities.push({'label': data.modality_name, 'value': data.modality_name})
        }
        dispatch({type: 'SET_CURRENT_MODALITY_NAMES', payload: current_modalities})
    }
}

export const clearModalityMapping = (folder_name) => {
    return (dispatch, getState) => {
        dispatch({type: 'CLEAR_MODALITY_MAPPING', payload: folder_name})

        let state = getState()
        let medical_folder = state.medicalFolderDataset
        let m2f = medical_folder.mod2fol_mapping
        for(const modality in m2f) {
            if(m2f[modality].includes(folder_name)){
                m2f[modality] = m2f[modality].filter(folder => folder !== folder_name)
                if(m2f[modality].length === 0){
                    delete m2f[modality]
                }
                break
            }
        }
        dispatch({type: 'UPDATE_MOD2FOL_MAPPING', payload: m2f})

        let dlp_is_same = isSameDlpContent(medical_folder, state.dataLoadingPlan)
        dispatch({type: 'SET_DLP_SAME_AS_PREEXISTING', payload: dlp_is_same})

        dispatch({type: 'UPDATE_HAS_ALL_MAPPINGS', payload: false})
    }
}


export const setChangeDlpMedicalFolderDataset = (use_dlp, state) => {
    return (dispatch) => {
        if(!use_dlp || state.dataLoadingPlan.selected_dlp_index === null) {
            dispatch({
                type: "RESET_MEDICAL_CHANGE_USED_DLP",
                payload: JSON.parse(JSON.stringify(state.medicalFolderDataset.default_modality_names))})
            dispatch({type: 'RESET_DLP_PREEXISTING'})
        } else {
            let dlp_id = state.dataLoadingPlan.existing_dlps.data[state.dataLoadingPlan.selected_dlp_index][1]

            dispatch({type:'SET_LOADING', payload: {status: true, text: "Reading Data Loading Plan content..."}})
            axios.post(EP_READ_DATA_LOADING_PLAN, {'dlp_id': dlp_id}).then(response => {
                dispatch({type:'SET_LOADING', payload: {status: false}})
                let data = response.data.result
                // default value if not set by DLP
                let dlp = {
                    use_custom_mod2fol: false,
                    current_modality_names: JSON.parse(JSON.stringify(state.medicalFolderDataset.default_modality_names)), // careful, not null
                    modalities_mapping: {}, // careful, not the initial null
                    mod2fol_mapping: {}, // careful, not the initial nul
                    has_all_mappings: false,
                    reference_csv: null,
                    ignore_reference_csv: false,
                }

                // DLP contains modality mapping
                if('map' in data) {
                    dlp['use_custom_mod2fol'] = true
                    dlp['mod2fol_mapping'] = JSON.parse(JSON.stringify(data.map))

                    // current_modality_names
                    for(const mod in data['map']) {
                        let found = false
                        for(const curmod of dlp['current_modality_names']) {
                            if(curmod['value'] === mod) {
                                found = true
                                break
                            }
                        }
                        if(!found) {
                            dlp['current_modality_names'].push({'label': mod, 'value': mod})
                        }
                    }

                    // modalities_mapping
                    for(const mod in data['map']) {
                        for(const folder of data['map'][mod]) {
                            if(state.medicalFolderDataset.modality_folders.includes(folder)) {
                                dlp['modalities_mapping'][folder] = mod
                            } else {
                            }
                            // ignore mappings that dont correspond to a folder in this dataset
                        }
                    }

                    // has_all_mappings
                    let has_all_mappings = true
                    for(const folder of state.medicalFolderDataset.modality_folders.values()) {
                        if(!dlp['modalities_mapping'][folder]) {
                            has_all_mappings = false
                            break
                        }
                    }
                    dlp['has_all_mappings'] = has_all_mappings
                }
                dispatch({type: "SET_MEDICAL_CHANGE_USED_DLP", payload: dlp})
                dispatch({type: 'SET_DLP_PREEXISTING', payload: {
                    dlp_id: dlp_id,
                    mod2fol_mapping: JSON.parse(JSON.stringify(data.map)),
                }})
                // dirty hack: need to force refresh of the ModalitiesToFolders and CSV selection
                if(dlp['use_custom_mod2fol'] === true) {
                    dispatch({type: 'SET_CUSTOMIZE_MOD2FOL', payload: false})
                    setTimeout(() => {
                        dispatch({type: 'SET_CUSTOMIZE_MOD2FOL', payload: true})
                    }, 200)
                }
                if(dlp['has_all_mappings']){
                    dispatch({type: 'UPDATE_HAS_ALL_MAPPINGS', payload: false})
                    setTimeout(() => {
                        dispatch({type: 'UPDATE_HAS_ALL_MAPPINGS', payload: true})
                    }, 400)
                }
                //end of dirty hack to refresh
            }).catch(error => {
                dispatch({type:'SET_LOADING', payload: {status: false}})
                dispatch(displayError(error, "Error while reading Data Loading Plan content."))
            })
        }
    }
}


function checkSubjectsAllModalities(dispatch, mf, dlp_id) {
    return new Promise((resolve, reject) =>
    { 
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Checking some data folders have all modalities..."}})
        let params_check = {
            'medical_folder_root': mf.medical_folder_root,
            'reference_csv_path': (!mf.ignore_reference_csv && mf.reference_csv ? mf.reference_csv.path : null),
            'index_col' : (!mf.ignore_reference_csv && mf.medical_folder_ref ? mf.medical_folder_ref.ref.index : null),
            'dlp_id': dlp_id,
        }
        if(mf.use_custom_mod2fol) {
            params_check['modalities'] = Object.keys(mf.mod2fol_mapping)
        } else {
            params_check['modalities'] = mf.modality_folders
        }
        axios.post(EP_VALIDATE_SUBJECTS_ALL_MODALITIES, params_check).then( response => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            if(response.status === 200){
                let data = response.data.result
                resolve(data.subjects.length > 0)
            } else {
                dispatch({type: 'ERROR_MODAL', payload: response.data.result.message})
                reject()
            }
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while checking some folders have all modalities: "))
            reject()
        })
    })
}

/**
 * Sends Medical Folder dataset add request and validate result
 * @returns {(function(*))|*}
 */
export const addMedicalFolderDataset = (navigator) => {
    return (dispatch, getState) => {
        let state = getState()
        let medical_folder = state.medicalFolderDataset
        let dlp = state.dataLoadingPlan
        let dlp_function = null

        let used_dlp_id = null
        // reusing an unchanged loaded DLP
        if(medical_folder.use_custom_mod2fol && (dlp.use_preexisting_dlp && dlp.same_as_preexisting_dlp)) {
            used_dlp_id = dlp.preexisting_dlp.dlp_id
        }

        // only try to save DLP when we have customizations, and not the same as loaded customizations
        if(medical_folder.use_custom_mod2fol && (!dlp.use_preexisting_dlp || !dlp.same_as_preexisting_dlp)) {
            dispatch({type:'SET_LOADING', payload: {status: true, text: "Saving data customizations..."}})
            let params_add_dlp = {
                'name': dlp.dlp_name
            }
            if(medical_folder.use_custom_mod2fol) {
                params_add_dlp = {
                    ...params_add_dlp,
                    'modalities_mapping': medical_folder.mod2fol_mapping
                }
            }
            dlp_function = axios.post(EP_ADD_DATA_LOADING_PLAN, params_add_dlp)
        } else {
            dlp_function = new Promise((resolve) => { resolve('DUMMY') })
        }

        // sequentially save DLP with next actions
        dlp_function.then( response => {
            if(response === 'DUMMY' || response.status === 200) {
                let saved_dlp_id = null
                if(response !== 'DUMMY'){ 
                    dispatch({type:'SET_LOADING', payload: {status: false}})
                    saved_dlp_id = response.data.result
                    used_dlp_id = response.data.result
                }

                checkSubjectsAllModalities(dispatch, medical_folder, used_dlp_id).then(
                (resolve) => {
                    if(resolve) {
                        let data = {
                            medical_folder_root : medical_folder.medical_folder_root,
                            name : medical_folder.metadata.name,
                            desc : medical_folder.metadata.desc,
                            tags : medical_folder.metadata.tags,
                            dlp_id: used_dlp_id
                        }
                        if(!medical_folder.ignore_reference_csv){
                            data = {
                                ...data,
                                index_col: medical_folder.medical_folder_ref.ref.index,
                                reference_csv_path: medical_folder.reference_csv ? medical_folder.reference_csv.path : null,
                            }
                        }

                        dispatch({type:'SET_LOADING', payload: {status: true, text: "Adding MedicalFolder dataset and validating all the inputs..."}})
                        axios.post(EP_ADD_MEDICAL_FOLDER_DATASET, data).then( response => {
                                dispatch({type: 'SUCCESS_MODAL' , payload: "Dataset has been successfully added"})
                                dispatch({type:'SET_LOADING', payload: {status: false}})
                                navigator('/datasets')
                                dispatch({type:'RESET_MEDICAL_FOLDER'})
                                dispatch({type: 'RESET_DLP'})
                        }).catch(error => {
                            cleanDLP(dispatch, saved_dlp_id)
                            dispatch({type:'SET_LOADING', payload: {status: false}})
                            dispatch(displayError(error, "Error while adding MedicalFolder dataset: "))
                        })
                    } else {
                        cleanDLP(dispatch, saved_dlp_id)
                        dispatch({'type': 'ERROR_MODAL', payload: 'No subject from the dataset has one folder ' +
                            'for each defined modalities, or some subjects have more than one folder for some ' +
                            'modality. Check and update your customized associations'})
                    }
                },
                (reject) => {
                    cleanDLP(dispatch, saved_dlp_id)
                })
            } else {
                dispatch({type: 'ERROR_MODAL' , payload: response.data.result.message})
            }
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while saving data customizations: "))
        })
    }
}

function cleanDLP(dispatch, dlp_id) {
    if(dlp_id !== null) {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Cleaning data loading plan customizations..."}})
        axios.post(EP_DELETE_DATA_LOADING_PLAN, {dlp_id: dlp_id}).then( response => {
                dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            // dont add messages, this is already an error case
        })
    }
}

/**
 *
 * @param dataset_id
 * @returns {(function(*))|*}
 */
export const getMedicalFolderPreview = (dataset_id) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Loading MedicalFolder dataset subject table..."}})
        axios.post(EP_PREVIEW_MEDICAL_FOLDER_DATASET, {dataset_id: dataset_id}).then(response => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch({type: "SET_MEDICAL_FOLDER_PREVIEW", payload : {...response.data.result, dataset_id : dataset_id}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while getting preview of MedicalFolder dataset."))
        })
    }
}

/*
 * API call to check subdirectories in MedicalFolderDataset root folder to verify they can be parsed
 * @param path
 * @returns {(function(*): void)|*}
 */
const checkSubDirectories = (path) => {
    return (dispatch) => {
        dispatch({type:'SET_LOADING', payload: {status: true, text: "Checking sub directories for MedicalFolder..."}})
        axios.post(EP_REPOSITORY_LIST, {path: path}).then(response => {
            if(response.status !== 200){
                dispatch({type: 'ERROR_MODAL' , payload: response.data.result.message})
            }
            dispatch({type:'SET_LOADING', payload: {status: false}})
        }).catch(error => {
            dispatch({type:'SET_LOADING', payload: {status: false}})
            dispatch(displayError(error, "Error while checking sub-directories of root MedicalFolderDataset folder."))
        })
    }
}

