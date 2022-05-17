import React, {useState} from 'react';
import styles from "./AddDataset.module.css"
import Step from "../../components/layout/Step"
import {connect, useDispatch} from "react-redux"
import FileBrowser from "../../components/common/FileBrowser";
import {setFolderPath, setReferenceCSV} from "../../store/actions/bidsDatasetActions"




const BidsStandard = (props) => {


    const [stepCount, setStepCount] = useState(1)

    const setDataPath = (path) => {
        console.log(path)
        props.setFolderPath(path)
    }

    const setReferenceCSV = (path) => {
        props.setReferenceCSV(path)
    }

    return (
        <div className={styles.main}>
            <Step key={1}
                  disable={stepCount > 0 ? false : true}
                  step={1}
                  desc={'Please select the root folder that contains BIDS Nifti format brain images. '}
            >
               <FileBrowser
                    folderPath = {props.bidsDataset.data_path ? props.bidsDataset.data_path.path : null}
                    onSelect = {setDataPath}
               />
            </Step>

            <Step
                key={2}
                disable={stepCount > 0 ? false : true}
                step={2}
                desc={'Please select reference CSV file where al patient IDs are stored '}
            >
               <FileBrowser
                    folderPath = {props.bidsDataset.data_path ? props.bidsDataset.data_path.path : null}
                    onSelect = {setReferenceCSV}
               />
            </Step>
        </div>
    );
};


/**
 * Map global bidsDataset of global state to local props.
 * @param state
 * @returns {{bidsDataset: ((function(*=, *): ({identifiers, format: null, folder_path: null} |
 *           {identifiers: {}, format: null, folder_path: null} |
 *           {identifiers: {}, format: null, folder_path}))|*)}}
 */
const mapStateToProps = (state) => {
    return {
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
        setReferenceCSV : (data) => dispatch(setReferenceCSV(data))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(BidsStandard);