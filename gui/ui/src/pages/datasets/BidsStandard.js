import React, {useState} from 'react';
import styles from "./AddDataset.module.css"
import Step from "../../components/layout/Step"
import {connect, useDispatch} from "react-redux"
import FileBrowser from "../../components/common/FileBrowser";
import {setFolderPath, setFolderRefColumn, setReferenceCSV} from "../../store/actions/bidsDatasetActions"
import {SelectiveTable} from "../../components/common/Tables";




const BidsStandard = (props) => {


    const [stepCount, setStepCount] = useState(1)

    const setDataPath = (path) => {
        console.log(path)
        props.setFolderPath(path)
    }

    const setReferenceCSV = (path) => {

        if(props.bidsDataset.reference_csv){
            props.setFolderRefColumn({name: null, index: null})
        }

        props.setReferenceCSV(path)
    }

    const setReferenceFolderIDColumn = (index) => {
        props.setFolderRefColumn({
            index : index,
            name : props.bidsDataset.reference_csv.data.columns[index]
        })
    }

    return (
        <div className={styles.main}>
            <Step key={1}
                  disable={stepCount > 0 ? false : true}
                  step={1}
                  desc={'Please select the root folder that contains BIDS Nifti format brain images. '}
            >
               <FileBrowser
                    folderPath = {props.bidsDataset.data_path ? props.bidsDataset.data_path : null}
                    onSelect = {setDataPath}
               />
            </Step>

            {props.bidsDataset.data_path ?(
                <Step
                    key={2}
                    disable={stepCount > 0 ? false : true}
                    step={2}
                    desc={'Please select reference CSV file where al patient IDs are stored '}
                >
                   <FileBrowser
                        folderPath = {props.bidsDataset.reference_csv ? props.bidsDataset.reference_csv.path : null}
                        onSelect = {setReferenceCSV}
                   />
                </Step>
                ) : null
            }

            {props.bidsDataset.reference_csv != null ? (
                <Step
                    key={3}
                    disable={stepCount > 0 ? false : true}
                    step={3}
                    desc={'Please select to ID column from reference csv'}
                >
                    <SelectiveTable
                        style={{maxHeight:350}}
                        table={props.bidsDataset.reference_csv.data}
                        onSelect={setReferenceFolderIDColumn}
                        selectedLabel={"FolderID"}
                        selectedColIndex={props.bidsDataset.folder_ref_column.index}
                    />
                </Step>
            ) : null }



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
        setReferenceCSV : (data) => dispatch(setReferenceCSV(data)),
        setFolderRefColumn : (data) => dispatch(setFolderRefColumn(data))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(BidsStandard);