import React from 'react';
import styles from "./AddDataset.module.css"
import Step from "../../components/layout/Step"
import {Label} from "../../components/common/Inputs";
import Button from "../../components/common/Button";


const BidsStandard = (props) => {


    return (
        <div className={styles.main}>
            <Step
                step={1}
                desc={'Please select the root folder that contains BIDS Nifti format brain images. '}
            >
               <div className={`form-control`}>
                    <div className={"repository-select"}>
                        <Button>Select Data File</Button>
                        <div className={`path`}>
                        </div>
                    </div>
                </div>
            </Step>
        </div>
    );
};

export default BidsStandard;