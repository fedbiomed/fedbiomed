import React from 'react';
import Tab from "../../components/common/Tab"
import {Outlet} from "react-router-dom";
import styles from "./AddDataset.module.css"


const AddDataset = (props) => {

    return (
        <React.Fragment>
            <Tab
                tabs={[{name: "Common Standards", to:'/datasets/add-dataset/common-standards'}, {name : "Medical Folder Dataset", to:'/datasets/add-dataset/medical-folder-dataset/'}]}
            />
            <div className={styles.main}>
                <Outlet/>
            </div>
        </React.Fragment>

    );
};

export default AddDataset;
