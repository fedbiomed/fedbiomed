import React from 'react';
import Tab from "../../components/common/Tab"
import {Outlet} from "react-router-dom";
import styles from "./AddDataset.module.css"


const AddDataset = (props) => {

    return (
        <React.Fragment>
            <Tab
                tabs={[{name: "Common Standards", to:'/datasets/add-dataset/'}, {name : "BIDS", to:'/datasets/add-dataset/bids/'}]}
            />
            <div className={styles.main}>
                <Outlet/>
            </div>
        </React.Fragment>

    );
};

export default AddDataset;