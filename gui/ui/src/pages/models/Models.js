import React from 'react';
import ModelsList from "./ModelsList";

const Models = () => {

    return (
        <React.Fragment>
            <div className="frame-header">
                <div>
                    <p> This page display <b>TrainingPlans</b> that are requested by researcher, registered by node
                        owner or default models
                    </p>
                </div>
                <div className={"row"}>
                    <div className={"note"} style={{width:'80%'}}>
                        <div>
                            Please display models and take action in the preview page
                        </div>
                    </div>
                </div>
            </div>
            <div className="frame-content">
                <ModelsList/>
            </div>
            <div className="frame-footer">
            </div>
        </React.Fragment>

    );
};

export default Models;