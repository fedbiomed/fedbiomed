import React from 'react';
import AuthLayout from "./AuthLayout";
import {EuiTitle,
        EuiFlexItem,
        EuiFlexGroup,
        EuiSpacer,
        EuiCallOut,
} from '@elastic/eui';


import styles from "./Auth.module.css";
import logo from "../../assets/img/fedbiomed-logo-small.png";
import {Link} from "react-router-dom";
import RegisterForm from "./RegisterForm";


const Register = (props) => {

    return (
        <AuthLayout>
            <div className={styles.centered}>
                <img className={styles.imgLogo} alt="fedbiomed-logo" src={logo}/>
                <EuiTitle>
                    <h1>Register</h1>
                </EuiTitle>
                <EuiSpacer size="m" />
                <p>Register to Fed-BioMed. <br/> </p>
                <EuiSpacer size="m" />
                <EuiCallOut title="Attention!" color="warning" iconType="user" style={{textAlign:"left"}}>
                    <p>
                      By submitting this form, you will be requesting registration. You can only
                        <br/> log in if your request is approved by the system administrator.
                    </p>
                </EuiCallOut>
            </div>
            <EuiFlexGroup justifyContent="spaceAround">
                <EuiFlexItem grow={false} style={{minWidth:400}}>
                        <RegisterForm navigate_to={'/login'}/>
                    <EuiSpacer size="l" />
                    <EuiFlexItem>
                        <p>Already have an account! <Link to={"/login"}>Go to login page!</Link></p>
                    </EuiFlexItem>
                </EuiFlexItem>
            </EuiFlexGroup>
        </AuthLayout>
    );
};

export default Register;