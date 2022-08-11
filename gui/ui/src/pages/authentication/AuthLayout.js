import React from 'react';
import {EuiPage,
        EuiPageContent,
        EuiPageBody
} from '@elastic/eui';


const AuthLayout = ({children}) => {
    return (
        <EuiPage paddingSize="l" grow={false}  style={{minHeight: "100vh"}}>
            <EuiPageBody paddingSize="l">
                <EuiPageContent
                  horizontalPosition="center"
                  paddingSize="l"
                >
                    {children}
                </EuiPageContent>
            </EuiPageBody>
        </EuiPage>
    );
};

export default AuthLayout;