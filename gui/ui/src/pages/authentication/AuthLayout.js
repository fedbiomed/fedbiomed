import React from 'react';
import {EuiPage,
        EuiPageSection
} from '@elastic/eui';


const AuthLayout = ({children}) => {
    return (
        <EuiPage paddingSize="l" grow={true}  style={{minHeight: "100vh"}}>
                <EuiPageSection
                  alignment={'center'}
                  paddingSize="l"
                  color={'plain'}
                  grow={true}
                >
                    {children}
                </EuiPageSection>
        </EuiPage>
    );
};

export default AuthLayout;