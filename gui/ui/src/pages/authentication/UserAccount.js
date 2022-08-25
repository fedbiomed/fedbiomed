import React, {useState} from 'react';
import {EuiPageBody,
        EuiSpacer,
        EuiTab,
        EuiTabs,
        EuiPageSection,
        EuiIcon,
        EuiTextColor} from "@elastic/eui"
import {Outlet, useNavigate} from 'react-router-dom'
import {useSelector, shallowEqual} from 'react-redux';

const UserAccount = (props) => {

    const userInfo = useSelector((state) => state.auth, shallowEqual)
    const [selectedTabId, setSelectedTabId] = useState('user-info');
    const navigate = useNavigate()

    const tabs = [
              {
                  id: 'user-info',
                  label: 'Account',
                  isSelected: true,
                  to:'/user-account/info',
                  color: "default",
                  display: true,
                   prepend: <EuiIcon type="user" />
              },

              {
                  id: 'change-password',
                  label: 'Change Password',
                  color: "default",
                  display: true,
                  to:'/user-account/change-password',
                  prepend: <EuiIcon type="tokenKey" />
              },
              {
                  id: 'user-management',
                  label: 'Manage Users',
                  to:'/user-account/user-management',
                  color: "default",
                  display: userInfo.role === 'Admin',
                  prepend: <EuiIcon type="users" />

              },
              {
                  id: 'request-account-user',
                  to:'/user-account/account-requests',
                  label: 'Approve new Users',
                  color: "default",
                  display: userInfo.role === 'Admin',
                  prepend: <EuiIcon type="lockOpen" />

             },
    ]
    const onSelectedTabChanged = (to, id) => {
        setSelectedTabId(id);
        navigate(to)
    };

    const renderTabs = () => {
        return tabs.map( (tab, index) => {
            return (
                <EuiTab
                    key={index}
                    href={tab.href}
                    onClick={() => onSelectedTabChanged(tab.to, tab.id)}
                    isSelected={tab.id === selectedTabId}
                    disabled={tab.disabled}
                    prepend={tab.prepend}
                    append={tab.append}
                >
                    {tab.display ? <EuiTextColor color={tab.color}>{tab.label}</EuiTextColor>: null}

                </EuiTab>
            )
        })

    }

    return (
        <React.Fragment>
                <EuiPageBody paddingSize="l" >
                    <EuiPageSection
                        hasBorder={false}
                        hasShadow={false}
                        paddingSize="none"
                        color="transparent"
                        borderRadius="none"
                    >

                        <EuiSpacer size={'l'}/>
                        <EuiTabs>
                            {renderTabs()}
                        </EuiTabs>
                        <EuiSpacer size={'l'}/>
                        <EuiPageSection
                            hasBorder={false}
                            hasShadow={false}
                            paddingSize="none"
                            color="transparent"
                            borderRadius="none"
                        >
                            <Outlet/>
                        </EuiPageSection>
                    </EuiPageSection>
                </EuiPageBody>
        </React.Fragment>

    )
}

export default UserAccount
