import React from 'react';
import {EuiPageBody,
        EuiSpacer,
        EuiTab,
        EuiTabs,
        EuiPageSection,
        EuiIcon,
        EuiTextColor} from "@elastic/eui"
import {Outlet, useNavigate} from 'react-router-dom'
import {useSelector, shallowEqual} from 'react-redux';
import { useLocation } from 'react-router-dom';

const UserAccount = () => {

    const userInfo = useSelector((state) => state.auth, shallowEqual)
    const navigate = useNavigate()
    const location = useLocation()

    const tabs = [
              {
                  id: 'user-info',
                  label: 'Account',
                  isSelected: location.pathname.includes('/user-account/info') ||
                      location.pathname === '/user-account',
                  to:'/user-account/info',
                  color: "default",
                  display: true,
                   prepend: <EuiIcon type="user" />
              },

              {
                  id: 'change-password',
                  label: 'Change Password',
                  color: "default",
                  isSelected: location.pathname.includes('user-account/change-password'),
                  display: true,
                  to:'/user-account/change-password',
                  prepend: <EuiIcon type="tokenKey" />
              },
              {
                  id: 'user-management',
                  label: 'Manage Users',
                  to:'/user-account/user-management',
                   isSelected: location.pathname.includes('user-account/user-management'),
                  color: "default",
                  display: userInfo.role === 'Admin',
                  prepend: <EuiIcon type="users" />

              },
              {
                  id: 'request-account-user',
                  to:'/user-account/account-requests',
                  label: 'Approve new Users',
                  isSelected: location.pathname.includes('user-account/account-requests'),
                  color: "default",
                  display: userInfo.role === 'Admin',
                  prepend: <EuiIcon type="lockOpen" />

             },
    ]
    const onSelectedTabChanged = (to, id) => {
        navigate(to)
    };

    const renderTabs = () => {
        return tabs.map( (tab, index) => {
            if(tab.display){
                return (
                    <EuiTab
                        key={index}
                        href={tab.href}
                        onClick={() => onSelectedTabChanged(tab.to)}
                        isSelected={tab.isSelected}
                        disabled={tab.disabled}
                        prepend={tab.prepend}
                        append={tab.append}
                    >
                         <EuiTextColor color={tab.color}>{tab.label}</EuiTextColor>

                    </EuiTab>
                )
            }else{
                return null
            }
        })

    }

    return (
        <React.Fragment>
                <EuiPageBody paddingSize="l" >
                    <EuiPageSection
                        paddingSize="none"
                        color="transparent"
                    >

                        <EuiSpacer size={'l'}/>
                        <EuiTabs>
                            {renderTabs()}
                        </EuiTabs>
                        <EuiSpacer size={'l'}/>
                        <EuiPageSection
                            paddingSize="none"
                            color="transparent"
                        >
                            <Outlet/>
                        </EuiPageSection>
                    </EuiPageSection>
                </EuiPageBody>
        </React.Fragment>

    )
}

export default UserAccount
