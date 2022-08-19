import React, {useState, useMemo} from 'react';
import {EuiPageBody,
        EuiPageContentBody,
        EuiPageHeader,
        EuiSpacer,
        EuiTab,
        EuiTabs,
        EuiPageContent,
        EuiTextColor} from "@elastic/eui"
import { EP_UPDATE_PASSWORD } from '../../constants';
import Button from '../../components/common/Button';
import Modal from '../../components/common/Modal';
import {Text, Label} from '../../components/common/Inputs';
import {useSelector, useDispatch} from 'react-redux';
import axios from 'axios';
import UserInfo from "./UserInfo";
import PasswordChange from "./PasswordChange";
import UserManagement from '../admin/userManagement';
import UserRequestManagement from '../admin/userRequestManagement';
import {EP_ADMIN} from '../../constants';

const UserAccount = () => {
    // displays webpage for user account management 

    const [newPassword, setNewPassword] = useState({show: false})
    const userInfo = useSelector((state) => state.auth)
    const [selectedTabId, setSelectedTabId] = useState('user-info');


    let initial_passwords = {'password1': '', 'password2': ''}

    const [password, setPassword] = useState(initial_passwords);

    const dispatch = useDispatch();

    const onChangePassword = (key, target_value) => {
        setNewPassword({show:true})
        setPassword({...password, [key]: target_value})
    }


    const onCloseModal = () => {
        // Closes modal window (change password window)
        setNewPassword({show:false})
        setPassword(initial_passwords)
    }

    const feedUserData = () =>{
        // collects access token info about user and adds them into a dictionary.
        // Adds a field password mapping a `Button` component
        const user_info = userInfo;
        user_info['password'] = <Button onClick={onChangePassword}>Change</Button>
        return user_info

    }

    const [isadmin, setIsAdmin] = useState(false)
    const isAccessGranted = () => {
        axios.get(EP_ADMIN).then((response) => {
            //alert("admin logged in")
            setIsAdmin(true)
        }).catch((error) => {
           // alert(error.response.data.message)
           setIsAdmin(false)
        })}
    const tabs = [
              {
                  id: 'user-info',
                  label: 'Account',
                  isSelected: true,
                  content: <UserInfo/>,
                  color: "default",
                  display: true
              },

              {
                  id: 'change-password',
                  label: 'Change Password',
                  content: <PasswordChange user={userInfo} />,
                  color: "default",
                  display: true
              },
              {
                 id: 'manage-user',
                 label: 'Manage Users',
                 content: <UserManagement/>,
                 color: "#788083",
                 display: isadmin

              },
              {
                id: 'request-account-user',
                label: 'Approve new Users',
                content: <UserRequestManagement/>,
                color: "#788083",
                display: isadmin

             },
    ]

    const selectedTabContent = useMemo(() => {
        return tabs.find((obj) => obj.id === selectedTabId)?.content;
      }, [selectedTabId]);

    const selectedTabColor = useMemo(() => {
        return tabs.find((obj) => obj.id === selectedTabId)?.color;
      }, [selectedTabId]);

    const onSelectedTabChanged = (id) => {
        setSelectedTabId(id);
    };
    isAccessGranted()

    const renderTabs = () => {
        return tabs.map( (tab, index) => {
            return (
                <div>
                    
                    <EuiTab
                        key={index}
                        href={tab.href}
                        onClick={() => onSelectedTabChanged(tab.id)}
                        isSelected={tab.id === selectedTabId}
                        disabled={tab.disabled}
                        prepend={tab.prepend}
                        append={tab.append}
                        label={ tab.content}
                    >   
                        {tab.display ? <EuiTextColor color={tab.color}>{tab.label}</EuiTextColor>: null}
                        
                    </EuiTab>
                    </div>
                )
        })

    }

    return (
        <React.Fragment>
                <EuiPageBody paddingSize="l" >
                    <EuiPageContent
                        hasBorder={false}
                        hasShadow={false}
                        paddingSize="none"
                        color="transparent"
                        borderRadius="none"
                    >
                        <EuiPageHeader
                            paddingSize={'xs'}
                            pageTitle="User Account"
                            iconType="user"
                            description="This page contains user account info"
                          />

                        <EuiSpacer size={'l'}/>
                              <EuiTabs>
                                    {renderTabs()}
                              </EuiTabs>
                        <EuiSpacer size={'l'}/>
                        <EuiPageContent
                            hasBorder={false}
                            hasShadow={false}
                            paddingSize="none"
                            color="transparent"
                            borderRadius="none"
                        >
                            <EuiPageContentBody>
                                {selectedTabContent}
                            </EuiPageContentBody>
                        </EuiPageContent>
                    </EuiPageContent>
                </EuiPageBody>
        </React.Fragment> 
        
    )
}

export default UserAccount




 //
 // <Modal show={newPassword.show} onModalClose={onCloseModal}>
 //                    <Modal.Header><h2>Change Password</h2></Modal.Header>
 //                    <Modal.Content>
 //                        <Label>Submit new password</Label>
 //                        <Text placeholder={"Enter password"}  type="password" value={password.password1} onChange={(e) => onChangePassword("password1", e.target.value)} minlength='10' required>
 //                            {"Enter password"}
 //                        </Text>
 //                        <Label>write again new password (both of them should match)</Label>
 //                        <Text placeholder={"Enter password"} type="password"  value={password.password2} onChange={(e) => onChangePassword("password2", e.target.value)}  required minlength='10'>
 //                            {"Enter password"}
 //                        </Text >
 //                        <Button onClick={onSubmitNewPassword}>Confirm</Button>
 //                        <Button onClick={onCloseModal}>Cancel</Button>
 //                    </Modal.Content>
 //                </Modal>