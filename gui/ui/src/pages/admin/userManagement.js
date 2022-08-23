import axios from "axios";
import {EP_ADMIN} from '../../constants';

import React, {
    Fragment,
    useCallback,
    useEffect,
    useState,
    createContext,
    useContext,
    useRef,
    createRef,
  } from 'react';
  import {
    EuiButton,
    EuiButtonEmpty,
    EuiButtonIcon,
    EuiCode,
    EuiContextMenuItem,
    EuiContextMenuPanel,
    EuiDataGrid,
    EuiFlexItem,
    EuiFlyout,
    EuiFlyoutBody,
    EuiFlyoutFooter,
    EuiFlyoutHeader,
    EuiLink,
    EuiModal,
    EuiModalBody,
    EuiModalFooter,
    EuiModalHeader,
    EuiModalHeaderTitle,
    EuiPopover,
    EuiScreenReaderOnly,
    EuiText,
    EuiTitle,
    EuiHealth, EuiPanel
  } from '@elastic/eui';
  import { Link } from 'react-router-dom';
  import Button from '../../components/common/Button';
  import {UserManagementModal, UserPasswordResetManagement, UserAccountCreation} from './userManagementModal';
  
  // see https://elastic.github.io/eui/#/tabular-content/data-grid


const UserManagement = (props) => {

    const [isadmin, setIsAdmin] = useState(false)
    const DataContext = createContext();
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [showResetPwdModal, setShowResetPwdModal] = useState(false);
    const [showAccountCreationModal, setShowAccountCreationModal] = useState(false);
    const raw_data = [];

    const closeDeleteModal = () => {setShowDeleteModal(false)}
    const closeResetPwdModal = () => {setShowResetPwdModal(false)}

    // pagination stuff for data grid
    const [pagination, setPagination] = useState({ pageIndex: 0, pageSize: 10 });
    const setPageSize = useCallback(
        (pageSize) =>
          setPagination((pagination) => ({
            ...pagination,
            pageSize,
            pageIndex: 0,
          })),
        [pagination, setPagination]
      );
      const setPageIndex = useCallback(
        (pageIndex) =>
          setPagination((pagination) => ({ ...pagination, pageIndex })),
        [pagination,setPagination]
      );
    
    const isAccessGranted = () => {
        // check if access is granted for admin
        axios.get(EP_ADMIN).then((response) => {
            
            setIsAdmin(true)
        }).catch((error) => {
            alert(error.response.data.message)
            
        })
        
    }

        // create dummy data for testing sake
        for (let i = 1; i < 100; i++) {
            raw_data.push({
                name:  "lolo",
                surname: "tata",
                email: "lolo@email.com",
                user_role: "simple user",
                creation_date:"01/01/1999",
                last_connection: "08/19/2022",
                privileges: <Fragment><EuiButton>Promote</EuiButton></Fragment>, // add option to logout users
                reset_password: <Fragment><EuiFlexItem grow={false} aria-label="1"><EuiButtonEmpty onClick={()=>(setShowResetPwdModal(true))} iconType="refresh"></EuiButtonEmpty></EuiFlexItem></Fragment>,
                delete_account: <Fragment><EuiFlexItem grow={false} aria-label="2"><EuiButtonIcon onClick={() => (setShowDeleteModal(true))} iconType="trash"></EuiButtonIcon></EuiFlexItem></Fragment>
                
            })
        }

    // columns contains scheme for desigining grid
    const columns = [
        {
          id: 'name',
          displayAsText: 'Name',
          defaultSortDirection: 'asc',
          initialWidth: 100,
          actions: { showMoveLeft: false, showMoveRight: false },
        },
        {
            id: 'surname',
            displayAsText: 'Surname',
            defaultSortDirection: 'asc',
            initialWidth: 100,
            actions: { showMoveLeft: false, showMoveRight: false },
          },
        {
            id: 'email',
            displayAsText: 'Email address',
            initialWidth: 130,
            actions: { showMoveLeft: false, showMoveRight: false },
        },
        {
            id: 'user_role',
            displayAsText: 'User Role',
            initialWidth: 130,
            actions: { showMoveLeft: false, showMoveRight: false },
        },
        {
            id: 'creation_date',
            displayAsText: 'account creation',
            initialWidth: 130,
            actions: { showMoveLeft: false, showMoveRight: false },
        },
        {
            id: 'last_connection',
            displayAsText: 'last connection',
            initialWidth: 130,
            actions: { showMoveLeft: false, showMoveRight: false },
        },
        {
            id: 'privileges',
            displayAsText: 'Upgrade/Downgrade',
            initialWidth: 130,
            actions: { showMoveLeft: false, showMoveRight: false },
        },
        {
            id: 'reset_password',
            displayAsText: 'Reset Password',
            initialWidth: 130,
            actions: { showMoveLeft: false, showMoveRight: false },
        },
        {
            id: 'delete_account',
            displayAsText: 'Delete Account',
            initialWidth: 130,
            actions: { showMoveLeft: false, showMoveRight: false },
        }
    ]

      // Column visibility
    const [visibleColumns, setVisibleColumns] = useState(
        columns.map(({ id }) => id) // initialize to the full set of columns
    );

    return (
        <Fragment>
           
            <div>
                {isAccessGranted()}
                {isadmin?<h1>
                    This is User Management webpage. Only admin should be able to reach this page</h1>:
                    <h1>You are simple user. you cannot access this page</h1>}
            </div>
            <div>
                <p>Add a new account</p><EuiButton onClick={()=> (setShowAccountCreationModal(true))}>Create new account</EuiButton>
            </div>
            <div>
                    

                    <EuiDataGrid
                        aria-label="Data grid demo"
                        columns={columns}
                        columnVisibility={{ visibleColumns, setVisibleColumns }}
                        renderCellValue={({ rowIndex, columnId }) => raw_data[rowIndex][columnId]}
                        gridStyle={{
                            border: 'none',
                            stripes: true,
                            rowHover: 'highlight',
                            header: 'underline',
                            // If showDisplaySelector.allowDensity={true} from toolbarVisibility, fontSize and cellPadding will be superceded by what the user decides.
                            cellPadding: 'm',
                            fontSize: 'm',
                            footer: 'overline'
                          }}
                        rowCount={raw_data.length}
                        pagination={{
                            ...pagination,
                            pageSizeOptions: [5, 10, 20, 50],

                            onChangeItemsPerPage: setPageSize,
                            onChangePage: setPageIndex
                        }}

                    />
                    
            </div>
            
            <div>
            {showDeleteModal?<UserManagementModal
                             show={showDeleteModal}
                             title="Delete Account?"
                             onClose={closeDeleteModal}
                             text={"Are you sure you want to delete this acount?"}/>:null}
            {showResetPwdModal?<UserPasswordResetManagement
                                 show={showResetPwdModal}
                                 title="Reset Password"
                                 onClose={closeResetPwdModal}
                                 />:null}
            {showAccountCreationModal? <UserAccountCreation></UserAccountCreation>:null}
        </div>
        </Fragment>

    )
}

export default UserManagement;