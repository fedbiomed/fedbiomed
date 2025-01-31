import React, {
    Fragment,
    useCallback,
    useState,
  } from 'react';
  import {
    EuiButton,
    EuiButtonEmpty,
    EuiButtonIcon,
    EuiDataGrid,
    EuiFlexItem,
    EuiTitle,
    EuiSpacer
  } from '@elastic/eui';
  import {UserManagementModal, UserPasswordResetManagement, UserAccountCreation} from './userManagementModal';


const UserManagement = (props) => {

    const [isadmin, setIsAdmin] = useState(false)
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

            <EuiTitle size={'s'}>
                <h2>This is User Management webpage. Only admin should be able to reach this page</h2>
            </EuiTitle>
            <EuiSpacer size={'l'}/>
            <EuiButton onClick={()=> (setShowAccountCreationModal(true))}>Create new account</EuiButton>
            <EuiSpacer size={'l'}/>
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


            <div>
            {showDeleteModal?<UserManagementModal
                             show={showDeleteModal}
                             title="Delete Account?"
                             onClose={closeDeleteModal}
                             text={"Are you sure you want to delete this account?"}/>:null}
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