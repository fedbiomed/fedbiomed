import React, { Fragment, useState } from 'react';
import {
    EuiButton,
    EuiButtonIcon,
    EuiToast,
    formatDate,
    EuiInMemoryTable,
    EuiSuperSelect,
    EuiSpacer
  } from '@elastic/eui';

import {
    UserPasswordResetManagement,
    UserAccountCreation,
    UserManagementConfirmation
} from './UserManagementModal';
import {deleteUser, listUsers} from "../../store/actions/userManagementActions";
import {connect, useDispatch} from 'react-redux'
import {USER_MANAGEMENT_ERROR, USER_MANAGEMENT_SUCCESS_MESSAGE} from "../../store/actions/actions";
import UserRoleSelectBox from "./UserRoleSelectBox";


const UserManagement = (props) => {

    const [pageIndex, setPageIndex] = useState(0);
    const [pageSize, setPageSize] = useState(15);
    const [sortField, setSortField] = useState('name');
    const [sortDirection, setSortDirection] = useState('asc');
    const [showAccountCreationModal, setShowAccountCreationModal] = useState(false);

    const [items, setItems] = useState([])
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [showResetPwdModal, setShowResetPwdModal] = useState(false);
    const [userToDelete, setUserToDelete] = useState(null)
    const [userToResetPassword, setUserToResetPassword] = useState(null)

    const closeDeleteModal = () => {setShowDeleteModal(false); setUserToDelete(null)}
    const closeResetPwdModal = () => {setShowResetPwdModal(false)}

    const dispatch = useDispatch()

    /**
     * Call list user API when component loaded for the first time
     */
    React.useEffect(  () => {
        props.listUsers()
    }, [props.listUsers])


    /**
     * Update table items each time user list changed
     */
    React.useEffect( () => {
        setItems(props.user_list)
    }, [props.user_list])

    /**
     * Lifecycle method to keep track change on use table
     */
    React.useEffect( () => {
        let begin = pageIndex * pageSize
        let end = pageIndex * pageSize + pageSize
        let display = props.user_list.slice(begin, end)

        // Sort with custom function
        display.sort( (a, b) => {
                        if ( a[sortField] < b[sortField] ){
                            return  sortDirection === "asc" ? -1 : 1 ;
                        }

                        if ( a[sortField] > b[sortField] ){
                            return sortDirection === "asc" ? 1 : -1 ;
                        }

                        return 0;
        })
        setItems(display)
    }, [pageIndex, pageSize, sortField, sortDirection])

    /**
     * On table value is changed
     * @param page
     * @param sort
     */
    const onTableChange = ({ page = {}, sort = {} }) => {
        const { index: pageIndex, size: pageSize } = page;
        const { field: sortField, direction: sortDirection } = sort;
        setPageIndex(pageIndex);
        setPageSize(pageSize);
        setSortField(sortField);
        setSortDirection(sortDirection);
    };

    /**
     * Sorting credentials
     * @type {{sort: {field: string, direction: string}}}
     */
    const sorting = {
        sort: {
          field: sortField,
          direction: sortDirection,
        },
      };

        // Column contains scheme for designing grid
    const columns = [
        {
            name: 'Name',
            render: (item) => `${item.user_surname?.toUpperCase()} ${item?.user_name}`,
            truncateText: true,
            sortable: true,
        },
        {
            field: 'user_email',
            name: 'E-Mail',
            truncateText: false,
            sortable: true,
        },
        {
            field: 'creation_date',
            name: 'Account Created',
            dataType: 'date',
            render: (date) => formatDate(date, 'Do MMMM YYYY HH:MM'),
            truncateText: false,
            sortable: true,
        },
        {
            field: 'last_login',
            name: 'Last Login Date',
            render: (date) => formatDate(date, 'Do MMMM YYYY HH:MM'),
            truncateText: false,
            sortable: true,
        },
        {
          name: 'Change Role',
          actions: [
              {render: (item) => <UserRoleSelectBox selected={item.user_role} userId={item.user_id} />}
          ],
        },

        {
          name: 'Reset Pass / Remove',
          actions: [
              {render: (item) => <EuiButtonIcon  aria-label={'Reset'}
                                                 onClick={()=> {setShowResetPwdModal(true);setUserToResetPassword(item.user_id)}}
                                                 iconType="tokenKey"
                                                 color={"warning"}>Reset Pass</EuiButtonIcon>},
              {render: (item) => <EuiButtonIcon aria-label={'Delete'}
                                                onClick={() => {setShowDeleteModal(true);setUserToDelete(item.user_id)}}
                                                iconType="trash"
                                                color={"danger"}>Delete</EuiButtonIcon>},
          ] ,
        },
    ]

    /**
     * Pagination options
     * @type {{pageSizeOptions: number[], pageIndex: number, pageSize: number, totalItemCount: number}}
     */
    const pagination = {
        pageIndex: pageIndex,
        pageSize: pageSize,
        totalItemCount: props.user_list.length,
        pageSizeOptions: [10, 20, 30],
    };


    const deleteUserConfirmationHandler = () => {
        let user_id = userToDelete
        props.deleteUser(user_id)
    }

    return (
        <Fragment>
            {props.error ? (
                 <React.Fragment>
                     <EuiSpacer size="l" />
                     <EuiToast
                            title="Opps!"
                            color="danger"
                            iconType="alert"
                            onClose={() => dispatch({type: USER_MANAGEMENT_ERROR, payload:false})}
                         >
                         <p>{props.error}</p>
                     </EuiToast>
                 </React.Fragment>
                ) : null}
             {props.success ? (
                 <React.Fragment>
                     <EuiSpacer size="l" />
                     <EuiToast
                            title="Done!"
                            color="success"
                            iconType="alert"
                            onClose={() => dispatch({type: USER_MANAGEMENT_SUCCESS_MESSAGE, payload:null})}
                        >
                         <p>{props.success}</p>
                     </EuiToast>
                 </React.Fragment>

                         ) : null}
            <EuiSpacer size={'l'}/>
            <EuiButton onClick={()=> (setShowAccountCreationModal(true))}>Create new account</EuiButton>
            <EuiSpacer size={'l'}/>
            {props.user_list && items  ? (
                <EuiInMemoryTable
                        aria-label={"User table"}
                        items={items}
                        columns={columns}
                        pagination={pagination}
                        sorting={sorting}
                        hasActions={true}
                        onChange={onTableChange}
                        loading={props.loading}
                        search={{box: {
                                  incremental: true,
                                }}}
                />
            ) : null}

            <UserManagementConfirmation
                             show={showDeleteModal}
                             title="Delete Account?"
                             onClose={closeDeleteModal}
                             onConfirm={deleteUserConfirmationHandler}
                             text={"Are you sure you want to delete this account?"}/>

            <UserPasswordResetManagement
                                 show={showResetPwdModal}
                                 title="Reset Password"
                                 onClose={closeResetPwdModal}
                                 userId={userToResetPassword}
                                 />
            <UserAccountCreation
                show={showAccountCreationModal}
                onClose={() => setShowAccountCreationModal(false)}
                afterRegister={() => {dispatch(listUsers())}}
            />
        </Fragment>

    )
}

const mapDispatchToProps = (dispatch) => {
    return {
        listUsers : () => dispatch(listUsers()),
        deleteUser : (id) => dispatch(deleteUser(id))
    }
}


const mapStateToProps = (state) => {
    return {
        error : state.users.error,
        user_list : state.users.user_list,
        loading : state.users.loading,
        success: state.users.success,
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(UserManagement);