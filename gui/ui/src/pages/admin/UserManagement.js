import React, { Fragment, useState } from 'react';
import {
    EuiButton,
    EuiButtonIcon,
    EuiBasicTable,
    EuiTitle,
    EuiSpacer
  } from '@elastic/eui';

import {UserManagementModal, UserPasswordResetManagement, UserAccountCreation} from './userManagementModal';
import {listUsers} from "../../store/actions/userManagementActions";
import {connect} from 'react-redux'

let raw_data = [
     { name:  "lolo", surname: "tata", email: "lolo@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lo123123lo2", surname: "tata", email: "lolo23244@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo3", surname: "tataa", email: "lolo@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "loasdlo4", surname: "taasdta", email: "lolo234@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo6", surname: "tata", email: "lolo@ema234il.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo7", surname: "atata", email: "lolo@emai234l.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "ldasdolo8", surname: "tatasda", email: "lol234o@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo9", surname: "tata", email: "lolo@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolso9", surname: "tata", email: "lolo@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo9", surname: "tata", email: "lolo@emai234l.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "Aosasdlo9", surname: "t2344ata", email: "lolo@em234ail.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
         { name:  "ldasdolo8", surname: "tatasda", email: "lol234o@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo9", surname: "tata", email: "lolo@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolso9", surname: "tata", email: "lolo@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo9", surname: "tata", email: "lolo@emai234l.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "Aosasdlo9", surname: "t2344ata", email: "lolo@em234ail.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
         { name:  "ldasdolo8", surname: "tatasda", email: "lol234o@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo9", surname: "tata", email: "lolo@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolso9", surname: "tata", email: "lolo@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo9", surname: "tata", email: "lolo@emai234l.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "Aosasdlo9", surname: "t2344ata", email: "lolo@em234ail.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
         { name:  "ldasdolo8", surname: "tatasda", email: "lol234o@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo9", surname: "tata", email: "lolo@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolso9", surname: "tata", email: "lolo@email.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "lolo9", surname: "tata", email: "lolo@emai234l.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },
     { name:  "Aosasdlo9", surname: "t2344ata", email: "lolo@em234ail.com",  role: "simple user", created:"01/01/1999", last_connection: "08/19/2022" },

]


const UserManagement = (props) => {

    const [pageIndex, setPageIndex] = useState(0);
    const [pageSize, setPageSize] = useState(15);
    const [sortField, setSortField] = useState('name');
    const [sortDirection, setSortDirection] = useState('asc');
    const [showAccountCreationModal, setShowAccountCreationModal] = useState(false);

    const [tableLoading, setTableLoading] = useState(true)

    const [items, setItems] = useState([])
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [showResetPwdModal, setShowResetPwdModal] = useState(false);


    const closeDeleteModal = () => {setShowDeleteModal(false)}
    const closeResetPwdModal = () => {setShowResetPwdModal(false)}


    React.useEffect( () => {
        props.listUsers()
    }, [props.listUsers])

    console.log(props.user_list)
    /**
     * Lifecycle method to keep track change on use table
     */
    React.useEffect( () => {

        setTableLoading(true)
        setTimeout(() => {
            let begin = pageIndex * pageSize
            let end = pageIndex * pageSize + pageSize
            let display = raw_data.slice(begin, end)

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
            setTableLoading(false)
        }, 500)

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
            field: 'name',
            name: 'Name',
            truncateText: true,
            sortable: true,
        },
        {
            field: 'surname',
            name: 'Surname',
            truncateText: true,
            sortable: true,
          },
        {
            field: 'email',
            name: 'E-Mail',
            truncateText: true,
            sortable: true,
        },
        {
            field: 'role',
            name: 'User Role',
            truncateText: true,
            sortable: true,
        },
        {
            field: 'created',
            name: 'Account Created',
            truncateText: true,
            sortable: true,
        },
        {
            field: 'last_connection',
            name: 'Last Login Date',
            truncateText: true,
            sortable: true,
        },
        {
          name: 'Change Role',
          actions: [
              {render: (item) => <EuiButton onClick={()=>(setShowResetPwdModal(true))}  iconType="user" color={"primary"}>Change Role</EuiButton>}
          ] ,
        },
        {
          name: 'Reset Pass',
          actions: [
              {render: (item) => <EuiButton  onClick={()=>(setShowResetPwdModal(true))} iconType="tokenKey" color={"warning"}>Reset Pass</EuiButton>}
          ] ,
        },
                {
          name: 'Remove',
          actions: [
              {render: (item) => <EuiButton onClick={() => (setShowDeleteModal(true))} iconType="trash" color={"danger"}>Delete</EuiButton>},
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
        totalItemCount: raw_data.length,
        pageSizeOptions: [20, 40, 60],
    };

    return (
        <Fragment>
           
            <EuiTitle size={'s'}>
                <h2>This is User Management webpage. Only admin should be able to reach this page</h2>
            </EuiTitle>
            <EuiSpacer size={'l'}/>
            <EuiButton onClick={()=> (setShowAccountCreationModal(true))}>Create new account</EuiButton>
            <EuiSpacer size={'l'}/>

            <EuiBasicTable
                aria-label={"User table"}
                items={items}
                itemId="id"
                columns={columns}
                pagination={pagination}
                sorting={sorting}
                hasActions={true}
                onChange={onTableChange}
                loading={tableLoading}
            />

            
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

const mapDispatchToProps = (dispatch) => {
    return {
        listUsers : () => dispatch(listUsers())
    }
}


const mapStateToProps = (state) => {
    return {
        error : state.users.error,
        user_list : state.users.user_list
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(UserManagement);