import React, {
    Fragment,
    useState,
  } from 'react';
import {
    EuiButton,
    EuiBasicTable,
    EuiTitle,
    EuiSpacer
  } from '@elastic/eui';

  import { AccountRequestManagementModal } from './accountRequestManagementModal';
  import { listAccountRequests } from '../../store/actions/accountRequestActions';
  import { connect } from "react-redux";


const AccountRequestManagement = (props) => {


    
    React.useEffect(() => {
        // Get list of account creation requests
        props.listAccountRequests()
    }, [props.listAccountRequests])

    console.log(props.requests)

    const [pageIndex, setPageIndex] = useState(0);
    const [pageSize, setPageSize] = useState(20);
    const [sortField, setSortField] = useState('user_name');
    const [sortDirection, setSortDirection] = useState('asc');

    const [tableLoading, setTableLoading] = useState(true)

    const [items, setItems] = useState([])
    const [showApproveRequestModal, setShowApproveRequestModal] = useState(false);
    const [showRejectRequestModal, setShowRejectRequestModal] = useState(false);

    const closeApproveRequestModal = () => {setShowApproveRequestModal(false)}
    const closeRejectRequestModal = () => {setShowRejectRequestModal(false)}

    /**
     * Lifecycle method to keep track change on use table
     */
     React.useEffect( () => {

        setTableLoading(true)
        setTimeout(() => {
            let begin = pageIndex * pageSize
            let end = pageIndex * pageSize + pageSize
            let display = props.requests.slice(begin, end)

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
            field: 'user_name',
            name: 'Name',
            truncateText: true,
            sortable: true,
        },
        {
            field: 'user_surname',
            name: 'Surname',
            truncateText: true,
            sortable: true,
          },
        {
            field: 'user_email',
            name: 'E-Mail',
            truncateText: true,
            sortable: true,
        },
        {
            field: 'user_role',
            name: 'User Role',
            truncateText: true,
            sortable: true,
        },
        {
            field: 'creation_date',
            name: 'Account Created',
            truncateText: true,
            sortable: true,
        },
        {
            field: 'request_status',
            name: 'Status',
            truncateText: true,
            sortable: true,
        },
        {
          name: 'Approve',
          actions: [
              {render: (item) => <EuiButton onClick={()=>(setShowApproveRequestModal(true))}  iconType="checkInCircleFilled" color={"primary"}>Approve</EuiButton>}
          ] ,
        },
        {
          name: 'Reject',
          actions: [
              {render: (item) => <EuiButton  onClick={()=>(setShowRejectRequestModal(true))} iconType="crossInACircleFilled" color={"warning"}>Reject</EuiButton>}
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
        totalItemCount: props.requests.length,
        pageSizeOptions: [20, 40, 60],
    };


    return (
        <Fragment>

            <EuiTitle size={'s'}>
                <h2>This is Account Request Management webpage. Only admin should be able to reach this page</h2>
            </EuiTitle>
            <EuiSpacer size={'l'}/>

            <EuiBasicTable
                aria-label={"User requests table"}
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
                {showApproveRequestModal?<AccountRequestManagementModal
                                show={showApproveRequestModal}
                                title="Approve account request creation ?"
                                onClose={closeApproveRequestModal}
                                text={"Are you sure you want to approve this request?"}/>:null}
                {showRejectRequestModal?<AccountRequestManagementModal
                                    show={showRejectRequestModal}
                                    title="Reject account request creation ?"
                                    onClose={closeRejectRequestModal}
                                    text={"Are you sure you want to reject this request?"}/>:null}
            </div>
        </Fragment>

    )
}

/**
 * Pass action to props of component
 * @param {function} dispatch 
 * @returns 
 */
 const mapDispatchToProps = (dispatch) => {
    return {
        listAccountRequests : () => dispatch(listAccountRequests())
    }
}


/**
 * Map global state
 * @param {*} state 
 * @returns 
 */
 const mapStateToProps = (state) => {
    return {
        requests : state.user_requests.requests
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(AccountRequestManagement);