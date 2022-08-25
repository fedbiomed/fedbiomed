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
  import { useDispatch, useSelector, shallowEqual } from "react-redux";


const AccountRequestManagement = (props) => {

    React.useEffect(() => {
        listAccountRequests()
    }, [listAccountRequests])

    const users = useSelector((state) => state.user_requests_list, shallowEqual)

    const [pageIndex, setPageIndex] = useState(0);
    const [pageSize, setPageSize] = useState(20);
    const [sortField, setSortField] = useState('name');
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
            let display = users.slice(begin, end)

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
            field: 'password',
            name: 'Password',
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
            field: 'request_id',
            name: 'Request ID',
            truncateText: true,
            sortable: true,
        },
        {
            field: 'status',
            name: 'Status',
            truncateText: true,
            sortable: true,
        },
        {
          name: 'Approve Request',
          actions: [
              {render: (item) => <EuiButton onClick={()=>(setShowApproveRequestModal(true))}  iconType="checkInCircleFilled" color={"primary"}>Approve Request</EuiButton>}
          ] ,
        },
        {
          name: 'Reject Request',
          actions: [
              {render: (item) => <EuiButton  onClick={()=>(setShowRejectRequestModal(true))} iconType="crossInACircleFilled" color={"warning"}>Reject Request</EuiButton>}
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
        totalItemCount: users.length,
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

export default AccountRequestManagement