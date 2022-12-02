import React, {
    Fragment,
    useState,
  } from 'react';
import {
    EuiButton,
    EuiBasicTable,
    formatDate,
    EuiSpacer,
    EuiToast
  } from '@elastic/eui';

import {AccountRequestManagementModal} from "./UserManagementModal";
import { approveAccountRequest, listAccountRequests, rejectAccountRequest } from '../../store/actions/accountRequestActions';
import { connect, useDispatch } from "react-redux";
import {
    USER_REQUESTS_ERROR_MESSAGE,
    USER_REQUESTS_SUCCESS_MESSAGE
} from "../../store/actions/actions";


const AccountRequestManagement = (props) => {

    const [pageIndex, setPageIndex] = useState(0);
    const [pageSize, setPageSize] = useState(20);
    const [sortField, setSortField] = useState('user_name');
    const [sortDirection, setSortDirection] = useState('asc');

    const [items, setItems] = useState([])
    const [selectedItem, setSelectedItem] = useState({})
    const [showModal, setShowModal] = useState(false);
    const [actionType, setActionType] = useState('')
    const [title, setTitle] = useState('')
    const dispatch = useDispatch()
    const {listAccountRequests} = props

    const onSelect = (item, actionType, title) => {
        setActionType(actionType)
        setTitle(title)
        setSelectedItem(item)
        setShowModal(true)
    }
    const confirmAccountRequestModal = () => {
        actionType === 'APPROVE' ? props.approveAccountRequestAction(selectedItem) 
        : actionType === 'REJECT' ? props.rejectAccountRequestAction(selectedItem)
        : setShowModal(false)
        closeAccountRequestModal()
    }
    const closeAccountRequestModal = () => {
        setActionType('')
        setTitle('')
        setSelectedItem({})
        setShowModal(false)
    }

     /**
     * Call list user requests API when component loaded for the first time
     */
    React.useEffect(() => {
        listAccountRequests()
    }, [listAccountRequests ])



    /**
     * Lifecycle method to keep track change on use table
     */
     React.useEffect( () => {
         setItems(props.user_requests)
        let begin = pageIndex * pageSize
        let end = pageIndex * pageSize + pageSize
        let display = props.user_requests.slice(begin, end)

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

    }, [pageIndex, pageSize, sortField, sortDirection, props.user_requests])


     /**
     * Update table items each time user request list changed
     */
     React.useEffect( () => {
        setItems(props.user_requests)
    }, [props.user_requests])

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
            truncateText: true,
            sortable: true,
        },
        {
            field: 'creation_date',
            name: 'Request Created',
            dataType: 'date',
            render: (creation_date) => formatDate(creation_date, 'Do MMMM YYYY HH:MM'),
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
          align: 'center',
            render: (item) => <EuiButton
                                        onClick={
                                            ()=>(onSelect(item,
                                                'APPROVE', 'Approve account request creation ?'))}
                                        iconType="checkInCircleFilled" color={"primary"}>
                                        {item.request_status === "REJECTED" ? 'Approve Back' : 'Approve'}
                                 </EuiButton>


        },
        {
          name: 'Reject',
          align: 'center',
          render: (item) => <EuiButton
                                    disabled={item.request_status === "REJECTED" ? true : false}
                                    onClick={()=>(onSelect(item,
                                        'REJECT', 'Reject account request creation ?'))}
                                    iconType="crossInACircleFilled" color={"warning"}>
                                          Reject
                                  </EuiButton>

        },
    ]


    /**
     * Pagination options
     * @type {{pageSizeOptions: number[], pageIndex: number, pageSize: number, totalItemCount: number}}
     */
    const pagination = {
        pageIndex: pageIndex,
        pageSize: pageSize,
        totalItemCount: props.user_requests.length,
        pageSizeOptions: [20, 40, 60],
    };


    return (
        <Fragment>
            <EuiSpacer size={'l'}/>
            { props.error ? (
                <React.Fragment>
                     <EuiSpacer size="l" />
                     <EuiToast
                            title="Opps!"
                            color="danger"
                            iconType="alert"
                            onClose={() => dispatch({type: USER_REQUESTS_ERROR_MESSAGE, payload: null})}
                         >
                         <p>{props.error}</p>
                     </EuiToast>
                 </React.Fragment>
            ) : null


            }
            { props.success ? (
                <React.Fragment>
                     <EuiSpacer size="l" />
                     <EuiToast
                            title="Done!"
                            color="success"
                            iconType="checkInCircleFilled"
                            onClose={() => dispatch({type: USER_REQUESTS_SUCCESS_MESSAGE, payload: null})}
                         >
                         <p>{props.success}</p>
                     </EuiToast>
                 </React.Fragment>
            ) : null

            }
            <EuiSpacer size="l" />
            {props.user_requests ? (

                <EuiBasicTable
                    aria-label={"User requests table"}
                    items={items}
                    itemId="id"
                    columns={columns}
                    pagination={pagination}
                    sorting={sorting}
                    hasActions={true}
                    onChange={onTableChange}
                    loading={props.loading}
                />
            ) : null}
            <AccountRequestManagementModal
                show={showModal}
                title={title}
                onConfirmAccountRequestModal={confirmAccountRequestModal}
                onClose={closeAccountRequestModal}
                text={"Are you sure you want to perform this action?"}
                color={actionType === "APPROVE" ? 'primary' : 'danger' }
            />
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
        listAccountRequests : () => dispatch(listAccountRequests()),
        approveAccountRequestAction: (data) => dispatch(approveAccountRequest(data)),
        rejectAccountRequestAction: (data) => dispatch(rejectAccountRequest(data))
    }
}


/**
 * Map global state
 * @param {*} state 
 * @returns 
 */
 const mapStateToProps = (state) => {
    return {
        user_requests : state.user_requests.requests,
        error : state.user_requests.error,
        loading : state.user_requests.loading,
        success : state.user_requests.success
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(AccountRequestManagement);