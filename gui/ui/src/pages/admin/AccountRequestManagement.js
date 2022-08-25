import React, {
    Fragment,
    useState,
  } from 'react';
import {
    EuiButton,
    EuiButtonIcon,
    EuiBasicTable,
    EuiTitle,
    EuiSpacer
  } from '@elastic/eui';

let raw_data = [
     { name:  "lolo", surname: "tata", email: "lolo@email.com",  password: "c7ad44cbad762a5da0a452f9e854fdc1e0e7a52a38015f23f3eab1d80b931dd472634dfac71cd34ebc35d16ab7fb8a90c81f975113d6c7538dc69dd8de9077ed", role: "USER", created:"01/01/1999", request_id: "request_dd7175a3-3826-4f3d-8610-619204585f0d", status: "NEW" },
     { name:  "lo123123lo2", surname: "tata", email: "lolo23244@email.com",  password: "c7ad44cbad762a5da0a452f9e854fdc1e0e7a52a38015f23f3eab1d80b931dd472634dfac71cd34ebc35d16ab7fb8a90c81f975113d6c7538dc69dd8de9077ef", role: "USER", created:"01/01/1999", request_id: "request_dd7175a3-3826-4f3d-8610-619204585f0v", status: "NEW" },
     { name:  "lolo3", surname: "tataa", email: "lolo@email.com", password: "c7ad44cbad762a5da0a452f9e854fdc1e0e7a52a38015f23f3eab1d80b931dd472634dfac71cd34ebc35d16ab7fb8a90c81f975113d6c7538dc69dd8de9077eg", role: "ADMIN", created:"01/01/1999", request_id: "request_dd7175a3-3826-4f3d-8610-619204585f0x", status: "NEW" },
]


const AccountRequestManagement = (props) => {

    const [pageIndex, setPageIndex] = useState(0);
    const [pageSize, setPageSize] = useState(20);
    const [sortField, setSortField] = useState('name');
    const [sortDirection, setSortDirection] = useState('asc');

    const [items, setItems] = useState([])
    const [showApproveRequestModal, setShowApproveRequestModal] = useState(false);
    const [showRejectRequestModal, setShowRejectRequestModal] = useState(false);

    const closeApproveRequestModal = () => {setShowApproveRequestModal(false)}
    const closeRejectRequestModal = () => {setShowRejectRequestModal(false)}

    return (
        <div>
            <h1>
                This is User Request account creation webpage. Only admin should be able to reach this page
            </h1>
        </div>
    )
}

export default AccountRequestManagement