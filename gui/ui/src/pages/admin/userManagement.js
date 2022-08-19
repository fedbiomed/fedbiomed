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
  } from '@elastic/eui';
  import { Link } from 'react-router-dom';

  // see https://elastic.github.io/eui/#/tabular-content/data-grid


const UserManagement = (props) => {

    const [isadmin, setIsAdmin] = useState(false)
    const gridRef = createRef();
    const DataContext = createContext();
    const raw_data = [];

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
                is_connected: "connected"
                
            })
        }
    
    console.log(raw_data)
    // columns contains scheme for desigining grid
    const columns = [
        {
          id: 'name',
          displayAsText: 'Name',
          defaultSortDirection: 'asc',
          initialWidth: 130,
          actions: { showMoveLeft: false, showMoveRight: false },
        },
        {
            id: 'surname',
            displayAsText: 'Surname',
            defaultSortDirection: 'asc',
            initialWidth: 130,
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
            id: 'is_connected',
            displayAsText: 'currently connected',
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
                    <DataContext.Provider >
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
                        rowCount={15}
                        ref={gridRef}
                    />
                    </DataContext.Provider>
            </div>
        </Fragment>

    )
}

export default UserManagement;