import React from 'react';
import  {TableInfo, TableCol, TableRow} from "../../components/common/Tables";
import { readToken } from '../../store/actions/tokenFunc';



const UserAccount = () => {

    return (
        <React.Fragment> 
            <div>
                <h2>
                    User account
                </h2>
                <h4>
                    This page contains user account info
                </h4>
                <TableInfo info={readToken()} mode={false}/>
                <TableRow key={'lolo'}>
                    <TableCol>
                    <TableInfo info={{'Password': 'Change'}} mode={false}/>
                    </TableCol>
                </TableRow>
                
            </div>
            
        </React.Fragment> 
        
    )
}

export default UserAccount