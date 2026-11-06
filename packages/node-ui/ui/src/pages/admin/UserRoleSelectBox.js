import React from 'react';
import {EuiSuperSelect} from '@elastic/eui'
import {changeUserRole} from "../../store/actions/userManagementActions";
import {useDispatch} from "react-redux"
import {USER_MANAGEMENT_ERROR, USER_MANAGEMENT_SUCCESS_MESSAGE} from "../../store/actions/actions";
import {ROLE} from "../../constants";

const UserRoleSelectBox = (props) => {

    const [selected, setSelected] = React.useState(props.selected)
    const [loading, setLoading] = React.useState(false)
    const dispatch = useDispatch()

    const options = [
        {
            value: 1,
            inputDisplay: "Admin"
        },
        {
            value: 2,
            inputDisplay: "User"
        },
    ]

    const onSelectChange = (value) => {

        setLoading(true)
        changeUserRole(props.userId, value).then(response => {
            let {role, email} = response.data.result
            setSelected(role)
            dispatch({type: USER_MANAGEMENT_SUCCESS_MESSAGE, payload: `User role for ${email} 
            has been changed to "${ROLE[role]}"`})
            setLoading(false)
        }).catch(error => {
            dispatch({type: USER_MANAGEMENT_ERROR, payload: error.response.data.message })
            setLoading(false)
        }).catch(error => {
            dispatch({type: USER_MANAGEMENT_ERROR, payload: "Unknown error occurred while changing user role" })
            setLoading(false)
        })
    }


    return (
        <EuiSuperSelect
            isLoading={loading}
            valueOfSelected={selected}
            onChange={onSelectChange}
            options={options}
            fullWidth={true}
            style={{minWidth: 85}}
        />
    );
};

export default UserRoleSelectBox;