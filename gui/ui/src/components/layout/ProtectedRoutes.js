import React from 'react';
import SideNav from './SideNav'
import { Navigate, Outlet, useNavigate } from "react-router-dom";
import {decodeToken, removeToken, setUser} from "../../store/actions/authActions";
import {useDispatch, useSelector, shallowEqual, connect} from "react-redux";


const mapStateToProps = (state) => {
    return {
        user : state.auth
    }
}

export const LoginProtected = connect(mapStateToProps, null)((props) => {

    const dispatch = useDispatch()
    let user = decodeToken()

    if(user && !props.user.is_auth){
        dispatch(setUser(user))
    }


    if(user) {
        return(
            <div className="layout-wrapper">
                <div className="main-side-bar" id="#my_id">
                    <SideNav/>
                </div>
                <div className="main-frame">
                    <div className="router-frame">
                        <div className="inner">
                            <Outlet />
                        </div>
                    </div>
                </div>
            </div>
        )

    }else{
        return(
            <Navigate to="/login/" />
        )
    }

})



export const AdminProtected = (props) => {

    const {role} = useSelector((state) => state.auth, shallowEqual)

    if (role == "Admin"){
        return props.children
    }else{
        return(
            <Navigate to="/user-account/" />
        )
    }

};

