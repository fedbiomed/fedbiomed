import React from 'react'
import SideNav from './SideNav'
import { Navigate, Outlet } from "react-router-dom";
import {decodeToken, setUser, removeToken} from "../../store/actions/authActions";
import {useDispatch, useSelector} from "react-redux";

export const LogUserLayout = (props) => {

    const accessToken= sessionStorage.getItem("accessToken");
    const isAuthenticated = !accessToken && accessToken!== "" && accessToken!== undefined ? false : true
    var {is_auth} = useSelector((state) => state.auth)
    const dispatch = useDispatch()


    // Saves user info into global state

    React.useEffect( () => {
        console.log(is_auth)
        if(is_auth){

            dispatch(setUser(decodeToken()))
            //is_auth = true
        }else{
            //removeToken()
            
        }
    }, [is_auth])


    if(isAuthenticated) {
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

}

export default LogUserLayout