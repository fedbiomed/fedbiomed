import React from 'react'
import SideNav from './SideNav'
import { Navigate, Outlet } from "react-router-dom";


export const LogUserLayout = (props) => {

    const accessToken= sessionStorage.getItem("accessToken");
    const isAuthenticated = !accessToken && accessToken!== "" && accessToken!== undefined ? false : true



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
            <Navigate to="/login" />
        )
    }

}

                //   {/* If the user is not logged in, redirect towards login page */}
                //   {!accessToken && accessToken!=="" && accessToken!== undefined?  
                //   <Login setToken={setToken} />
                //   :(





// export const ProtectedRoute = ({ component: Component, ...restOfProps }) => {

//     const accessToken= localStorage.getItem("accessToken");
//     const isAuthenticated = !accessToken && accessToken!== "" && accessToken!== undefined ? true : false

//     return (
//       <Route
//         {...restOfProps}
//         render={(props) =>
//           isAuthenticated ? <Component {...props} /> : <Navigate to="/login" />
//         }
//       />
//     );
//   }


export default LogUserLayout