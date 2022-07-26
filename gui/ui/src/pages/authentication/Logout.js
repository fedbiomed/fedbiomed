import useToken from "./useToken";
import Login from "./Login";

const Logout = (props) => {
    // logOut user
    const { accessToken, removeToken, setToken } = useToken();

    //removeToken();
    console.log("user log out!")
    

  
  };


  export default Logout;