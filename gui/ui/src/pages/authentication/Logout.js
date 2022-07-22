import {} from "./useToken";

const Logout = (props) => {
    // logOut user
    props.removeToken();
    console.log("user log out!")
  };


  export default Logout;