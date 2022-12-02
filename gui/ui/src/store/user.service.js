import axios from "axios";

import { API_ROOT } from "../constants";


const authHeader = () => {
    const user = JSON.parse(localStorage.getItem('user'));
  
    if (user && user.accessToken) {
      return { Authorization: 'Bearer ' + user.accessToken }; // for Spring Boot back-end
      // return { 'x-access-token': user.accessToken };       // for Node.js Express back-end
    } else {
      return {};
    }
};


const getUserBoard = () => {
  return axios.get(API_ROOT + "user", { headers: authHeader() });
};



const getAdminBoard = () => {
  return axios.get(API_ROOT + "admin", { headers: authHeader() });
};

const UserService = {

  getUserBoard,
  getAdminBoard,
};

export {UserService, authHeader};