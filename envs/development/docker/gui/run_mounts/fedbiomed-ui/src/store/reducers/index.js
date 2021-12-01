import { combineReducers } from "redux";
import { repositoryReducer } from "./repository";

export default combineReducers({
    repository : repositoryReducer,
  })