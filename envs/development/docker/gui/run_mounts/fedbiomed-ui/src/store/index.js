import { combineReducers } from "redux";
import { repositoryReducer, datasetsreducer, datasetPreviewReducer } from "./reducers";

export default combineReducers({
    repository : repositoryReducer,
    datasets   : datasetsreducer,
    preview    : datasetPreviewReducer
    
  })