import { combineReducers } from "redux";
import { repositoryReducer,
         datasetsReducer,
         datasetPreviewReducer,
         resultReducer} from "./reducers";
import {bidsReducer} from "./bidsReducer";


export default combineReducers({
    bidsDataset : bidsReducer,
    repository  : repositoryReducer,
    datasets    : datasetsReducer,
    preview     : datasetPreviewReducer,
    resultModal : resultReducer
    
  })