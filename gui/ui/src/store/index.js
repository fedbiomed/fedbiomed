import { combineReducers } from "redux";
import { repositoryReducer,
         datasetsReducer,
         datasetPreviewReducer,
         resultReducer} from "./reducers";



export default combineReducers({
    repository : repositoryReducer,
    datasets   : datasetsReducer,
    preview    : datasetPreviewReducer,
    resultModal: resultReducer
    
  })