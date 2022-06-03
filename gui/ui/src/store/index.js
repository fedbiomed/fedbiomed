import { combineReducers } from "redux";
import { repositoryReducer,
         datasetsReducer,
         datasetPreviewReducer,
         resultReducer} from "./reducers";
import {bidsReducer, bidsPreviewReducer} from "./bidsReducer";
import {modelsReducer} from "./modelsReducer";


/**
 * Combines reducers for the global state
 */
export default combineReducers({
    bidsDataset : bidsReducer,
    repository  : repositoryReducer,
    datasets    : datasetsReducer,
    preview     : datasetPreviewReducer,
    resultModal : resultReducer,
    bidsPreview : bidsPreviewReducer,
    models      : modelsReducer
  })