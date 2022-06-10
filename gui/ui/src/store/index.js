import { combineReducers } from "redux";
import { repositoryReducer,
         datasetsReducer,
         datasetPreviewReducer,
         resultReducer} from "./reducers";
import {medicalFolderReducer, medicalFolderPreviewReducer} from "./medicalFolderReducer";
import {modelsReducer} from "./modelsReducer";


/**
 * Combines reducers for the global state
 */
export default combineReducers({
    medicalFolderDataset : medicalFolderReducer,
    repository  : repositoryReducer,
    datasets    : datasetsReducer,
    preview     : datasetPreviewReducer,
    resultModal : resultReducer,
    medicalFolderPreview : medicalFolderPreviewReducer,
    models      : modelsReducer
  })