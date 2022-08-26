import { combineReducers } from "redux";
import { repositoryReducer,
         datasetsReducer,
         datasetPreviewReducer,
         resultReducer} from "./reducers";
import {medicalFolderReducer, medicalFolderPreviewReducer} from "./medicalFolderReducer";
import {modelsReducer} from "./modelsReducer";
import { authReducer } from "./authReducer";
import {usersReducer} from "./userManagementReducers";
import { accountRequestReducer } from "./accountRequestReducer";


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
    models      : modelsReducer,
    auth    : authReducer,
    users_   : usersReducer,
    users       : accountRequestReducer,
  })