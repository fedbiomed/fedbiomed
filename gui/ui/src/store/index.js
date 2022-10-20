import { combineReducers } from "redux";
import { repositoryReducer,
         datasetsReducer,
         datasetPreviewReducer,
         resultReducer} from "./reducers";
import {medicalFolderReducer, medicalFolderPreviewReducer} from "./medicalFolderReducer";
import {trainingPlansReducer} from "./trainingPlansReducer";
import {dataLoadingPlanReducer} from "./dataLoadingPlanReducer";
import { authReducer } from "./authReducer";
import {usersReducer} from "./userManagementReducers";
import {accountRequestReducer } from "./accountRequestReducer";


/**
 * Combines reducers for the global state
 */
export default combineReducers({
    medicalFolderDataset    : medicalFolderReducer,
    repository              : repositoryReducer,
    datasets                : datasetsReducer,
    preview                 : datasetPreviewReducer,
    resultModal             : resultReducer,
    medicalFolderPreview    : medicalFolderPreviewReducer,
    training_plans          : trainingPlansReducer,
    dataLoadingPlan         : dataLoadingPlanReducer,
    auth                    : authReducer,
    users                   : usersReducer,
    user_requests           : accountRequestReducer,
  })