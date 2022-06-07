import { combineReducers } from "redux";
import { repositoryReducer,
         datasetsReducer,
         datasetPreviewReducer,
         resultReducer} from "./reducers";
import {medicalFolderReducer, medicalFolderPreviewReducer} from "./medicalFolderReducer";


export default combineReducers({
    medicalFolderDataset : medicalFolderReducer,
    repository  : repositoryReducer,
    datasets    : datasetsReducer,
    preview     : datasetPreviewReducer,
    resultModal : resultReducer,
    medicalFolderPreview : medicalFolderPreviewReducer
  })