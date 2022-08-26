
// API Endpoints
export const API_ROOT                   = '/api'
export const EP_DATASET_PREVIEW         = '/api/datasets/preview'
export const EP_DATASETS_LIST           = '/api/datasets/list'
export const EP_DATASET_REMOVE          = '/api/datasets/remove'
export const EP_REPOSITORY_LIST         = '/api/repository/list'
export const EP_DATASET_UPDATE          = '/api/datasets/update'
export const EP_DATASET_ADD             = '/api/datasets/add'
export const EP_DEFAULT_DATASET_ADD     = '/api/datasets/add-default-dataset'
export const EP_CONFIG_NODE_ENVIRON     = '/api/config/node-environ'
export const EP_LOAD_CSV_DATA           = '/api/datasets/get-csv-data'

// Authentication endpoints
export const EP_LOGIN                   = '/api/token/login'
export const EP_AUTH                   = '/api/token/auth'
export const EP_REFRESH                 = '/api/token/refresh'
export const EP_LOGOUT                  = '/api/token/remove'
export const EP_REGISTER                = '/api/register'
export const EP_UPDATE_PASSWORD         = '/api/update-password'

// Authentication actions
export const LOGIN                      = 'LOGIN'
export const REGISTER                   = 'REGISTER'

// Temporary endpoints
export const EP_PROTECTED               = '/api/protected'
export const EP_ADMIN                   = '/api/admin'

// Admin endpoints
export const EP_REQUESTS_LIST           = '/api/admin/requests/list'
export const EP_REQUEST_APPROVE         = '/api/admin/requests/approve'
export const EP_REQUEST_REJECT          = '/api/admin/requests/reject'
export const EP_LIST_USERS              = '/api/admin/users/list'
export const EP_REMOVE_USER             = '/api/admin/users/remove'
export const EP_CREATE_USER             = '/api/admin/users/create'

// MedicalFolder Dataset Endpoints
export const EP_VALIDATE_MEDICAL_FOLDER_ROOT    = '/api/datasets/medical-folder-dataset/validate-root'
export const EP_VALIDATE_REFERENCE_COLUMN       = '/api/datasets/medical-folder-dataset/validate-reference-column'
export const EP_ADD_MEDICAL_FOLDER_DATASET      = '/api/datasets/medical-folder-dataset/add'
export const EP_PREVIEW_MEDICAL_FOLDER_DATASET  = '/api/datasets/medical-folder-dataset/preview'

//Models
export const EP_LIST_MODELS     = '/api/model/list'
export const EP_APPROVE_MODEL   = '/api/model/approve'
export const EP_REJECT_MODEL    = '/api/model/reject'
export const EP_DELETE_MODEL    = '/api/model/delete'
export const EP_PREVIEW_MODEL   = '/api/model/preview'

// Messages
export const DATA_NOTFOUND = 'There is no data found for the dataset. It might be deleted'

// Form Handler
export const ADD_DATASET_ERROR_MESSAGES = {
    0 : { key: 'name', message: 'Dataset name is a required field'},
    1 : { key: 'type', message: 'Please select data type'},
    2 : { key: 'path', message: 'Please select data file'},
    3 : { key: 'tags', message: 'Please enter at least one tag for the dataset'},
    4 : { key: 'desc', message: 'Please enter a description for dataset'}
}

//Allowed file extensions for data loader
export const ALLOWED_EXTENSIONS = ['.csv', '.txt']

// role for authentication (User or admin)
export const ROLE = {1: 'Admin', 2: 'User'}
