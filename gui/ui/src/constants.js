
// API Endpoints
export const EP_DATASET_PREVIEW         = '/api/datasets/preview'
export const EP_DATASETS_LIST           = '/api/datasets/list'
export const EP_DATASET_REMOVE          = '/api/datasets/remove'
export const EP_REPOSITORY_LIST         = '/api/repository/list'
export const EP_DATASET_UPDATE          = '/api/datasets/update'
export const EP_DATASET_ADD             = '/api/datasets/add'
export const EP_DEFAULT_DATASET_ADD     = '/api/datasets/add-default-dataset'
export const EP_CONFIG_NODE_ENVIRON     = '/api/config/node-environ'
export const EP_LOAD_CSV_DATA           = '/api/datasets/get-csv-data'

// DataLoadingPlan Endpoints
export const EP_LIST_DATA_LOADING_PLANS         = '/api/datasets/list-dlps'
export const EP_READ_DATA_LOADING_PLAN          = '/api/datasets/read-dlp'
export const EP_ADD_DATA_LOADING_PLAN           = '/api/datasets/medical-folder-dataset/add-dlp'
export const EP_DELETE_DATA_LOADING_PLAN        = '/api/datasets/medical-folder-dataset/delete-dlp'

// MedicalFolder Dataset Endpoints
export const EP_VALIDATE_MEDICAL_FOLDER_ROOT    = '/api/datasets/medical-folder-dataset/validate-root'
export const EP_VALIDATE_REFERENCE_COLUMN       = '/api/datasets/medical-folder-dataset/validate-reference-column'
export const EP_VALIDATE_SUBJECTS_ALL_MODALITIES = '/api/datasets/medical-folder-dataset/validate-all-modalities'
export const EP_ADD_MEDICAL_FOLDER_DATASET      = '/api/datasets/medical-folder-dataset/add'
export const EP_PREVIEW_MEDICAL_FOLDER_DATASET  = '/api/datasets/medical-folder-dataset/preview'
export const EP_DEFAULT_MODALITY_NAMES          = '/api/datasets/medical-folder-dataset/default-modalities'

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

