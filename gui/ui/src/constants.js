
// API Endpoints
export const EP_DATASET_PREVIEW = '/api/datasets/preview'
export const EP_DATASETS_LIST   = '/api/datasets/list'
export const EP_DATASET_REMOVE   = '/api/datasets/remove'
export const EP_REPOSITORY_LIST = '/api/repository/list'
export const EP_DATASET_UPDATE  = '/api/datasets/update'
export const EP_DATASET_ADD     = '/api/datasets/add'
export const EP_DEFAULT_DATASET_ADD     = '/api/datasets/add-default-dataset'
export const EP_CONFIG_NODE_ENVIRON = '/api/config/node-environ'

// Messages
export const DATA_NOTFOUND = 'There is no data has been found for the dataset. It might be deleted'

// Form Handler
export const ADD_DATASET_ERROR_MESSAGES = {
    0 : { key: 'name', message: 'Dataset name is a required field'},
    1 : { key: 'type', message: 'Please select data type.'},
    2 : { key: 'path', message: 'Please enter at least one tags for the dataset'},
    3 : { key: 'tags', message: 'No path is provided'},
    4 : { key: 'desc', message: 'Please enter a description for dataset'}
}