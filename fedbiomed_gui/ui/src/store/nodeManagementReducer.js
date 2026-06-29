import {
    NODE_ACTION_ERROR,
    NODE_ACTION_LOADING,
    NODE_LOG_FILES_ERROR,
    NODE_LOG_FILES_LOADING,
    NODE_LOG_FILES_SUCCESS,
    NODE_LOGS_ERROR,
    NODE_LOGS_LOADING,
    NODE_LOGS_SUCCESS,
    NODE_PROCESS_STATE_ERROR,
    NODE_PROCESS_STATE_LOADING,
    NODE_PROCESS_STATE_SUCCESS,
} from './actions/actions'

const initialNodeManagementState = {
    processState: null,
    loading: false,
    actionLoading: null,
    actionError: null,
    processStateError: null,
    lastRefresh: null,
    logItems: [],
    logLoading: false,
    logError: null,
    logLastBatchSize: 0,
    logLastRefresh: null,
    logCursor: null,
    logHasMore: false,
    logFileSize: 0,
    logFiles: [],
    logFilesLoading: false,
    logFilesError: null,
}

export const nodeManagementReducer = (
    state = initialNodeManagementState,
    action
) => {
    switch (action.type) {
        case NODE_PROCESS_STATE_LOADING:
            return {
                ...state,
                loading: Boolean(action.payload),
                processStateError: action.payload
                    ? null
                    : state.processStateError,
            }

        case NODE_PROCESS_STATE_SUCCESS:
            return {
                ...state,
                processState: action.payload.processState,
                processStateError: null,
                lastRefresh: action.payload.markRefresh
                    ? action.payload.lastRefresh
                    : state.lastRefresh,
            }

        case NODE_PROCESS_STATE_ERROR:
            return {
                ...state,
                processState: null,
                processStateError: action.payload,
            }

        case NODE_ACTION_LOADING:
            return {
                ...state,
                actionLoading: action.payload,
                actionError: action.payload ? null : state.actionError,
            }

        case NODE_ACTION_ERROR:
            return {
                ...state,
                actionError: action.payload,
            }

        case NODE_LOGS_LOADING:
            return {
                ...state,
                logLoading: Boolean(action.payload),
                logError: action.payload ? null : state.logError,
            }

        case NODE_LOGS_SUCCESS:
            return {
                ...state,
                logItems: action.payload.mode === 'prepend'
                    ? [...action.payload.items, ...state.logItems]
                    : action.payload.items,
                logLastBatchSize: action.payload.lastBatchSize,
                logLastRefresh: action.payload.lastRefresh,
                logCursor: action.payload.cursor,
                logHasMore: action.payload.hasMore,
                logFileSize: action.payload.fileSize,
                logError: null,
            }

        case NODE_LOGS_ERROR:
            return {
                ...state,
                logItems: [],
                logLastBatchSize: 0,
                logCursor: null,
                logHasMore: false,
                logFileSize: 0,
                logError: action.payload,
            }

        case NODE_LOG_FILES_LOADING:
            return {
                ...state,
                logFilesLoading: Boolean(action.payload),
                logFilesError: action.payload ? null : state.logFilesError,
            }

        case NODE_LOG_FILES_SUCCESS:
            return {
                ...state,
                logFiles: action.payload,
                logFilesError: null,
            }

        case NODE_LOG_FILES_ERROR:
            return {
                ...state,
                logFiles: [],
                logFilesError: action.payload,
            }

        default:
            return state
    }
}
