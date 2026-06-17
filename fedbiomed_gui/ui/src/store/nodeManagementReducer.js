import {
    NODE_ACTION_ERROR,
    NODE_ACTION_LOADING,
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

        default:
            return state
    }
}
