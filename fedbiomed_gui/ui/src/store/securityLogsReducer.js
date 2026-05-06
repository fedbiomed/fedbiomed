import {
  SECURITY_LOGS_SET_FACETS,
  SECURITY_LOGS_SET_ITEMS,
  SECURITY_LOGS_SET_LAST_BATCH_SIZE,
  SECURITY_LOGS_SET_LOADING,
} from './actions/actions'

const initialSecurityLogsState = {
  operationOptions: [],
  statusOptions: [],
  researcherOptions: [],
  items: [],
  loading: false,
  lastBatchSize: 0,
}

export const securityLogsReducer = (state = initialSecurityLogsState, action) => {
  switch (action.type) {
    case SECURITY_LOGS_SET_LOADING:
      return {
        ...state,
        loading: Boolean(action.payload),
      }

    case SECURITY_LOGS_SET_ITEMS:
      return {
        ...state,
        items: Array.isArray(action.payload) ? action.payload : [],
      }

    case SECURITY_LOGS_SET_LAST_BATCH_SIZE:
      return {
        ...state,
        lastBatchSize: Number.isFinite(action.payload) ? action.payload : 0,
      }

    case SECURITY_LOGS_SET_FACETS:
      return {
        ...state,
        operationOptions: action.payload?.operationOptions || [],
        statusOptions: action.payload?.statusOptions || [],
        researcherOptions: action.payload?.researcherOptions || [],
      }

    default:
      return state
  }
}
