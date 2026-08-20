import {
    CERTIFICATES_CONNECTION_ERROR,
    CERTIFICATES_CONNECTION_SUCCESS,
    CERTIFICATES_ERROR,
    CERTIFICATES_LOADING,
    CERTIFICATES_RESET_MESSAGES,
    CERTIFICATES_SUCCESS,
    CERTIFICATES_WRITE_ERROR,
    CERTIFICATES_WRITE_LOADING,
    CERTIFICATES_WRITE_SUCCESS,
} from './actions/actions'

const initialCertificatesState = {
    status: null,
    connection: null,
    loading: false,
    writing: false,
    error: null,
    connectionError: null,
    writeError: null,
    successMessage: null,
    requiresRestart: false,
}

export const certificatesReducer = (
    state = initialCertificatesState,
    action
) => {
    switch (action.type) {
        case CERTIFICATES_LOADING:
            return {
                ...state,
                loading: Boolean(action.payload),
                error: action.payload ? null : state.error,
            }

        case CERTIFICATES_SUCCESS:
            return {...state, status: action.payload, error: null}

        case CERTIFICATES_ERROR:
            return {...state, error: action.payload}

        case CERTIFICATES_CONNECTION_SUCCESS:
            return {...state, connection: action.payload, connectionError: null}

        case CERTIFICATES_CONNECTION_ERROR:
            return {...state, connectionError: action.payload}

        case CERTIFICATES_WRITE_LOADING:
            return {
                ...state,
                writing: Boolean(action.payload),
                writeError: action.payload ? null : state.writeError,
                successMessage: action.payload ? null : state.successMessage,
            }

        case CERTIFICATES_WRITE_SUCCESS:
            return {
                ...state,
                successMessage: action.payload.message,
                requiresRestart: action.payload.requiresRestart,
                writeError: null,
            }

        case CERTIFICATES_WRITE_ERROR:
            return {...state, writeError: action.payload, successMessage: null}

        case CERTIFICATES_RESET_MESSAGES:
            return {
                ...state,
                writeError: null,
                successMessage: null,
                requiresRestart: false,
            }

        default:
            return state
    }
}
