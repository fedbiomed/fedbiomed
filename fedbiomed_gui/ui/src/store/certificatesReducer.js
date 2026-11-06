import {
    CERTIFICATES_CONNECTION_ERROR,
    CERTIFICATES_CONNECTION_SUCCESS,
    CERTIFICATES_ERROR,
    CERTIFICATES_SUCCESS,
    CERTIFICATES_WRITE_LOADING,
} from './actions/actions'

const initialCertificatesState = {
    status: null,
    connection: null,
    writing: false,
    error: null,
    connectionError: null,
}

export const certificatesReducer = (
    state = initialCertificatesState,
    action
) => {
    switch (action.type) {
        case CERTIFICATES_SUCCESS:
            return {...state, status: action.payload, error: null}

        case CERTIFICATES_ERROR:
            return {...state, error: action.payload}

        case CERTIFICATES_CONNECTION_SUCCESS:
            return {...state, connection: action.payload, connectionError: null}

        case CERTIFICATES_CONNECTION_ERROR:
            return {...state, connectionError: action.payload}

        case CERTIFICATES_WRITE_LOADING:
            return {...state, writing: Boolean(action.payload)}

        default:
            return state
    }
}
