import axios from 'axios'

import {
    EP_CERTIFICATES,
    EP_CERTIFICATES_CONNECTION,
    EP_CERTIFICATES_EXPORT,
    EP_CERTIFICATES_GENERATE,
    EP_CERTIFICATES_REPLACE,
    EP_CERTIFICATES_STATUS,
} from '../../constants'
import {
    CERTIFICATES_CONNECTION_ERROR,
    CERTIFICATES_CONNECTION_SUCCESS,
    CERTIFICATES_ERROR,
    CERTIFICATES_RESET_MESSAGES,
    CERTIFICATES_SUCCESS,
    CERTIFICATES_WRITE_ERROR,
    CERTIFICATES_WRITE_LOADING,
    CERTIFICATES_WRITE_SUCCESS,
} from './actions'

const getErrorMessage = (error, fallback) => {
    return error?.response?.data?.message || fallback
}

/**
 * Reads the node's mutual-TLS posture: its own certificate, what it has
 * registered, and what would stop it from starting.
 */
export const fetchCertificateStatus = () => {
    return async (dispatch) => {
        try {
            const response = await axios.get(EP_CERTIFICATES_STATUS)
            dispatch({
                type: CERTIFICATES_SUCCESS,
                payload: response.data.result,
            })
        } catch (error) {
            dispatch({
                type: CERTIFICATES_ERROR,
                payload: getErrorMessage(
                    error,
                    'Could not get the certificate status'
                ),
            })
        }
    }
}

/** Reads the connection state the node recorded, and its recent history. */
export const fetchConnectionState = () => {
    return async (dispatch) => {
        try {
            const response = await axios.get(EP_CERTIFICATES_CONNECTION)
            dispatch({
                type: CERTIFICATES_CONNECTION_SUCCESS,
                payload: response.data.result,
            })
        } catch (error) {
            dispatch({
                type: CERTIFICATES_CONNECTION_ERROR,
                payload: getErrorMessage(
                    error,
                    'Could not get the researcher connection state'
                ),
            })
        }
    }
}

/**
 * Registers a certificate received from another component.
 *
 * `upsert` replaces an existing registration of the same component, which the
 * user confirms once the conflict has been reported. `componentId` is needed
 * only for a certificate that carries no component id of its own.
 */
export const registerCertificate = (
    certificate,
    {upsert = false, componentId = null} = {}
) => {
    return async (dispatch) => {
        dispatch({type: CERTIFICATES_WRITE_LOADING, payload: true})

        try {
            const response = await axios.post(EP_CERTIFICATES, {
                certificate,
                upsert,
                component_id: componentId,
            })
            dispatch({
                type: CERTIFICATES_WRITE_SUCCESS,
                payload: response.data.message,
            })
            await dispatch(fetchCertificateStatus())

            return true
        } catch (error) {
            dispatch({
                type: CERTIFICATES_WRITE_ERROR,
                payload: getErrorMessage(
                    error,
                    'Could not register the certificate'
                ),
            })

            return false
        } finally {
            dispatch({type: CERTIFICATES_WRITE_LOADING, payload: false})
        }
    }
}

/** Removes a component's certificate from the node's registry. */
export const deleteCertificate = (componentId) => {
    return async (dispatch) => {
        dispatch({type: CERTIFICATES_WRITE_LOADING, payload: true})

        try {
            const response = await axios.delete(
                `${EP_CERTIFICATES}/${encodeURIComponent(componentId)}`
            )
            dispatch({
                type: CERTIFICATES_WRITE_SUCCESS,
                payload: response.data.message,
            })
            await dispatch(fetchCertificateStatus())
        } catch (error) {
            dispatch({
                type: CERTIFICATES_WRITE_ERROR,
                payload: getErrorMessage(
                    error,
                    'Could not delete the certificate'
                ),
            })
        } finally {
            dispatch({type: CERTIFICATES_WRITE_LOADING, payload: false})
        }
    }
}

/**
 * Runs a write that replaces the node's own certificate pair.
 *
 * Both ways of replacing it report what they did the same way, and the
 * refreshed status is pulled once the write lands.
 */
const writeOwnCertificate = (request, fallbackMessage) => {
    return async (dispatch) => {
        dispatch({type: CERTIFICATES_WRITE_LOADING, payload: true})

        try {
            const response = await request()
            dispatch({
                type: CERTIFICATES_WRITE_SUCCESS,
                payload: response.data.message,
            })
            await dispatch(fetchCertificateStatus())

            return true
        } catch (error) {
            dispatch({
                type: CERTIFICATES_WRITE_ERROR,
                payload: getErrorMessage(error, fallbackMessage),
            })

            return false
        } finally {
            dispatch({type: CERTIFICATES_WRITE_LOADING, payload: false})
        }
    }
}

/** Issues this node a fresh certificate and private key. */
export const generateOwnCertificate = () => writeOwnCertificate(
    () => axios.post(EP_CERTIFICATES_GENERATE, {}),
    'Could not generate a new certificate'
)

/**
 * Replaces this node's certificate and private key with a supplied pair.
 *
 * Both parts are required: the server validates them together and refuses the
 * write outright rather than leaving the node with a pair it cannot serve.
 */
export const replaceOwnCertificate = (certificate, privateKey) => (
    writeOwnCertificate(
        () => axios.post(EP_CERTIFICATES_REPLACE, {
            certificate,
            private_key: privateKey,
        }),
        'Could not replace the certificate'
    )
)

/**
 * Downloads this node's certificate, to be shared with the other components.
 *
 * The public certificate only; the private key never leaves the node.
 */
export const downloadOwnCertificate = () => {
    return async (dispatch) => {
        try {
            const response = await axios.get(EP_CERTIFICATES_EXPORT)
            const {certificate, filename} = response.data.result

            const url = window.URL.createObjectURL(
                new Blob([certificate], {type: 'application/x-pem-file'})
            )
            const link = document.createElement('a')
            link.href = url
            link.setAttribute('download', filename || 'certificate.pem')
            document.body.appendChild(link)
            link.click()
            link.remove()
            window.URL.revokeObjectURL(url)
        } catch (error) {
            dispatch({
                type: 'ERROR_MODAL',
                payload: getErrorMessage(
                    error,
                    'Could not read the node certificate'
                ),
            })
        }
    }
}

export const resetCertificateMessages = () => {
    return {type: CERTIFICATES_RESET_MESSAGES}
}
