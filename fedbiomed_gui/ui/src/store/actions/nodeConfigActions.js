import axios from 'axios'

import {EP_NODE_CONFIG} from '../../constants'
import {
    NODE_CONFIG_ERROR,
    NODE_CONFIG_LOADING,
    NODE_CONFIG_RESET_MESSAGES,
    NODE_CONFIG_SUCCESS,
    NODE_CONFIG_WRITE_CONFLICT,
    NODE_CONFIG_WRITE_ERROR,
    NODE_CONFIG_WRITE_LOADING,
    NODE_CONFIG_WRITE_SUCCESS,
} from './actions'

const getErrorMessage = (error, fallback) => {
    return error?.response?.data?.message || fallback
}

export const fetchNodeConfig = () => {
    return async (dispatch) => {
        dispatch({type: NODE_CONFIG_LOADING, payload: true})

        try {
            const response = await axios.get(EP_NODE_CONFIG)
            dispatch({
                type: NODE_CONFIG_SUCCESS,
                payload: response.data.result || {},
            })
        } catch (error) {
            dispatch({
                type: NODE_CONFIG_ERROR,
                payload: getErrorMessage(
                    error,
                    'Could not get node configuration'
                ),
            })
        } finally {
            dispatch({type: NODE_CONFIG_LOADING, payload: false})
        }
    }
}

export const writeNodeConfigSection = (
    section,
    values,
    baseValues,
    {force = false} = {}
) => {
    return async (dispatch) => {
        dispatch({type: NODE_CONFIG_WRITE_LOADING, payload: true})

        try {
            const response = await axios.patch(EP_NODE_CONFIG, {
                section,
                values,
                base_values: baseValues,
                force,
            })
            const result = response.data.result || {}
            dispatch({
                type: NODE_CONFIG_WRITE_SUCCESS,
                payload: {
                    section: result.section,
                    values: result.values || {},
                    nodeState: result.node_state,
                    requiresRestart: result.requires_restart,
                    configModifiedAfterStartup: (
                        result.config_modified_after_startup
                    ),
                    configStartupCheckMessage: (
                        result.config_startup_check_message
                    ),
                    message: response.data.message,
                },
            })

            return result
        } catch (error) {
            if (error?.response?.status === 409) {
                dispatch({
                    type: NODE_CONFIG_WRITE_CONFLICT,
                    payload: {
                        message: getErrorMessage(
                            error,
                            'Configuration file has been modified'
                        ),
                        result: error.response.data.result || {},
                    },
                })
                return {conflict: true, result: error.response.data.result || {}}
            }

            dispatch({
                type: NODE_CONFIG_WRITE_ERROR,
                payload: getErrorMessage(
                    error,
                    'Could not update node configuration'
                ),
            })
            return null
        } finally {
            dispatch({type: NODE_CONFIG_WRITE_LOADING, payload: false})
        }
    }
}

export const resetNodeConfigMessages = () => ({
    type: NODE_CONFIG_RESET_MESSAGES,
})
