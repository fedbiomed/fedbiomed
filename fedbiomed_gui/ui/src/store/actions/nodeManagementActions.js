import axios from 'axios'

import {
    EP_NODE_LOG_FILES,
    EP_NODE_LOGS,
    EP_NODE_LOGS_DOWNLOAD,
    EP_NODE_PROCESS_STATE,
    EP_NODE_RESTART,
    EP_NODE_START,
    EP_NODE_STOP,
} from '../../constants'
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
} from './actions'

const nodeActionEndpoints = {
    restart: EP_NODE_RESTART,
    start: EP_NODE_START,
    stop: EP_NODE_STOP,
}

const getErrorMessage = (error, fallback) => {
    return error?.response?.data?.message || fallback
}

const buildLogParams = ({
    cursor,
    fileName,
    pageSize,
}) => {
    const params = {
        page_size: pageSize,
    }

    if (fileName) {
        params.file = fileName
    }

    if (cursor !== null && cursor !== undefined) {
        params.cursor = cursor
    }

    return params
}

export const fetchNodeProcessState = ({markRefresh = false} = {}) => {
    return async (dispatch) => {
        dispatch({type: NODE_PROCESS_STATE_LOADING, payload: true})

        try {
            const response = await axios.get(EP_NODE_PROCESS_STATE)
            dispatch({
                type: NODE_PROCESS_STATE_SUCCESS,
                payload: {
                    processState: response.data.result,
                    lastRefresh: markRefresh ? new Date().toISOString() : null,
                    markRefresh,
                },
            })
        } catch (error) {
            dispatch({
                type: NODE_PROCESS_STATE_ERROR,
                payload: getErrorMessage(
                    error,
                    'Could not get node process state'
                ),
            })
        } finally {
            dispatch({type: NODE_PROCESS_STATE_LOADING, payload: false})
        }
    }
}

export const executeNodeAction = (action, nodeArgs) => {
    return async (dispatch) => {
        const endpoint = nodeActionEndpoints[action]
        dispatch({type: NODE_ACTION_LOADING, payload: action})

        try {
            if (action === 'stop') {
                await axios.post(endpoint)
            } else {
                await axios.post(endpoint, {
                    ...nodeArgs,
                    gpu_num: Math.max(0, Number(nodeArgs.gpu_num) || 0),
                })
            }
            await dispatch(fetchNodeProcessState({markRefresh: true}))
        } catch (error) {
            dispatch({
                type: NODE_ACTION_ERROR,
                payload: getErrorMessage(
                    error,
                    `Could not ${action} node process`
                ),
            })
        } finally {
            dispatch({type: NODE_ACTION_LOADING, payload: null})
        }
    }
}

export const fetchNodeLogs = (args) => {
    return async (dispatch) => {
        dispatch({type: NODE_LOGS_LOADING, payload: true})

        try {
            const response = await axios.get(EP_NODE_LOGS, {
                params: buildLogParams(args),
            })
            const result = response.data.result || {}
            const items = Array.isArray(result.items) ? result.items : []
            dispatch({
                type: NODE_LOGS_SUCCESS,
                payload: {
                    items,
                    lastBatchSize: items.length,
                    lastRefresh: new Date().toISOString(),
                    cursor: result.next_cursor,
                    hasMore: Boolean(result.has_more),
                    fileSize: result.file_size,
                    mode: args?.mode || 'reset',
                },
            })
        } catch (error) {
            dispatch({
                type: NODE_LOGS_ERROR,
                payload: getErrorMessage(
                    error,
                    'Could not get node application logs'
                ),
            })
        } finally {
            dispatch({type: NODE_LOGS_LOADING, payload: false})
        }
    }
}

export const fetchNodeLogFiles = () => {
    return async (dispatch) => {
        dispatch({type: NODE_LOG_FILES_LOADING, payload: true})

        try {
            const response = await axios.get(EP_NODE_LOG_FILES)
            const result = response.data.result || {}
            dispatch({
                type: NODE_LOG_FILES_SUCCESS,
                payload: Array.isArray(result.files) ? result.files : [],
            })
        } catch (error) {
            dispatch({
                type: NODE_LOG_FILES_ERROR,
                payload: getErrorMessage(
                    error,
                    'Could not get node application log files'
                ),
            })
        } finally {
            dispatch({type: NODE_LOG_FILES_LOADING, payload: false})
        }
    }
}

export const downloadNodeLogFile = (fileName) => {
    return async (dispatch) => {
        try {
            const response = await axios.get(EP_NODE_LOGS_DOWNLOAD, {
                params: fileName ? {file: fileName} : {},
                responseType: 'blob',
            })
            const url = window.URL.createObjectURL(new Blob([response.data]))
            const link = document.createElement('a')
            link.href = url
            link.setAttribute('download', fileName || 'application.log')
            document.body.appendChild(link)
            link.click()
            link.remove()
            window.URL.revokeObjectURL(url)
        } catch (error) {
            dispatch({
                type: 'ERROR_MODAL',
                payload: getErrorMessage(
                    error,
                    'Could not download node application log'
                ),
            })
        }
    }
}
