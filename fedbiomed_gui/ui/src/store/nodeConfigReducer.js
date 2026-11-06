import {
    NODE_CONFIG_ERROR,
    NODE_CONFIG_LOADING,
    NODE_CONFIG_RESET_MESSAGES,
    NODE_CONFIG_SUCCESS,
    NODE_CONFIG_WRITE_CONFLICT,
    NODE_CONFIG_WRITE_ERROR,
    NODE_CONFIG_WRITE_LOADING,
    NODE_CONFIG_WRITE_SUCCESS,
} from './actions/actions'

const initialNodeConfigState = {
    node: null,
    sections: {},
    loading: false,
    writing: false,
    error: null,
    writeError: null,
    writeConflict: null,
    successMessage: null,
    nodeState: null,
    requiresRestart: false,
    configModifiedAfterStartup: false,
    configStartupCheckMessage: null,
}

const applyWrittenValues = (sections, writtenSections = {}) => {
    // Fold the values confirmed by the backend back into the loaded sections so
    // they become the new baseline for change detection.
    return Object.keys(writtenSections).reduce((updated, section) => ({
        ...updated,
        [section]: {
            ...(sections[section] || {}),
            fields: Object.keys(writtenSections[section]).reduce(
                (fields, key) => ({
                    ...fields,
                    [key]: {
                        ...(sections[section]?.fields?.[key] || {}),
                        value: writtenSections[section][key],
                    },
                }),
                sections[section]?.fields || {}
            ),
        },
    }), sections)
}

export const nodeConfigReducer = (
    state = initialNodeConfigState,
    action
) => {
    switch (action.type) {
        case NODE_CONFIG_LOADING:
            return {
                ...state,
                loading: Boolean(action.payload),
                error: action.payload ? null : state.error,
            }

        case NODE_CONFIG_SUCCESS:
            return {
                ...state,
                sections: action.payload.sections || {},
                nodeState: action.payload.node_state || null,
                configModifiedAfterStartup: Boolean(
                    action.payload.config_modified_after_startup
                ),
                configStartupCheckMessage: (
                    action.payload.config_startup_check_message || null
                ),
                requiresRestart: false,
                error: null,
                writeConflict: null,
            }

        case NODE_CONFIG_ERROR:
            return {
                ...state,
                error: action.payload,
            }

        case NODE_CONFIG_WRITE_LOADING:
            return {
                ...state,
                writing: Boolean(action.payload),
                writeError: action.payload ? null : state.writeError,
            }

        case NODE_CONFIG_WRITE_SUCCESS:
            return {
                ...state,
                sections: applyWrittenValues(
                    state.sections,
                    action.payload.sections
                ),
                nodeState: action.payload.nodeState || state.nodeState,
                requiresRestart: Boolean(action.payload.requiresRestart),
                configModifiedAfterStartup: Boolean(
                    action.payload.configModifiedAfterStartup
                ),
                configStartupCheckMessage: (
                    action.payload.configStartupCheckMessage || null
                ),
                writeError: null,
                writeConflict: null,
                successMessage: action.payload.message,
            }

        case NODE_CONFIG_WRITE_ERROR:
            return {
                ...state,
                writeError: action.payload,
                writeConflict: null,
                successMessage: null,
            }

        case NODE_CONFIG_WRITE_CONFLICT:
            return {
                ...state,
                writeError: action.payload.message,
                writeConflict: action.payload.result,
                successMessage: null,
            }

        case NODE_CONFIG_RESET_MESSAGES:
            return {
                ...state,
                error: null,
                writeError: null,
                writeConflict: null,
                successMessage: null,
            }

        default:
            return state
    }
}
