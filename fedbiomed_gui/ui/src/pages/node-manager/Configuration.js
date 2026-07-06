import React from 'react'
import {connect} from 'react-redux'
import {
    EuiButton,
    EuiCallOut,
    EuiFieldNumber,
    EuiFieldText,
    EuiForm,
    EuiFormRow,
    EuiIcon,
    EuiModal,
    EuiModalBody,
    EuiModalFooter,
    EuiModalHeader,
    EuiModalHeaderTitle,
    EuiSelect,
    EuiSpacer,
    EuiText,
} from '@elastic/eui'

import {
    fetchNodeConfig,
    resetNodeConfigMessages,
    writeNodeConfigSection,
} from '../../store/actions/nodeConfigActions'
import {
    executeNodeAction,
    fetchNodeProcessState,
} from '../../store/actions/nodeManagementActions'

const labelFor = (key) => key
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')

const securityFieldDescriptions = {
    training_plan_approval: 'Require manual approval before a training plan can run.',
    allow_default_training_plans: (
        'Permit built-in training plans without extra registration.'
    ),
    secure_aggregation: 'Enable secure aggregation for compatible experiments.',
    force_secure_aggregation: (
        'Reject jobs that do not use secure aggregation.'
    ),
    secagg_insecure_validation: (
        'Allow insecure validation mode for secure aggregation setup.'
    ),
    allow_preproc: 'Allow preprocessing steps before model training starts.',
    allow_federated_analytics: (
        'Permit analytics queries that do not train a model.'
    ),
}

const getFieldDescription = (section, key, field) => {
    if (section === 'security' && securityFieldDescriptions[key]) {
        return securityFieldDescriptions[key]
    }

    return field.editable
        ? 'Editable value from config.ini.'
        : 'Read-only value from config.ini.'
}

const sectionIconFor = (section) => {
    switch (section) {
        case 'security':
            return 'lock'
        case 'default':
            return 'node'
        case 'certificate':
            return 'document'
        case 'researcher':
            return 'users'
        case 'syslog':
            return 'console'
        default:
            return 'controlsHorizontal'
    }
}

const sectionMarkFor = (section) => (
    String(section || '?').charAt(0).toUpperCase()
)

const getSectionFields = (sections, section) => (
    sections?.[section]?.fields || {}
)

const normalizeFieldValue = (value, field) => {
    // Keep form values comparable with backend-normalized config values.
    if (!field) {
        return value
    }

    if (field.type === 'boolean') {
        return Boolean(value)
    }

    if (field.type === 'integer') {
        const minValue = Number.isFinite(Number(field.min))
            ? Number(field.min)
            : 0
        return Math.max(
            minValue,
            Number.parseInt(value ?? field.default ?? minValue, 10) || minValue
        )
    }

    if (field.type === 'enum') {
        const options = Array.isArray(field.options) ? field.options : []
        return value || field.default || options[0] || ''
    }

    return value ?? field.default ?? ''
}

const normalizeSectionValues = (values = {}, fields = {}) => {
    return Object.keys(fields).reduce((normalized, key) => ({
        ...normalized,
        [key]: normalizeFieldValue(values[key], fields[key]),
    }), {})
}

const getSectionValues = (fields = {}) => {
    // Extract the last values loaded from config.ini for one section.
    return Object.keys(fields).reduce((values, key) => ({
        ...values,
        [key]: normalizeFieldValue(fields[key].value, fields[key]),
    }), {})
}

const getEditableSectionValues = (current, fields) => {
    const normalizedCurrent = normalizeSectionValues(current, fields)

    // Send every editable value in the section. The backend compares all of
    // them against the file before writing the complete section update.
    return Object.keys(fields).reduce((values, key) => {
        if (!fields[key].editable) {
            return values
        }

        return {
            ...values,
            [key]: normalizedCurrent[key],
        }
    }, {})
}

const areSectionValuesEqual = (first, second, fields) => {
    const normalizedFirst = normalizeSectionValues(first, fields)
    const normalizedSecond = normalizeSectionValues(second, fields)

    return Object.keys(fields).every((key) => (
        !fields[key].editable
        ||
        normalizedFirst[key] === normalizedSecond[key]
    ))
}

const Configuration = ({
    sections,
    nodeState: configNodeState,
    requiresRestart,
    loading,
    writing,
    error,
    writeError,
    writeConflict,
    successMessage,
    configModifiedAfterStartup,
    configStartupCheckMessage,
    processState,
    fetchNodeConfig,
    writeNodeConfigSection,
    resetNodeConfigMessages,
    executeNodeAction,
    fetchNodeProcessState,
    fetchProcessStateOnMount = true,
    embedded = false,
}) => {
    const [activeSection, setActiveSection] = React.useState(null)
    // Draft values are the current form values. They may differ from config.ini
    // until the user saves or resets the section.
    const [draftValues, setDraftValues] = React.useState({})
    // Remember whether a conflict interrupted a Save & Restart action, so an
    // explicit overwrite can continue with restart after the forced write.
    const [restartAfterConflictWrite, setRestartAfterConflictWrite] = (
        React.useState(false)
    )
    const [restartLoading, setRestartLoading] = React.useState(false)
    const sectionNames = React.useMemo(
        () => Object.keys(sections || {}),
        [sections]
    )
    const resolvedSection = activeSection || sectionNames[0]
    const sectionInfo = sections?.[resolvedSection] || {}
    const fields = React.useMemo(
        () => getSectionFields(sections, resolvedSection),
        [sections, resolvedSection]
    )
    // Saved values are the last values loaded from config.ini. They are used as
    // the base for dirty-state detection and backend conflict detection.
    const savedValues = React.useMemo(
        () => getSectionValues(fields),
        [fields]
    )

    React.useEffect(() => {
        fetchNodeConfig()
        if (fetchProcessStateOnMount) {
            fetchNodeProcessState()
        }
    }, [fetchNodeConfig, fetchNodeProcessState, fetchProcessStateOnMount])

    React.useEffect(() => {
        if (!activeSection && sectionNames.length) {
            setActiveSection(sectionNames[0])
        }
    }, [activeSection, sectionNames])

    React.useEffect(() => {
        if (!resolvedSection) {
            return
        }

        setDraftValues(normalizeSectionValues(savedValues, fields))
    }, [fields, resolvedSection, savedValues])

    const updateValue = (key, value) => {
        setDraftValues((current) => ({
            ...current,
            [key]: normalizeFieldValue(value, fields[key]),
        }))
        resetNodeConfigMessages()
    }

    const saveCurrentSection = async ({restart = false, force = false} = {}) => {
        if (!resolvedSection) {
            return
        }

        const normalized = normalizeSectionValues(draftValues, fields)

        // Always send all editable values in the section. For overwrite, only
        // the force flag changes so the backend bypasses conflict checks.
        const values = getEditableSectionValues(normalized, fields)

        // Base values are what the user last loaded. The backend compares them
        // against the file before writing to detect external modifications.
        const baseValues = Object.keys(values).reduce((base, key) => ({
            ...base,
            [key]: savedValues[key],
        }), {})

        const result = await writeNodeConfigSection(
            resolvedSection,
            values,
            baseValues,
            {force}
        )
        if (!result) {
            return
        }

        if (result.conflict) {
            // Stop here. The conflict modal lets the user refresh, cancel, or
            // retry with force before any restart is attempted.
            setRestartAfterConflictWrite(restart)
            return
        }

        setRestartAfterConflictWrite(false)
        if (restart) {
            await restartNode()
        }
    }

    const restartNode = async () => {
        resetNodeConfigMessages()
        setRestartLoading(true)
        try {
            await executeNodeAction('restart', processState?.node_args || {})
            await fetchNodeProcessState({markRefresh: true})
            await fetchNodeConfig()
        } finally {
            setRestartLoading(false)
        }
    }

    const resetChanges = () => {
        setDraftValues(normalizeSectionValues(savedValues, fields))
        resetNodeConfigMessages()
    }

    const refreshConfig = async () => {
        resetNodeConfigMessages()
        await fetchNodeConfig()
    }

    const refreshAfterConflict = async () => {
        setRestartAfterConflictWrite(false)
        await refreshConfig()
    }

    const overwriteAfterConflict = async () => {
        // Retry the same draft write with force=true. If the interrupted action
        // was Save & Restart, restart continues after this write succeeds.
        await saveCurrentSection({
            restart: restartAfterConflictWrite,
            force: true,
        })
    }

    const cancelConflict = () => {
        setRestartAfterConflictWrite(false)
        resetNodeConfigMessages()
    }

    const displayNodeState = String(
        processState?.state || configNodeState || ''
    ).toLowerCase()
    const isRunning = displayNodeState === 'running'
    // Dirty means at least one editable form value differs from the last values
    // loaded from config.ini. It enables 'Save' and 'Save & Restart' buttons in the frontend.
    const dirty = resolvedSection
        && !areSectionValuesEqual(draftValues, savedValues, fields)
    const actionInProgress = writing || restartLoading
    const sectionOptions = sectionNames.map((section) => ({
        value: section,
        text: sections?.[section]?.label || labelFor(section),
    }))
    const scalarFieldKeys = Object.keys(fields).filter((key) => (
        fields[key].type !== 'boolean'
    ))
    const booleanFieldKeys = Object.keys(fields).filter((key) => (
        fields[key].type === 'boolean'
    ))
    const activeSectionHasEditableFields = Object.keys(fields).some((key) => (
        fields[key].editable
    ))
    const conflictItems = Object.keys(writeConflict?.conflicts || {}).map(
        // Flatten backend conflict details to display in the modal.
        (key) => ({
            key,
            ...writeConflict.conflicts[key],
        })
    )

    return (
        <div className={embedded ? 'node-config-page' : 'node-management-page'}>
            <section className="node-management-card node-management-header">
                <div className="node-management-header-top">
                    <div className="node-management-heading">
                        <span className="node-management-heading-icon">
                            <EuiIcon type="controlsHorizontal" size="xl" />
                        </span>
                        <div>
                            <h1>Node Configuration</h1>
                            <p>Node settings from config.ini</p>
                        </div>
                    </div>
                    <div className="node-management-header-actions">
                        <EuiButton
                            iconType="refresh"
                            fill
                            onClick={refreshConfig}
                            isLoading={loading}
                        >
                            Refresh
                        </EuiButton>
                    </div>
                </div>
            </section>

            {error || writeError ? (
                <EuiCallOut
                    color="danger"
                    iconType="alert"
                    title={error || writeError}
                />
            ) : null}

            {successMessage ? (
                <>
                    <EuiSpacer size="m" />
                    <EuiCallOut
                        color="success"
                        iconType="check"
                        title={successMessage}
                    />
                </>
            ) : null}

            {configModifiedAfterStartup ? (
                <>
                    <EuiSpacer size="m" />
                    <EuiCallOut
                        color="warning"
                        iconType="alert"
                        title={
                            'The config file config.ini has been modified after node startup. '
                            + 'The values shown here may not represent the values effective '
                            + 'in the current node process'
                        }
                    />
                </>
            ) : null}

            {configStartupCheckMessage ? (
                <>
                    <EuiSpacer size="m" />
                    <EuiCallOut
                        color="warning"
                        iconType="alert"
                        title={configStartupCheckMessage}
                    />
                </>
            ) : null}

            {requiresRestart ? (
                <>
                    <EuiSpacer size="m" />
                    <EuiCallOut
                        color="warning"
                        iconType="refresh"
                        title="Restart required"
                    >
                        <p>
                            Saved configuration changes will apply after the
                            running node restarts.
                        </p>
                    </EuiCallOut>
                </>
            ) : null}

            <EuiSpacer size="m" />

            <div className="node-config-layout">
                <nav
                    className="node-config-section-nav"
                    aria-label="Configuration sections"
                >
                    {sectionOptions.map((sectionOption) => {
                        const section = sectionOption.value
                        const isActive = section === resolvedSection
                        const hasEditableFields = Object.values(
                            sections?.[section]?.fields || {}
                        ).some((field) => field.editable)

                        return (
                            <button
                                type="button"
                                className={`node-config-section-nav-item ${
                                    isActive ? 'active' : ''
                                } ${hasEditableFields ? '' : 'read-only'}`}
                                key={section}
                                onClick={() => {
                                    setActiveSection(section)
                                    resetNodeConfigMessages()
                                }}
                            >
                                <span className="node-config-section-mark">
                                    {sectionMarkFor(section)}
                                </span>
                                <span>
                                    <strong>{sectionOption.text}</strong>
                                    <small>
                                        {hasEditableFields
                                            ? 'Editable'
                                            : 'Read-only'}
                                    </small>
                                </span>
                            </button>
                        )
                    })}
                </nav>

                <section className="node-management-card node-config-panel">
                    <div className="node-config-panel-header">
                        <div className="node-management-section-heading">
                            <span className="node-management-section-icon">
                                <EuiIcon
                                    type={sectionIconFor(resolvedSection)}
                                    size="l"
                                />
                            </span>
                            <div>
                                <h2>
                                    {sectionInfo.label
                                        || labelFor(
                                            resolvedSection || 'configuration'
                                        )}
                                </h2>
                                <p>
                                    {activeSectionHasEditableFields
                                        ? 'Changes are written to the node configuration file.'
                                        : 'This section is displayed for reference.'}
                                </p>
                            </div>
                        </div>
                        <span className="node-config-status">
                            {String(displayNodeState || 'unknown').toUpperCase()}
                        </span>
                    </div>

                    <div className="node-config-form-area">
                        {resolvedSection ? (
                            <EuiForm
                                className="node-config-form"
                                component="form"
                            >
                                {scalarFieldKeys.length ? (
                                    <div className="node-config-fields-grid">
                                        {scalarFieldKeys.map((key) => {
                                            const field = fields[key]
                                            const label = field.label || labelFor(key)
                                            const disabled = !field.editable
                                            let fieldControl = (
                                                <EuiFieldText
                                                    className="node-config-input"
                                                    disabled={disabled}
                                                    value={draftValues[key]}
                                                    onChange={(event) => (
                                                        updateValue(
                                                            key,
                                                            event.target.value
                                                        )
                                                    )}
                                                />
                                            )

                                            if (field.type === 'integer') {
                                                fieldControl = (
                                                    <EuiFieldNumber
                                                        className="node-config-input"
                                                        disabled={disabled}
                                                        min={field.min ?? 0}
                                                        value={draftValues[key]}
                                                        onChange={(event) => (
                                                            updateValue(
                                                                key,
                                                                event.target.value
                                                            )
                                                        )}
                                                    />
                                                )
                                            }

                                            if (field.type === 'enum') {
                                                fieldControl = (
                                                    <EuiSelect
                                                        className="node-config-input"
                                                        disabled={disabled}
                                                        value={draftValues[key]}
                                                        options={(
                                                            field.options || []
                                                        ).map((option) => ({
                                                            value: option,
                                                            text: option,
                                                        }))}
                                                        onChange={(event) => (
                                                            updateValue(
                                                                key,
                                                                event.target.value
                                                            )
                                                        )}
                                                    />
                                                )
                                            }

                                            return (
                                                <EuiFormRow
                                                    className="node-config-form-row"
                                                    key={key}
                                                    label={label}
                                                >
                                                    {fieldControl}
                                                </EuiFormRow>
                                            )
                                        })}
                                    </div>
                                ) : null}

                                {booleanFieldKeys.length ? (
                                    <div className="node-config-settings-grid">
                                        {booleanFieldKeys.map((key) => {
                                            const field = fields[key]
                                            const label = field.label || labelFor(key)
                                            const disabled = !field.editable
                                            const checked = Boolean(draftValues[key])

                                            return (
                                                <article
                                                    className={`node-config-setting ${
                                                        disabled ? 'disabled' : ''
                                                    }`}
                                                    key={key}
                                                >
                                                    <div>
                                                        <div className="node-config-setting-name">
                                                            {label}
                                                        </div>
                                                        <div className="node-config-setting-help">
                                                            {getFieldDescription(
                                                                resolvedSection,
                                                                key,
                                                                field
                                                            )}
                                                        </div>
                                                    </div>
                                                    <div className="node-config-segmented">
                                                        <button
                                                            type="button"
                                                            className="true"
                                                            aria-pressed={checked}
                                                            disabled={disabled}
                                                            onClick={() => (
                                                                updateValue(
                                                                    key,
                                                                    true
                                                                )
                                                            )}
                                                        >
                                                            True
                                                        </button>
                                                        <button
                                                            type="button"
                                                            className="false"
                                                            aria-pressed={!checked}
                                                            disabled={disabled}
                                                            onClick={() => (
                                                                updateValue(
                                                                    key,
                                                                    false
                                                                )
                                                            )}
                                                        >
                                                            False
                                                        </button>
                                                    </div>
                                                </article>
                                            )
                                        })}
                                    </div>
                                ) : null}
                            </EuiForm>
                        ) : (
                            <EuiText>
                                <p>{loading ? 'Loading configuration...' : ''}</p>
                            </EuiText>
                        )}
                    </div>

                    <div className="node-management-header-control-actions node-config-actions">
                        <EuiButton
                            fill
                            iconType="save"
                            onClick={() => saveCurrentSection()}
                            isLoading={writing}
                            isDisabled={!dirty || actionInProgress}
                        >
                            Save
                        </EuiButton>
                        {isRunning ? (
                            <EuiButton
                                color="warning"
                                fill
                                iconType="refresh"
                                onClick={() => (
                                    dirty
                                        ? saveCurrentSection({restart: true})
                                        : restartNode()
                                )}
                                isLoading={actionInProgress}
                                isDisabled={
                                    (!dirty && !requiresRestart)
                                    || actionInProgress
                                }
                            >
                                {dirty ? 'Save & Restart' : 'Restart'}
                            </EuiButton>
                        ) : null}
                        <EuiButton
                            iconType="cross"
                            onClick={resetChanges}
                            isDisabled={!dirty || actionInProgress}
                        >
                            Reset
                        </EuiButton>
                    </div>
                </section>
            </div>

            {writeConflict ? (
                <EuiModal onClose={cancelConflict}>
                    <EuiModalHeader>
                        <EuiModalHeaderTitle>
                            Configuration file changed
                        </EuiModalHeaderTitle>
                    </EuiModalHeader>
                    <EuiModalBody>
                        <EuiText size="s">
                            <p>
                                The configuration file was modified after this
                                page loaded. Review the current file values,
                                refresh the form, or overwrite them.
                            </p>
                            {conflictItems.map((item) => (
                                <p key={item.key}>
                                    <strong>{labelFor(item.key)}:</strong>{' '}
                                    shown value "{String(item.base)}", file
                                    value "{String(item.current)}", requested
                                    value "{String(item.requested)}"
                                </p>
                            ))}
                        </EuiText>
                    </EuiModalBody>
                    <EuiModalFooter>
                        <EuiButton onClick={cancelConflict}>
                            Cancel
                        </EuiButton>
                        <EuiButton onClick={refreshAfterConflict}>
                            Refresh latest
                        </EuiButton>
                        <EuiButton
                            color="warning"
                            fill
                            onClick={overwriteAfterConflict}
                            isLoading={actionInProgress}
                        >
                            {restartAfterConflictWrite
                                ? 'Overwrite & Restart'
                                : 'Overwrite'}
                        </EuiButton>
                    </EuiModalFooter>
                </EuiModal>
            ) : null}
        </div>
    )
}

const mapStateToProps = (state) => ({
    sections: state.node_config.sections,
    nodeState: state.node_config.nodeState,
    requiresRestart: state.node_config.requiresRestart,
    loading: state.node_config.loading,
    writing: state.node_config.writing,
    error: state.node_config.error,
    writeError: state.node_config.writeError,
    writeConflict: state.node_config.writeConflict,
    successMessage: state.node_config.successMessage,
    configModifiedAfterStartup: state.node_config.configModifiedAfterStartup,
    configStartupCheckMessage: state.node_config.configStartupCheckMessage,
    processState: state.node_management.processState,
})

const mapDispatchToProps = (dispatch) => ({
    fetchNodeConfig: () => dispatch(fetchNodeConfig()),
    writeNodeConfigSection: (section, values, baseValues, options) => dispatch(
        writeNodeConfigSection(section, values, baseValues, options)
    ),
    resetNodeConfigMessages: () => dispatch(resetNodeConfigMessages()),
    fetchNodeProcessState: (options) => dispatch(fetchNodeProcessState(options)),
    executeNodeAction: (action, nodeArgs) => dispatch(
        executeNodeAction(action, nodeArgs)
    ),
})

export default connect(mapStateToProps, mapDispatchToProps)(Configuration)
