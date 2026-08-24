import React from 'react'
import {connect} from 'react-redux'
import {
    EuiButton,
    EuiButtonEmpty,
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
    writeNodeConfig,
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

const getSavedValues = (sections = {}) => {
    // Values last loaded from config.ini, for every section at once. They are
    // the baseline both for change detection and for backend conflict checks.
    return Object.keys(sections).reduce((values, section) => {
        const fields = sections[section]?.fields || {}

        return {
            ...values,
            [section]: Object.keys(fields).reduce((sectionValues, key) => ({
                ...sectionValues,
                [key]: normalizeFieldValue(fields[key].value, fields[key]),
            }), {}),
        }
    }, {})
}

const collectChanges = (sections = {}, draftValues = {}, savedValues = {}) => {
    // One flat list of every editable field whose form value differs from
    // config.ini. It drives the change count, the inline markers, the review
    // dialog, and the request payload.
    return Object.keys(sections).flatMap((section) => {
        const fields = sections[section]?.fields || {}

        return Object.keys(fields)
            .filter((key) => {
                if (!fields[key].editable) {
                    return false
                }

                const draft = draftValues[section]?.[key]
                return normalizeFieldValue(draft, fields[key])
                    !== savedValues[section]?.[key]
            })
            .map((key) => ({
                section,
                sectionLabel: sections[section]?.label || labelFor(section),
                key,
                label: fields[key].label || labelFor(key),
                current: savedValues[section]?.[key],
                next: normalizeFieldValue(draftValues[section]?.[key], fields[key]),
            }))
    })
}

const buildSectionsPayload = (changes) => {
    // Only changed fields are written. Their loaded values travel along as
    // base values so the backend can reject a write over an external edit.
    return changes.reduce((payload, change) => {
        const section = payload[change.section] || {values: {}, base_values: {}}

        return {
            ...payload,
            [change.section]: {
                values: {...section.values, [change.key]: change.next},
                base_values: {
                    ...section.base_values,
                    [change.key]: change.current,
                },
            },
        }
    }, {})
}

const formatValue = (value) => {
    if (typeof value === 'boolean') {
        return value ? 'True' : 'False'
    }

    return value === '' || value === null || value === undefined
        ? '(empty)'
        : String(value)
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
    writeNodeConfig,
    resetNodeConfigMessages,
    executeNodeAction,
    fetchNodeProcessState,
    fetchProcessStateOnMount = true,
    embedded = false,
}) => {
    // Draft values hold the whole form, section by section. They may differ
    // from config.ini until the user saves or resets.
    const [draftValues, setDraftValues] = React.useState({})
    const [reviewOpen, setReviewOpen] = React.useState(false)
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
    const savedValues = React.useMemo(
        () => getSavedValues(sections),
        [sections]
    )
    const changes = React.useMemo(
        () => collectChanges(sections, draftValues, savedValues),
        [sections, draftValues, savedValues]
    )

    React.useEffect(() => {
        fetchNodeConfig()
        if (fetchProcessStateOnMount) {
            fetchNodeProcessState()
        }
    }, [fetchNodeConfig, fetchNodeProcessState, fetchProcessStateOnMount])

    React.useEffect(() => {
        setDraftValues(savedValues)
    }, [savedValues])

    const updateValue = (section, key, value) => {
        const field = sections?.[section]?.fields?.[key]

        setDraftValues((current) => ({
            ...current,
            [section]: {
                ...current[section],
                [key]: normalizeFieldValue(value, field),
            },
        }))
        resetNodeConfigMessages()
    }

    const saveChanges = async ({restart = false, force = false} = {}) => {
        if (!changes.length) {
            return
        }

        const result = await writeNodeConfig(
            buildSectionsPayload(changes),
            {force}
        )
        if (!result) {
            setReviewOpen(false)
            return
        }

        if (result.conflict) {
            // Stop here. The conflict modal lets the user refresh, cancel, or
            // retry with force before any restart is attempted.
            setReviewOpen(false)
            setRestartAfterConflictWrite(restart)
            return
        }

        setReviewOpen(false)
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
        setDraftValues(savedValues)
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
        await saveChanges({restart: restartAfterConflictWrite, force: true})
    }

    const cancelConflict = () => {
        setRestartAfterConflictWrite(false)
        resetNodeConfigMessages()
    }

    const displayNodeState = String(
        processState?.state || configNodeState || ''
    ).toLowerCase()
    const isRunning = displayNodeState === 'running'
    const dirty = changes.length > 0
    const actionInProgress = writing || restartLoading
    const changedKeys = new Set(
        changes.map((change) => `${change.section}.${change.key}`)
    )
    const conflictItems = Object.keys(writeConflict?.sections || {}).flatMap(
        // Flatten backend conflict details to display in the modal.
        (section) => {
            const conflicts = writeConflict.sections[section]?.conflicts || {}

            return Object.keys(conflicts).map((key) => ({
                section,
                key,
                ...conflicts[key],
            }))
        }
    )

    const renderScalarField = (section, key, field) => {
        const changed = changedKeys.has(`${section}.${key}`)
        const disabled = !field.editable
        const value = draftValues[section]?.[key] ?? ''
        const onChange = (event) => updateValue(section, key, event.target.value)
        let fieldControl = (
            <EuiFieldText
                className="node-config-input"
                disabled={disabled}
                value={value}
                onChange={onChange}
            />
        )

        if (field.type === 'integer') {
            fieldControl = (
                <EuiFieldNumber
                    className="node-config-input"
                    disabled={disabled}
                    min={field.min ?? 0}
                    value={value}
                    onChange={onChange}
                />
            )
        }

        if (field.type === 'enum') {
            fieldControl = (
                <EuiSelect
                    className="node-config-input"
                    disabled={disabled}
                    value={value}
                    options={(field.options || []).map((option) => ({
                        value: option,
                        text: option,
                    }))}
                    onChange={onChange}
                />
            )
        }

        return (
            <EuiFormRow
                className={`node-config-form-row ${changed ? 'changed' : ''}`}
                key={key}
                label={field.label || labelFor(key)}
                labelAppend={changed ? (
                    <span className="node-config-change-hint">
                        was {formatValue(savedValues[section]?.[key])}
                    </span>
                ) : undefined}
            >
                {fieldControl}
            </EuiFormRow>
        )
    }

    const renderBooleanField = (section, key, field) => {
        const changed = changedKeys.has(`${section}.${key}`)
        const disabled = !field.editable
        const checked = Boolean(draftValues[section]?.[key])

        return (
            <article
                className={`node-config-setting ${disabled ? 'disabled' : ''} ${
                    changed ? 'changed' : ''
                }`}
                key={key}
            >
                <div>
                    <div className="node-config-setting-name">
                        {field.label || labelFor(key)}
                    </div>
                    <div className="node-config-setting-help">
                        {getFieldDescription(section, key, field)}
                    </div>
                    {changed ? (
                        <div className="node-config-change-hint">
                            was {formatValue(savedValues[section]?.[key])}
                        </div>
                    ) : null}
                </div>
                <div className="node-config-segmented">
                    <button
                        type="button"
                        className="true"
                        aria-pressed={checked}
                        disabled={disabled}
                        onClick={() => updateValue(section, key, true)}
                    >
                        True
                    </button>
                    <button
                        type="button"
                        className="false"
                        aria-pressed={!checked}
                        disabled={disabled}
                        onClick={() => updateValue(section, key, false)}
                    >
                        False
                    </button>
                </div>
            </article>
        )
    }

    const renderSection = (section) => {
        const sectionInfo = sections[section] || {}
        const fields = sectionInfo.fields || {}
        const fieldKeys = Object.keys(fields)

        if (!fieldKeys.length) {
            return null
        }

        const scalarFieldKeys = fieldKeys.filter(
            (key) => fields[key].type !== 'boolean'
        )
        const booleanFieldKeys = fieldKeys.filter(
            (key) => fields[key].type === 'boolean'
        )
        const editable = fieldKeys.some((key) => fields[key].editable)
        const changedCount = changes.filter(
            (change) => change.section === section
        ).length

        return (
            <section className="node-config-section" key={section}>
                <header className="node-config-section-header">
                    <div className="node-management-section-heading">
                        <span className="node-management-section-icon">
                            <EuiIcon type={sectionIconFor(section)} size="l" />
                        </span>
                        <div>
                            <h2>{sectionInfo.label || labelFor(section)}</h2>
                            <p>
                                {editable
                                    ? 'Changes are written to the node configuration file.'
                                    : 'This section is displayed for reference.'}
                            </p>
                        </div>
                    </div>
                    {changedCount ? (
                        <span className="node-config-section-count">
                            {changedCount} changed
                        </span>
                    ) : null}
                </header>

                <EuiForm className="node-config-form" component="form">
                    {scalarFieldKeys.length ? (
                        <div className="node-config-fields-grid">
                            {scalarFieldKeys.map((key) => (
                                renderScalarField(section, key, fields[key])
                            ))}
                        </div>
                    ) : null}

                    {booleanFieldKeys.length ? (
                        <div className="node-config-settings-grid">
                            {booleanFieldKeys.map((key) => (
                                renderBooleanField(section, key, fields[key])
                            ))}
                        </div>
                    ) : null}
                </EuiForm>
            </section>
        )
    }

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
                        <span className="node-config-status">
                            {String(displayNodeState || 'unknown').toUpperCase()}
                        </span>
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

            <div className="node-management-card node-config-panel">
                {sectionNames.length ? (
                    <div className="node-config-sections">
                        {sectionNames.map(renderSection)}
                    </div>
                ) : (
                    <EuiText>
                        <p>{loading ? 'Loading configuration...' : ''}</p>
                    </EuiText>
                )}
            </div>

            <div className="node-config-action-bar">
                <span className="node-config-action-bar-summary">
                    {dirty
                        ? `${changes.length} unsaved change${
                            changes.length > 1 ? 's' : ''
                        }`
                        : 'No unsaved changes'}
                </span>
                <div className="node-config-action-bar-buttons">
                    <EuiButtonEmpty
                        iconType="inspect"
                        onClick={() => setReviewOpen(true)}
                        isDisabled={!dirty}
                    >
                        Review
                    </EuiButtonEmpty>
                    <EuiButton
                        iconType="cross"
                        onClick={resetChanges}
                        isDisabled={!dirty || actionInProgress}
                    >
                        Reset
                    </EuiButton>
                    {isRunning && !dirty && requiresRestart ? (
                        <EuiButton
                            color="warning"
                            fill
                            iconType="refresh"
                            onClick={restartNode}
                            isLoading={actionInProgress}
                            isDisabled={actionInProgress}
                        >
                            Restart
                        </EuiButton>
                    ) : null}
                    <EuiButton
                        fill
                        iconType="save"
                        onClick={() => setReviewOpen(true)}
                        isLoading={writing}
                        isDisabled={!dirty || actionInProgress}
                    >
                        Save
                    </EuiButton>
                </div>
            </div>

            {reviewOpen ? (
                <EuiModal onClose={() => setReviewOpen(false)}>
                    <EuiModalHeader>
                        <EuiModalHeaderTitle>
                            Review changes ({changes.length})
                        </EuiModalHeaderTitle>
                    </EuiModalHeader>
                    <EuiModalBody>
                        <table className="node-config-review-table">
                            <thead>
                                <tr>
                                    <th>Field</th>
                                    <th>Current</th>
                                    <th>New</th>
                                </tr>
                            </thead>
                            <tbody>
                                {changes.map((change) => (
                                    <tr key={`${change.section}.${change.key}`}>
                                        <td>
                                            <strong>{change.label}</strong>
                                            <small>{change.sectionLabel}</small>
                                        </td>
                                        <td className="node-config-review-current">
                                            {formatValue(change.current)}
                                        </td>
                                        <td className="node-config-review-next">
                                            {formatValue(change.next)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {isRunning ? (
                            <>
                                <EuiSpacer size="m" />
                                <EuiCallOut
                                    color="warning"
                                    iconType="refresh"
                                    size="s"
                                    title={
                                        'The node is running. Changes apply after '
                                        + 'the node restarts.'
                                    }
                                />
                            </>
                        ) : null}
                    </EuiModalBody>
                    <EuiModalFooter>
                        <EuiButton onClick={() => setReviewOpen(false)}>
                            Back
                        </EuiButton>
                        <EuiButton
                            fill
                            iconType="save"
                            onClick={() => saveChanges()}
                            isLoading={writing}
                            isDisabled={actionInProgress}
                        >
                            Save
                        </EuiButton>
                        {isRunning ? (
                            <EuiButton
                                color="warning"
                                fill
                                iconType="refresh"
                                onClick={() => saveChanges({restart: true})}
                                isLoading={actionInProgress}
                                isDisabled={actionInProgress}
                            >
                                Save & Restart
                            </EuiButton>
                        ) : null}
                    </EuiModalFooter>
                </EuiModal>
            ) : null}

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
                                <p key={`${item.section}.${item.key}`}>
                                    <strong>
                                        {labelFor(item.section)} /{' '}
                                        {labelFor(item.key)}:
                                    </strong>{' '}
                                    shown value "{formatValue(item.base)}", file
                                    value "{formatValue(item.current)}",
                                    requested value "{formatValue(item.requested)}"
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
    writeNodeConfig: (sections, options) => dispatch(
        writeNodeConfig(sections, options)
    ),
    resetNodeConfigMessages: () => dispatch(resetNodeConfigMessages()),
    fetchNodeProcessState: (options) => dispatch(fetchNodeProcessState(options)),
    executeNodeAction: (action, nodeArgs) => dispatch(
        executeNodeAction(action, nodeArgs)
    ),
})

export default connect(mapStateToProps, mapDispatchToProps)(Configuration)
