import React from 'react'
import {connect} from 'react-redux'
import {
    EuiButton,
    EuiButtonEmpty,
    EuiButtonIcon,
    EuiIcon,
    EuiNotificationBadge,
    EuiSpacer,
    EuiTab,
    EuiTabs,
    EuiText,
} from '@elastic/eui'

import {
    OwnCertificateModal,
    ResearcherCertificateModal,
} from './Certificates'
import {ConfigGroup, Notices} from './ConfigurationFields'
import {ReviewChangesModal, WriteConflictModal} from './ConfigurationModals'
import {labelFor} from './configFormat'
import {downloadOwnCertificate} from '../../store/actions/certificatesActions'
import {
    fetchNodeConfig,
    resetNodeConfigMessages,
    writeNodeConfig,
} from '../../store/actions/nodeConfigActions'
import {
    executeNodeAction,
    fetchNodeProcessState,
} from '../../store/actions/nodeManagementActions'

// Sections and fields the certificate windows hang off
const CERTIFICATE_SECTION = 'certificate'
const RESEARCHER_SECTION = 'researcher'
const CERTIFICATE_PUBLIC_KEY = 'public_key'
const MTLS_SECTION = 'authentication'
const MTLS_ENABLED = 'mutual_authentication'

/**
 * How the page reads, which is not the order config.ini happens to store its
 * sections in. Settings that belong to one concern are shown together even
 * though the file keeps them apart.
 */
const SECTION_GROUPS = [
    {
        key: 'identity',
        label: 'Node identity',
        icon: 'node',
        sections: ['default'],
    },
    {
        key: 'connection',
        label: 'Connection & certificates',
        icon: 'globe',
        // Them, then us, then the policy binding both: mutual TLS is the only
        // one that depends on the other two already being in place
        sections: ['researcher', 'certificate', 'authentication'],
    },
    {
        key: 'security',
        label: 'Security',
        icon: 'lock',
        sections: ['security'],
    },
    {
        key: 'logging',
        label: 'Logging',
        icon: 'console',
        sections: ['syslog'],
    },
]

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

/**
 * The groups to render, in page order. A section the page does not place yet
 * still gets a group of its own, so a config that grows is never hidden.
 */
const buildGroups = (sections = {}) => {
    const placed = new Set(SECTION_GROUPS.flatMap((group) => group.sections))

    const groups = SECTION_GROUPS
        .map((group) => ({
            ...group,
            sections: group.sections.filter((section) => sections[section]),
        }))
        .filter((group) => group.sections.length)

    const unplaced = Object.keys(sections)
        .filter((section) => !placed.has(section))
        .map((section) => ({
            key: section,
            label: sections[section]?.label || labelFor(section),
            icon: 'controlsHorizontal',
            sections: [section],
        }))

    return [...groups, ...unplaced]
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
    downloadOwnCertificate,
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
    const [activeGroupKey, setActiveGroupKey] = React.useState(null)
    const [ownCertificateOpen, setOwnCertificateOpen] = React.useState(false)
    const [researcherCertificateOpen, setResearcherCertificateOpen] = (
        React.useState(false)
    )

    const groups = React.useMemo(() => buildGroups(sections), [sections])
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

        // Turning mutual TLS on is what makes a researcher certificate
        // necessary, so the window to register one comes with the switch
        if (
            section === MTLS_SECTION
            && key === MTLS_ENABLED
            && value === true
            && !draftValues[section]?.[key]
        ) {
            setResearcherCertificateOpen(true)
        }

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

    // What the file and the running process say about themselves, which is true
    // whether or not the user has done anything yet
    const fileNotices = [
        {
            color: 'warning',
            icon: 'alert',
            title: configModifiedAfterStartup
                ? 'The config file config.ini has been modified after node '
                    + 'startup. The values shown here may not represent the '
                    + 'values effective in the current node process'
                : null,
        },
        {color: 'warning', icon: 'alert', title: configStartupCheckMessage},
    ]

    // The result of what the user just did, kept next to the buttons that did it
    const actionNotices = [
        {color: 'danger', icon: 'alert', title: error || writeError},
        {color: 'success', icon: 'check', title: successMessage},
        {
            color: 'warning',
            icon: 'refresh',
            title: requiresRestart ? 'Restart required' : null,
            body: 'Saved configuration changes will apply after the running '
                + 'node restarts.',
        },
    ]

    // Falls back to the first group, so the panel is never blank while the
    // config loads or if a group disappears from a reloaded config
    const activeGroup = groups.find((group) => group.key === activeGroupKey)
        || groups[0]

    // Unsaved changes live outside the group on screen, so every tab reports
    // its own count and the action bar total stays traceable
    const changedPerGroup = (group) => changes.filter((change) => (
        group.sections.includes(change.section)
    )).length

    // Each certificate window opens from the section it belongs to: the
    // researcher's registry under Researcher, this node's pair under its own
    const sectionActions = {
        [RESEARCHER_SECTION]: (
            <EuiButton
                size="s"
                iconType="users"
                onClick={() => setResearcherCertificateOpen(true)}
            >
                Registered certificate
            </EuiButton>
        ),
        [CERTIFICATE_SECTION]: (
            <EuiButton
                size="s"
                iconType="document"
                onClick={() => setOwnCertificateOpen(true)}
            >
                Manage certificate
            </EuiButton>
        ),
    }

    // The certificate the node presents is the one it hands to a researcher
    const appendFor = (section, key) => (
        section === CERTIFICATE_SECTION && key === CERTIFICATE_PUBLIC_KEY
            ? (
                <EuiButtonIcon
                    iconType="download"
                    aria-label="Download certificate"
                    title="Download this node's certificate"
                    onClick={downloadOwnCertificate}
                />
            )
            : undefined
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

            <Notices notices={fileNotices} />

            <EuiSpacer size="m" />

            <div className="node-management-card node-config-panel">
                {activeGroup ? (
                    <>
                        <EuiTabs className="node-config-group-tabs">
                            {groups.map((group) => {
                                const changedCount = changedPerGroup(group)

                                return (
                                    <EuiTab
                                        key={group.key}
                                        isSelected={group.key === activeGroup.key}
                                        onClick={() => setActiveGroupKey(group.key)}
                                        prepend={<EuiIcon type={group.icon} />}
                                        append={changedCount ? (
                                            <EuiNotificationBadge color="accent">
                                                {changedCount}
                                            </EuiNotificationBadge>
                                        ) : undefined}
                                    >
                                        {group.label}
                                    </EuiTab>
                                )
                            })}
                        </EuiTabs>

                        <ConfigGroup
                            group={activeGroup}
                            sections={sections}
                            sectionActions={sectionActions}
                            draftValues={draftValues}
                            savedValues={savedValues}
                            changedKeys={changedKeys}
                            onChange={updateValue}
                            appendFor={appendFor}
                        />
                    </>
                ) : (
                    <EuiText>
                        <p>{loading ? 'Loading configuration...' : ''}</p>
                    </EuiText>
                )}
            </div>

            <Notices notices={actionNotices} />

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

            {ownCertificateOpen ? (
                <OwnCertificateModal
                    onClose={() => setOwnCertificateOpen(false)}
                />
            ) : null}

            {researcherCertificateOpen ? (
                <ResearcherCertificateModal
                    onClose={() => setResearcherCertificateOpen(false)}
                />
            ) : null}

            {reviewOpen ? (
                <ReviewChangesModal
                    changes={changes}
                    isRunning={isRunning}
                    writing={writing}
                    actionInProgress={actionInProgress}
                    onCancel={() => setReviewOpen(false)}
                    onSave={() => saveChanges()}
                    onSaveAndRestart={() => saveChanges({restart: true})}
                />
            ) : null}

            {writeConflict ? (
                <WriteConflictModal
                    conflictItems={conflictItems}
                    restartAfterWrite={restartAfterConflictWrite}
                    actionInProgress={actionInProgress}
                    onCancel={cancelConflict}
                    onRefresh={refreshAfterConflict}
                    onOverwrite={overwriteAfterConflict}
                />
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
    downloadOwnCertificate: () => dispatch(downloadOwnCertificate()),
})

export default connect(mapStateToProps, mapDispatchToProps)(Configuration)
