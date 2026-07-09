import React from 'react'
import {connect} from 'react-redux'
import {
    EuiButton,
    EuiFieldNumber,
    EuiFlexGroup,
    EuiFlexItem,
    EuiFormRow,
    EuiIcon,
    EuiSwitch,
    EuiTab,
    EuiTabs,
} from '@elastic/eui'

import ApplicationLogs from './ApplicationLogs'
import ProcessDetails from './ProcessDetails'
import Configuration from './Configuration'
import {
    downloadNodeLogFile,
    executeNodeAction,
    fetchNodeLogFiles,
    fetchNodeLogs,
    fetchNodeProcessState,
} from '../../store/actions/nodeManagementActions'

const emptyValue = '-'
const defaultNodeArgs = {
    gpu: false,
    gpu_num: 0,
    gpu_only: false,
    debug: false,
}

const normalizeNodeArgs = (nodeArgs = {}) => {
    const gpuNumber = Number(nodeArgs.gpu_num)

    return {
        gpu: Boolean(nodeArgs.gpu ?? defaultNodeArgs.gpu),
        gpu_num: Number.isFinite(gpuNumber)
            ? Math.max(0, gpuNumber)
            : defaultNodeArgs.gpu_num,
        gpu_only: Boolean(nodeArgs.gpu_only ?? defaultNodeArgs.gpu_only),
        debug: Boolean(nodeArgs.debug ?? defaultNodeArgs.debug),
    }
}

const areNodeArgsEqual = (firstNodeArgs, secondNodeArgs) => {
    const first = normalizeNodeArgs(firstNodeArgs)
    const second = normalizeNodeArgs(secondNodeArgs)

    return Object.keys(defaultNodeArgs).every((key) => (
        first[key] === second[key]
    ))
}

const nodeManagementTabs = {
    process: 'process',
    logs: 'logs',
}

const mainTabs = {
    management: 'management',
    configuration: 'configuration',
}

const applicationLogBasename = 'application.log'

const formatValue = (value) => {
    if (value === null || value === undefined || value === '') {
        return emptyValue
    }

    if (typeof value === 'object') {
        return JSON.stringify(value)
    }

    return String(value)
}

const formatDateTime = (value) => {
    if (!value) {
        return emptyValue
    }

    const date = value instanceof Date ? value : new Date(value)
    if (!(date instanceof Date) || Number.isNaN(date.getTime())) {
        return emptyValue
    }

    return date.toLocaleString()
}

const stateTone = (state) => {
    switch (String(state || '').toLowerCase()) {
        case 'running':
            return 'success'
        case 'stopping':
            return 'warning'
        case 'stopped':
            return 'danger'
        default:
            return 'neutral'
    }
}

const parseTimestamp = (timestamp) => {
    if (!timestamp) {
        return null
    }

    const date = new Date(timestamp)
    return Number.isNaN(date.getTime()) ? null : date
}

const formatDuration = (durationMs) => {
    if (!Number.isFinite(durationMs) || durationMs < 0) {
        return emptyValue
    }

    const totalSeconds = Math.floor(durationMs / 1000)
    const days = Math.floor(totalSeconds / 86400)
    const hours = Math.floor((totalSeconds % 86400) / 3600)
    const minutes = Math.floor((totalSeconds % 3600) / 60)
    const seconds = totalSeconds % 60

    const parts = []

    if (days) {
        parts.push(`${days}d`)
    }
    if (hours || parts.length) {
        parts.push(`${hours}h`)
    }
    if (minutes || parts.length) {
        parts.push(`${minutes}m`)
    }
    parts.push(`${seconds}s`)

    return parts.join(' ')
}

const getRunningFor = (processState, now) => {
    if (String(processState?.state || '').toLowerCase() !== 'running') {
        return emptyValue
    }

    const startedAt = parseTimestamp(processState?.started_at)
    if (!startedAt) {
        return emptyValue
    }

    return formatDuration(now.getTime() - startedAt.getTime())
}

const formatLogFileLabel = (file) => {
    const name = file?.name || applicationLogBasename
    if (name === applicationLogBasename) {
        return `${name} (current)`
    }

    return name
}

const StatusPill = ({state, uppercase = false}) => {
    const normalizedState = String(state || '').toLowerCase()
    const displayState = !normalizedState || normalizedState === 'unknown'
        ? 'stopped'
        : state
    const label = formatValue(displayState)

    return (
        <span className={`node-management-status-pill ${
            stateTone(displayState)
        }`}>
            <span className="node-management-status-dot" />
            {uppercase ? label.toUpperCase() : label}
        </span>
    )
}

const SummaryCard = ({className, label, value, description}) => (
    <div className={`node-management-summary-card ${className}`}>
        <span className="node-management-summary-label">{label}</span>
        <strong>{formatValue(value)}</strong>
        <span className="node-management-summary-description">
            {description}
        </span>
    </div>
)

const DetailItem = ({icon, label, value, valueContent}) => (
    <div className="node-management-detail-item">
        <span className="node-management-detail-icon">
            <EuiIcon type={icon} size="m" />
        </span>
        <span className="node-management-detail-content">
            <span className="node-management-detail-label">{label}</span>
            <span className="node-management-detail-value">
                {valueContent || formatValue(value)}
            </span>
        </span>
    </div>
)

const NodeManagement = ({
    processState,
    loading,
    actionLoading,
    actionError,
    processStateError,
    lastRefresh,
    logItems,
    logLoading,
    logError,
    logLastRefresh,
    logCursor,
    logHasMore,
    logFileSize,
    logFiles,
    logFilesLoading,
    logFilesError,
    fetchNodeProcessState,
    fetchNodeLogs,
    fetchNodeLogFiles,
    downloadNodeLogFile,
    executeNodeAction,
}) => {
    const [now, setNow] = React.useState(new Date())
    const [nodeArgs, setNodeArgs] = React.useState(defaultNodeArgs)
    const [isActorModalVisible, setIsActorModalVisible] = React.useState(false)
    const [activeMainTab, setActiveMainTab] = React.useState(mainTabs.management)
    const [activeTab, setActiveTab] = React.useState(nodeManagementTabs.process)
    const [logFileName, setLogFileName] = React.useState(applicationLogBasename)
    const [logPageSize, setLogPageSize] = React.useState(100)
    const logScrollRef = React.useRef(null)
    const preserveLogScrollRef = React.useRef(null)
    const scrollToLatestLogRef = React.useRef(false)
    const pendingOlderLogLoadRef = React.useRef(false)

    const updateNodeArg = (key, value) => {
        setNodeArgs((currentNodeArgs) => ({
            ...currentNodeArgs,
            [key]: value,
        }))
    }

    React.useEffect(() => {
        fetchNodeProcessState()
    }, [fetchNodeProcessState])

    const loadLatestNodeLogs = React.useCallback(() => {
        scrollToLatestLogRef.current = true
        fetchNodeLogs({
            fileName: logFileName,
            pageSize: logPageSize,
            mode: 'reset',
        })
    }, [fetchNodeLogs, logFileName, logPageSize])

    const loadOlderNodeLogs = React.useCallback(() => {
        if (!logHasMore || logLoading || pendingOlderLogLoadRef.current) {
            return
        }

        const scrollElement = logScrollRef.current
        if (scrollElement) {
            preserveLogScrollRef.current = {
                scrollHeight: scrollElement.scrollHeight,
                scrollTop: scrollElement.scrollTop,
            }
        }

        pendingOlderLogLoadRef.current = true
        Promise.resolve(fetchNodeLogs({
            cursor: logCursor,
            fileName: logFileName,
            pageSize: logPageSize,
            mode: 'prepend',
        })).finally(() => {
            pendingOlderLogLoadRef.current = false
        })
    }, [
        fetchNodeLogs,
        logCursor,
        logFileName,
        logHasMore,
        logLoading,
        logPageSize,
    ])

    const handleLogScroll = React.useCallback((event) => {
        if (event.currentTarget.scrollTop <= 80) {
            loadOlderNodeLogs()
        }
    }, [loadOlderNodeLogs])

    const handleLogFileChange = (fileName) => {
        preserveLogScrollRef.current = null
        setLogFileName(fileName)
    }

    const handleLogPageSizeChange = (pageSize) => {
        setLogPageSize(pageSize)
    }

    React.useEffect(() => {
        if (
            activeMainTab !== mainTabs.management
            || activeTab !== nodeManagementTabs.logs
        ) {
            return
        }

        fetchNodeLogFiles()
    }, [activeMainTab, activeTab, fetchNodeLogFiles])

    React.useEffect(() => {
        if (
            activeMainTab !== mainTabs.management
            || activeTab !== nodeManagementTabs.logs
        ) {
            return
        }

        loadLatestNodeLogs()
    }, [activeMainTab, activeTab, loadLatestNodeLogs])

    React.useEffect(() => {
        if (!logFiles.length) {
            return
        }

        if (!logFiles.some((file) => file.name === logFileName)) {
            setLogFileName(logFiles[0].name)
        }
    }, [logFileName, logFiles])

    React.useEffect(() => {
        if (lastRefresh) {
            setNow(new Date())
        }
    }, [lastRefresh])

    React.useEffect(() => {
        const savedNodeArgs = processState?.node_args
        if (!savedNodeArgs || typeof savedNodeArgs !== 'object') {
            return
        }

        setNodeArgs(normalizeNodeArgs(savedNodeArgs))
    }, [processState])

    const currentState = processState?.state
    const normalizedState = String(currentState || '').toLowerCase()
    const isStateKnown = Boolean(currentState) && normalizedState !== 'unknown'
    const isRunning = normalizedState === 'running'
    const isStopping = normalizedState === 'stopping'
    const isStopped = normalizedState === 'stopped'
    const hasSavedNodeArgs = processState?.node_args
        && typeof processState.node_args === 'object'
    const hasPendingNodeArgChanges = isRunning
        && hasSavedNodeArgs
        && !areNodeArgsEqual(nodeArgs, processState.node_args)
    const configModifiedAfterStartup = Boolean(
        processState?.config_modified_after_startup
    )
    const configStartupCheckMessage = (
        processState?.config_startup_check_message
    )

    React.useEffect(() => {
        if (!isRunning) {
            return undefined
        }

        const intervalId = setInterval(() => {
            setNow(new Date())
        }, 1000)

        return () => clearInterval(intervalId)
    }, [isRunning])

    React.useEffect(() => {
        const scrollElement = logScrollRef.current
        if (!scrollElement) {
            return
        }

        const preservedScroll = preserveLogScrollRef.current
        if (preservedScroll) {
            scrollElement.scrollTop = (
                scrollElement.scrollHeight
                - preservedScroll.scrollHeight
                + preservedScroll.scrollTop
            )
            preserveLogScrollRef.current = null
            return
        }

        if (scrollToLatestLogRef.current) {
            scrollElement.scrollTop = scrollElement.scrollHeight
            scrollToLatestLogRef.current = false
        }
    }, [logItems.length, logLastRefresh])

    const runningFor = getRunningFor(processState, now)
    const actor = processState?.actor || {}
    const actorName = [actor.name, actor.surname].filter(Boolean).join(' ')
    const logFileOptions = (logFiles.length ? logFiles : [
        {name: applicationLogBasename},
    ]).map((file) => ({
        value: file.name,
        text: formatLogFileLabel(file),
    }))
    const statusMessage = isRunning
        ? 'The node is currently running smoothly.'
        : isStopping
            ? 'The node process is stopping.'
            : isStopped
                ? 'The node process is currently stopped.'
                : 'Node process information is not available yet.'

    return (
        <div className="node-management-page">
            <section className="node-management-card node-management-tabs-card">
                <EuiTabs>
                    <EuiTab
                        isSelected={activeMainTab === mainTabs.management}
                        onClick={() => setActiveMainTab(mainTabs.management)}
                    >
                        Node Management
                    </EuiTab>
                    <EuiTab
                        isSelected={activeMainTab === mainTabs.configuration}
                        onClick={() => setActiveMainTab(mainTabs.configuration)}
                    >
                        Configuration
                    </EuiTab>
                </EuiTabs>
            </section>

            {activeMainTab === mainTabs.configuration ? (
                <Configuration
                    fetchProcessStateOnMount={false}
                    embedded
                />
            ) : (
                <>
            <section className="node-management-card node-management-header">
                <div className="node-management-header-top">
                    <div className="node-management-heading">
                        <span className="node-management-heading-icon">
                            <EuiIcon type="node" size="xl" />
                        </span>
                        <div>
                            <h1>Node Management</h1>
                            <p>Monitor and manage node processes</p>
                        </div>
                    </div>
                    <div className="node-management-header-actions">
                        <StatusPill state={currentState} uppercase />
                        <EuiButton
                            iconType="refresh"
                            fill
                            onClick={() => fetchNodeProcessState({
                                markRefresh: true,
                            })}
                            isLoading={loading}
                        >
                            Refresh
                        </EuiButton>
                    </div>
                </div>

                <div className="node-management-header-controls">
                    <EuiFlexGroup gutterSize="m" alignItems="flexEnd" wrap>
                        <EuiFlexItem grow={false}>
                            <EuiFormRow label="GPU">
                                <EuiSwitch
                                    label="Enable GPU"
                                    checked={nodeArgs.gpu}
                                    onChange={(event) => updateNodeArg(
                                        'gpu',
                                        event.target.checked
                                    )}
                                />
                            </EuiFormRow>
                        </EuiFlexItem>
                        <EuiFlexItem grow={false}>
                            <EuiFormRow label="GPU only">
                                <EuiSwitch
                                    label="GPU only"
                                    checked={nodeArgs.gpu_only}
                                    onChange={(event) => updateNodeArg(
                                        'gpu_only',
                                        event.target.checked
                                    )}
                                />
                            </EuiFormRow>
                        </EuiFlexItem>
                        <EuiFlexItem grow={false}>
                            <EuiFormRow label="Debug">
                                <EuiSwitch
                                    label="Debug"
                                    checked={nodeArgs.debug}
                                    onChange={(event) => updateNodeArg(
                                        'debug',
                                        event.target.checked
                                    )}
                                />
                            </EuiFormRow>
                        </EuiFlexItem>
                        <EuiFlexItem grow={false}>
                            <EuiFormRow label="GPU number">
                                <EuiFieldNumber
                                    min={0}
                                    value={nodeArgs.gpu_num}
                                    onChange={(event) => updateNodeArg(
                                        'gpu_num',
                                        Math.max(
                                            0,
                                            Number.parseInt(
                                                event.target.value || '0',
                                                10
                                            ) || 0
                                        )
                                    )}
                                />
                            </EuiFormRow>
                        </EuiFlexItem>
                    </EuiFlexGroup>

                    {hasPendingNodeArgChanges ? (
                        <div className="node-management-alert info">
                            <EuiIcon type="iInCircle" />
                            <span>
                                Node has to be restarted for the changes to
                                take effect.
                            </span>
                        </div>
                    ) : null}

                    <EuiFlexGroup
                        className="node-management-header-control-actions"
                        gutterSize="s"
                        wrap
                    >
                        <EuiFlexItem grow={false}>
                            <EuiButton
                                color="success"
                                fill
                                onClick={() => executeNodeAction(
                                    'start',
                                    nodeArgs
                                )}
                                isLoading={actionLoading === 'start'}
                                isDisabled={isRunning || isStopping}
                            >
                                Start
                            </EuiButton>
                        </EuiFlexItem>
                        <EuiFlexItem grow={false}>
                            <EuiButton
                                color="danger"
                                fill
                                onClick={() => executeNodeAction(
                                    'stop',
                                    nodeArgs
                                )}
                                isLoading={actionLoading === 'stop'}
                                isDisabled={!isStateKnown || isStopped}
                            >
                                Stop
                            </EuiButton>
                        </EuiFlexItem>
                        <EuiFlexItem grow={false}>
                            <EuiButton
                                color="warning"
                                fill
                                onClick={() => executeNodeAction(
                                    'restart',
                                    nodeArgs
                                )}
                                isLoading={actionLoading === 'restart'}
                                isDisabled={!isStateKnown || isStopping}
                            >
                                Restart
                            </EuiButton>
                        </EuiFlexItem>
                    </EuiFlexGroup>
                </div>
            </section>

            {processStateError ? (
                <div className="node-management-alert error">
                    <EuiIcon type="alert" />
                    <span>{processStateError}</span>
                </div>
            ) : null}

            {actionError ? (
                <div className="node-management-alert error">
                    <EuiIcon type="alert" />
                    <span>{actionError}</span>
                </div>
            ) : null}

            {configModifiedAfterStartup ? (
                <div className="node-management-alert warning">
                    <EuiIcon type="alert" />
                    <span>
                        The config file config.ini has been modified after node
                        startup
                    </span>
                </div>
            ) : null}

            {configStartupCheckMessage ? (
                <div className="node-management-alert warning">
                    <EuiIcon type="alert" />
                    <span>{configStartupCheckMessage}</span>
                </div>
            ) : null}

            <section className="node-management-card node-management-tabs-card">
                <EuiTabs>
                    <EuiTab
                        isSelected={activeTab === nodeManagementTabs.process}
                        onClick={() => setActiveTab(nodeManagementTabs.process)}
                    >
                        Process Details
                    </EuiTab>
                    <EuiTab
                        isSelected={activeTab === nodeManagementTabs.logs}
                        onClick={() => setActiveTab(nodeManagementTabs.logs)}
                    >
                        Application Logs
                    </EuiTab>
                </EuiTabs>
            </section>

            {activeTab === nodeManagementTabs.process ? (
                <ProcessDetails
                    actor={actor}
                    actorName={actorName}
                    currentState={currentState}
                    DetailItem={DetailItem}
                    formatDateTime={formatDateTime}
                    formatValue={formatValue}
                    isActorModalVisible={isActorModalVisible}
                    lastRefresh={lastRefresh}
                    onCloseActorModal={() => setIsActorModalVisible(false)}
                    onShowActorModal={() => setIsActorModalVisible(true)}
                    processState={processState}
                    runningFor={runningFor}
                    stateTone={stateTone}
                    statusMessage={statusMessage}
                    StatusPill={StatusPill}
                    SummaryCard={SummaryCard}
                />
            ) : (
                <ApplicationLogs
                    downloadNodeLogFile={downloadNodeLogFile}
                    formatDateTime={formatDateTime}
                    formatValue={formatValue}
                    handleLogScroll={handleLogScroll}
                    loadLatestNodeLogs={loadLatestNodeLogs}
                    loadOlderNodeLogs={loadOlderNodeLogs}
                    logError={logError}
                    logFileName={logFileName}
                    logFileOptions={logFileOptions}
                    logFileSize={logFileSize}
                    logFilesError={logFilesError}
                    logFilesLoading={logFilesLoading}
                    logHasMore={logHasMore}
                    logItems={logItems}
                    logLastRefresh={logLastRefresh}
                    logLoading={logLoading}
                    logPageSize={logPageSize}
                    logScrollRef={logScrollRef}
                    onLogFileChange={handleLogFileChange}
                    onLogPageSizeChange={handleLogPageSizeChange}
                />
            )}
                </>
            )}
        </div>
    )
}

const mapStateToProps = (state) => {
    return {
        processState: state.node_management.processState,
        loading: state.node_management.loading,
        actionLoading: state.node_management.actionLoading,
        actionError: state.node_management.actionError,
        processStateError: state.node_management.processStateError,
        lastRefresh: state.node_management.lastRefresh,
        logItems: state.node_management.logItems,
        logLoading: state.node_management.logLoading,
        logError: state.node_management.logError,
        logLastRefresh: state.node_management.logLastRefresh,
        logCursor: state.node_management.logCursor,
        logHasMore: state.node_management.logHasMore,
        logFileSize: state.node_management.logFileSize,
        logFiles: state.node_management.logFiles,
        logFilesLoading: state.node_management.logFilesLoading,
        logFilesError: state.node_management.logFilesError,
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        fetchNodeProcessState: (options) => dispatch(
            fetchNodeProcessState(options)
        ),
        fetchNodeLogs: (args) => dispatch(fetchNodeLogs(args)),
        fetchNodeLogFiles: () => dispatch(fetchNodeLogFiles()),
        downloadNodeLogFile: (fileName) => dispatch(
            downloadNodeLogFile(fileName)
        ),
        executeNodeAction: (action, nodeArgs) => dispatch(
            executeNodeAction(action, nodeArgs)
        ),
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(NodeManagement)
