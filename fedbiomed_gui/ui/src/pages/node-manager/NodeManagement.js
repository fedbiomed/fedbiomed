import React from 'react'
import {connect} from 'react-redux'
import {
    EuiButton,
    EuiFieldNumber,
    EuiFlexGroup,
    EuiFlexItem,
    EuiFormRow,
    EuiIcon,
    EuiModal,
    EuiModalBody,
    EuiModalFooter,
    EuiModalHeader,
    EuiModalHeaderTitle,
    EuiSwitch,
    EuiToolTip,
} from '@elastic/eui'

import {
    executeNodeAction,
    fetchNodeProcessState,
} from '../../store/actions/nodeManagementActions'

const emptyValue = '-'
const defaultNodeArgs = {
    gpu: false,
    gpu_num: 0,
    gpu_only: false,
    debug: false,
}

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
    fetchNodeProcessState,
    executeNodeAction,
}) => {
    const [now, setNow] = React.useState(new Date())
    const [nodeArgs, setNodeArgs] = React.useState(defaultNodeArgs)
    const [isActorModalVisible, setIsActorModalVisible] = React.useState(false)

    const updateNodeArg = (key, value) => {
        setNodeArgs((currentNodeArgs) => ({
            ...currentNodeArgs,
            [key]: value,
        }))
    }

    React.useEffect(() => {
        fetchNodeProcessState()
    }, [fetchNodeProcessState])

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

        const savedGpuNumber = Number(savedNodeArgs.gpu_num)
        setNodeArgs({
            gpu: Boolean(savedNodeArgs.gpu ?? defaultNodeArgs.gpu),
            gpu_num: Number.isFinite(savedGpuNumber)
                ? Math.max(0, savedGpuNumber)
                : defaultNodeArgs.gpu_num,
            gpu_only: Boolean(
                savedNodeArgs.gpu_only ?? defaultNodeArgs.gpu_only
            ),
            debug: Boolean(savedNodeArgs.debug ?? defaultNodeArgs.debug),
        })
    }, [processState])

    const currentState = processState?.state
    const normalizedState = String(currentState || '').toLowerCase()
    const isStateKnown = Boolean(currentState) && normalizedState !== 'unknown'
    const isRunning = normalizedState === 'running'
    const isStopping = normalizedState === 'stopping'
    const isStopped = normalizedState === 'stopped'

    React.useEffect(() => {
        if (!isRunning) {
            return undefined
        }

        const intervalId = setInterval(() => {
            setNow(new Date())
        }, 1000)

        return () => clearInterval(intervalId)
    }, [isRunning])

    const runningFor = getRunningFor(processState, now)
    const actor = processState?.actor || {}
    const actorName = [actor.name, actor.surname].filter(Boolean).join(' ')
    const statusMessage = isRunning
        ? 'The node is currently running smoothly.'
        : isStopping
            ? 'The node process is stopping.'
            : isStopped
                ? 'The node process is currently stopped.'
                : 'Node process information is not available yet.'

    return (
        <div className="node-management-page">
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

            <section className="node-management-card node-management-process">
                <div className="node-management-section-header">
                    <div className="node-management-section-heading">
                        <span className="node-management-section-icon">
                            <EuiIcon type="stats" size="l" />
                        </span>
                        <div>
                            <h2>Process Details</h2>
                            <p>Current process information and status</p>
                        </div>
                    </div>
                    <div className="node-management-process-header-actions">
                        <EuiToolTip
                            position="bottom"
                            title="Last Action Actor"
                            content={(
                                <div className="node-management-actor-tooltip">
                                    <span>
                                        <strong>Name:</strong>{' '}
                                        {formatValue(
                                            actorName || actor.local_username
                                        )}
                                    </span>
                                    <span>
                                        <strong>Email:</strong>{' '}
                                        {formatValue(actor.email)}
                                    </span>
                                    <span>
                                        <strong>Role:</strong>{' '}
                                        {formatValue(actor.role)}
                                    </span>
                                    <span>
                                        <strong>Source:</strong>{' '}
                                        {formatValue(actor.source)}
                                    </span>
                                    <span>
                                        <strong>User ID:</strong>{' '}
                                        {formatValue(actor.user_id)}
                                    </span>
                                </div>
                            )}
                        >
                            <EuiButton
                                size="s"
                                iconType="user"
                                onClick={() => setIsActorModalVisible(true)}
                            >
                                View actor
                            </EuiButton>
                        </EuiToolTip>
                        <StatusPill state={currentState} />
                    </div>
                </div>

                <div className="node-management-process-content">
                    <aside className="node-management-summary">
                        <SummaryCard
                            className="uptime"
                            label="Uptime"
                            value={runningFor}
                            description="Running for"
                        />
                        <SummaryCard
                            className="pid"
                            label="PID"
                            value={processState?.pid}
                            description="Process ID"
                        />
                    </aside>

                    <div className="node-management-details-grid">
                        <DetailItem
                            icon="tokenKey"
                            label="Node ID"
                            value={processState?.node_id}
                        />
                        <DetailItem
                            icon="calendar"
                            label="Started At"
                            value={formatDateTime(processState?.started_at)}
                        />
                        <DetailItem
                            icon="user"
                            label="Node Name"
                            value={processState?.node_name}
                        />
                        <DetailItem
                            icon="refresh"
                            label="Last Refresh"
                            value={formatDateTime(lastRefresh)}
                        />
                        <DetailItem
                            icon="iInCircle"
                            label="State"
                            valueContent={<StatusPill state={currentState} />}
                        />
                        <DetailItem
                            icon="calendar"
                            label="Updated At"
                            value={formatDateTime(processState?.updated_at)}
                        />
                        <DetailItem
                            icon="clock"
                            label="Running For"
                            value={runningFor}
                        />
                        <DetailItem
                            icon="clock"
                            label="Stopped At"
                            value={formatDateTime(processState?.stopped_at)}
                        />
                        <DetailItem
                            icon="play"
                            label="Action"
                            value={processState?.action}
                        />
                        <DetailItem
                            icon="console"
                            label="Exit Code"
                            value={processState?.exit_code}
                        />
                        <DetailItem
                            icon="flag"
                            label="Reason"
                            value={processState?.reason}
                        />
                        <DetailItem
                            icon="compute"
                            label="GPU"
                            value={processState?.node_args?.gpu}
                        />
                        <DetailItem
                            icon="number"
                            label="GPU Number"
                            value={processState?.node_args?.gpu_num}
                        />
                        <DetailItem
                            icon="check"
                            label="GPU Only"
                            value={processState?.node_args?.gpu_only}
                        />
                        <DetailItem
                            icon="bug"
                            label="Debug"
                            value={processState?.node_args?.debug}
                        />
                        <DetailItem
                            icon="desktop"
                            label="Background"
                            value={processState?.background}
                        />
                    </div>
                </div>

                <div className={`node-management-status-banner ${
                    stateTone(currentState)
                }`}>
                    <span className="node-management-status-banner-icon">
                        <EuiIcon type="iInCircle" size="l" />
                    </span>
                    <div>
                        <strong>Process Status</strong>
                        <p>{statusMessage}</p>
                    </div>
                </div>
            </section>

            {isActorModalVisible ? (
                <EuiModal
                    className="node-management-actor-modal"
                    onClose={() => setIsActorModalVisible(false)}
                >
                    <EuiModalHeader>
                        <EuiModalHeaderTitle>
                            Last Action Actor
                        </EuiModalHeaderTitle>
                    </EuiModalHeader>
                    <EuiModalBody>
                        <p className="node-management-actor-description">
                            User or process responsible for the latest action
                        </p>
                        <div className="node-management-actor-grid">
                            <DetailItem
                                icon="user"
                                label="Name"
                                value={actorName || actor.local_username}
                            />
                            <DetailItem
                                icon="email"
                                label="Email"
                                value={actor.email}
                            />
                            <DetailItem
                                icon="users"
                                label="Role"
                                value={actor.role}
                            />
                            <DetailItem
                                icon="inputOutput"
                                label="Source"
                                value={actor.source}
                            />
                            <DetailItem
                                icon="tokenKey"
                                label="User ID"
                                value={actor.user_id}
                            />
                        </div>
                    </EuiModalBody>
                    <EuiModalFooter>
                        <EuiButton
                            fill
                            onClick={() => setIsActorModalVisible(false)}
                        >
                            Close
                        </EuiButton>
                    </EuiModalFooter>
                </EuiModal>
            ) : null}
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
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        fetchNodeProcessState: (options) => dispatch(
            fetchNodeProcessState(options)
        ),
        executeNodeAction: (action, nodeArgs) => dispatch(
            executeNodeAction(action, nodeArgs)
        ),
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(NodeManagement)
