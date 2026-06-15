import React from 'react'
import {connect} from 'react-redux'
import {
    EuiBadge,
    EuiBasicTable,
    EuiButton,
    EuiFieldNumber,
    EuiFlexGroup,
    EuiFlexItem,
    EuiFormRow,
    EuiIcon,
    EuiPanel,
    EuiSpacer,
    EuiSwitch,
    EuiText,
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

const toRows = (data, keys) => {
    return keys.map((key) => ({
        key,
        value: formatValue(data ? data[key] : null),
    }))
}

const stateBadgeColor = (state) => {
    switch (String(state || '').toLowerCase()) {
        case 'running':
            return 'success'
        case 'stopping':
            return 'warning'
        case 'stopped':
            return 'danger'
        default:
            return 'primary'
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

    const currentState = processState?.state
    const normalizedState = String(currentState || '').toLowerCase()
    const isStateKnown = Boolean(currentState)
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

    const columns = [
        {
            field: 'key',
            name: 'Field',
            render: (value) => (
                <strong className="node-management-table-field">{value}</strong>
            ),
        },
        {
            field: 'value',
            name: 'Value',
            render: (value) => (
                <span className="node-management-table-value">{value}</span>
            ),
        },
    ]

    const processStateWithRuntime = {
        ...processState,
        running_for: getRunningFor(processState, now),
        last_refresh: formatDateTime(lastRefresh),
    }

    const stateRows = toRows(processStateWithRuntime,
        [
            'node_id',
            'node_name',
            'state',
            'running_for',
            'pid',
            'action',
            'reason',
            'updated_at',
            'last_refresh',
            'started_at',
            'stopped_at',
            'exit_code',
        ]
    )

    const actorRows = toRows(processState?.actor, [
        'source',
        'user_id',
        'email',
        'role',
        'name',
        'surname',
        'local_username',
    ])

    return (
        <React.Fragment>
            <EuiFlexGroup
                justifyContent="spaceBetween"
                alignItems="center"
                gutterSize="m"
                wrap
            >
                <EuiFlexItem grow={false}>
                    <EuiIcon type="grid" size="xxl" />
                </EuiFlexItem>
                <EuiFlexItem grow={false}>
                    <EuiText>      
                        <h1>Node Management</h1>
                    </EuiText>
                    <EuiText color="subdued">
                        <p>Monitor and manage node processes</p>
                    </EuiText>
                </EuiFlexItem>
                <EuiFlexItem grow={false}>
                    <EuiBadge color={stateBadgeColor(currentState)}>
                        <span className="node-management-status-badge">
                            {formatValue(currentState).toUpperCase()}
                        </span>
                    </EuiBadge>
                </EuiFlexItem>
                <EuiFlexItem grow={false}>
                    <EuiButton
                        size="m"
                        fill
                        onClick={() => fetchNodeProcessState({
                            markRefresh: true,
                        })}
                        isLoading={loading}
                    >
                        Refresh
                    </EuiButton>
                </EuiFlexItem>
            </EuiFlexGroup>

            {processStateError ? (
                <React.Fragment>
                    <EuiSpacer size="m" />
                    <EuiText color="subdued">
                        <p>{processStateError}</p>
                    </EuiText>
                </React.Fragment>
            ) : null}

            {actionError ? (
                <React.Fragment>
                    <EuiSpacer size="m" />
                    <EuiText color="danger">
                        <p>{actionError}</p>
                    </EuiText>
                </React.Fragment>
            ) : null}

            <EuiSpacer size="l" />
            <EuiPanel paddingSize="m" hasShadow={false} hasBorder>
                <EuiText>
                    <h3>Node Controls</h3>
                </EuiText>
                <EuiSpacer size="m" />
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
                <EuiSpacer size="m" />
                <EuiFlexGroup gutterSize="s" wrap>
                    <EuiFlexItem grow={false}>
                        <EuiButton
                            color="success"
                            fill
                            onClick={() => executeNodeAction('start', nodeArgs)}
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
                            onClick={() => executeNodeAction('stop', nodeArgs)}
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
                            onClick={() => executeNodeAction('restart', nodeArgs)}
                            isLoading={actionLoading === 'restart'}
                            isDisabled={!isStateKnown || isStopping}
                        >
                            Restart
                        </EuiButton>
                    </EuiFlexItem>
                </EuiFlexGroup>
            </EuiPanel>

            <EuiSpacer size="l" />
            <EuiPanel paddingSize="m" hasShadow={false} hasBorder>
                <EuiText>
                    <h3>Process State</h3>
                </EuiText>
                <EuiSpacer size="m" />
                <EuiBasicTable
                    itemId="key"
                    items={stateRows}
                    columns={columns}
                    loading={loading}
                    tableLayout="auto"
                />
            </EuiPanel>

            <EuiSpacer size="l" />
            <EuiPanel paddingSize="m" hasShadow={false} hasBorder>
                <EuiText>
                    <h3>Actor</h3>
                </EuiText>
                <EuiSpacer size="m" />
                <EuiBasicTable
                    itemId="key"
                    items={actorRows}
                    columns={columns}
                    loading={loading}
                    tableLayout="auto"
                />
            </EuiPanel>
            <EuiSpacer size="l" />
        </React.Fragment>
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
