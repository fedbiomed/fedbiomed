import React from 'react'
import axios from 'axios'
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
    EP_NODE_PROCESS_STATE,
    EP_NODE_RESTART,
    EP_NODE_START,
    EP_NODE_STOP,
} from '../../constants'
import {ReactComponent as StorageIcon}  from '../../assets/img/disk-storage.svg'
import Header from '../../components/layout/Header'


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

const formatDateTime = (date) => {
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

const NodeManagement = () => {
    const [processState, setProcessState] = React.useState(null)
    const [loading, setLoading] = React.useState(false)
    const [actionLoading, setActionLoading] = React.useState(null)
    const [actionError, setActionError] = React.useState(null)
    const [processStateError, setProcessStateError] = React.useState(null)
    const [now, setNow] = React.useState(new Date())
    const [lastRefresh, setLastRefresh] = React.useState(null)
    const [nodeArgs, setNodeArgs] = React.useState(defaultNodeArgs)

    const loadState = React.useCallback(async ({markRefresh = false} = {}) => {
        setLoading(true)
        setProcessStateError(null)

        try {
            const processStateResponse = await axios.get(EP_NODE_PROCESS_STATE)
            setProcessState(processStateResponse.data.result)
            const currentDate = new Date()
            setNow(currentDate)
            if (markRefresh) {
                setLastRefresh(currentDate)
            }
        } catch (error) {
            setProcessState(null)
            setProcessStateError(
                error?.response?.data?.message || 'Could not get node process state'
            )
        } finally {
            setLoading(false)
        }
    }, [])

    const updateNodeArg = (key, value) => {
        setNodeArgs((currentNodeArgs) => ({
            ...currentNodeArgs,
            [key]: value,
        }))
    }

    const nodeActionBody = () => ({
        ...nodeArgs,
        gpu_num: Math.max(0, Number(nodeArgs.gpu_num) || 0),
    })

    const runNodeAction = async (action, endpoint) => {
        setActionLoading(action)
        setActionError(null)

        try {
            if (action === 'stop') {
                await axios.post(endpoint)
            } else {
                await axios.post(endpoint, nodeActionBody())
            }
            await loadState({markRefresh: true})
        } catch (error) {
            setActionError(
                error?.response?.data?.message || `Could not ${action} node process`
            )
        } finally {
            setActionLoading(null)
        }
    }

    React.useEffect(() => {
        loadState()
    }, [])

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
                        onClick={() => loadState({markRefresh: true})}
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
                            onClick={() => runNodeAction('start', EP_NODE_START)}
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
                            onClick={() => runNodeAction('stop', EP_NODE_STOP)}
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
                            onClick={() => runNodeAction('restart', EP_NODE_RESTART)}
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

export default NodeManagement
