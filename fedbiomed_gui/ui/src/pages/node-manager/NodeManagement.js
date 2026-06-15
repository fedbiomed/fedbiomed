import React from 'react'
import axios from 'axios'
import {
    EuiBadge,
    EuiBasicTable,
    EuiButton,
    EuiFlexGroup,
    EuiFlexItem,
    EuiIcon,
    EuiPanel,
    EuiSpacer,
    EuiText,
} from '@elastic/eui'

import {EP_NODE_PROCESS_STATE} from '../../constants'
import {ReactComponent as StorageIcon}  from '../../assets/img/disk-storage.svg'
import Header from '../../components/layout/Header'


const emptyValue = '-'

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
    const [processStateError, setProcessStateError] = React.useState(null)
    const [now, setNow] = React.useState(new Date())
    const [lastRefresh, setLastRefresh] = React.useState(null)

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

    React.useEffect(() => {
        loadState()
    }, [])

    const currentState = processState?.state

    React.useEffect(() => {
        if (String(currentState || '').toLowerCase() !== 'running') {
            return undefined
        }

        const intervalId = setInterval(() => {
            setNow(new Date())
        }, 1000)

        return () => clearInterval(intervalId)
    }, [currentState])

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
