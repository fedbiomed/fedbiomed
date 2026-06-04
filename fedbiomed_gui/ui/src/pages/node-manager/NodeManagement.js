import React from 'react'
import axios from 'axios'
import {
    EuiBadge,
    EuiBasicTable,
    EuiButton,
    EuiFlexGroup,
    EuiFlexItem,
    EuiSpacer,
    EuiText,
    EuiTitle,
} from '@elastic/eui'

import {EP_NODE_PROCESS_STATE} from '../../constants'


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
            return 'hollow'
        default:
            return 'default'
    }
}

const NodeManagement = () => {
    const [processState, setProcessState] = React.useState(null)
    const [loading, setLoading] = React.useState(false)
    const [processStateError, setProcessStateError] = React.useState(null)

    const loadState = React.useCallback(async () => {
        setLoading(true)
        setProcessStateError(null)

        try {
            const processStateResponse = await axios.get(EP_NODE_PROCESS_STATE)
            setProcessState(processStateResponse.data.result)
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

    const columns = [
        {
            field: 'key',
            name: 'Field',
            render: (value) => <span style={{wordBreak: 'break-word'}}>{value}</span>,
        },
        {
            field: 'value',
            name: 'Value',
            render: (value) => <span style={{wordBreak: 'break-word'}}>{value}</span>,
        },
    ]

    const stateRows = toRows(processState,
        [
            'node_id',
            'node_name',
            'state',
            'pid',
            'action',
            'reason',
            'updated_at',
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

    const currentState = processState?.state

    return (
        <React.Fragment>
            <EuiFlexGroup justifyContent="spaceBetween" alignItems="center" gutterSize="m" wrap>
                <EuiFlexItem grow={false}>
                    <EuiTitle size="m">
                        <h2>Node Management</h2>
                    </EuiTitle>
                </EuiFlexItem>
                <EuiFlexItem grow={false}>
                    <EuiFlexGroup gutterSize="s" alignItems="center">
                        <EuiFlexItem grow={false}>
                            <EuiBadge color={stateBadgeColor(currentState)}>
                                <EuiText size="m">
                                    <strong>{formatValue(currentState).toUpperCase()}</strong>
                                </EuiText>
                            </EuiBadge>
                        </EuiFlexItem>
                        <EuiFlexItem grow={false}>
                            <EuiButton size="s" onClick={loadState} isLoading={loading}>
                                Refresh
                            </EuiButton>
                        </EuiFlexItem>
                    </EuiFlexGroup>
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

            <EuiSpacer size="m" />
            <EuiText>
                <h3>Process State</h3>
            </EuiText>
            <EuiBasicTable
                itemId="key"
                items={stateRows}
                columns={columns}
                loading={loading}
                tableLayout="auto"
            />

            <EuiSpacer size="l" />
            <EuiText>
                <h3>Actor</h3>
            </EuiText>
            <EuiBasicTable
                itemId="key"
                items={actorRows}
                columns={columns}
                loading={loading}
                tableLayout="auto"
            />
        </React.Fragment>
    )
}

export default NodeManagement
