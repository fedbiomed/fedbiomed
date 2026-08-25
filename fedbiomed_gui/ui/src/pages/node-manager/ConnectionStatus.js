import React from 'react'
import {connect} from 'react-redux'
import {
    EuiBadge,
    EuiButton,
    EuiCallOut,
    EuiFlexGroup,
    EuiFlexItem,
    EuiIcon,
    EuiSpacer,
    EuiText,
} from '@elastic/eui'

import {
    fetchCertificateStatus,
    fetchConnectionState,
} from '../../store/actions/certificatesActions'

const emptyValue = '-'

// Enough to see a connection settle or flap, without the whole retention window
const historyShown = 5

// What to do about the state the node last recorded, from the troubleshooting
// table of the mutual-TLS guide. Keyed by the event the node recorded.
const fixHints = {
    mtls_handshake_failure:
        'The pinned researcher certificate does not match the one served. '
        + 'Register the researcher\'s current certificate; if it is already '
        + 'current, treat this as a possible man-in-the-middle.',
    mtls_identity_rejected:
        'The researcher rejected this node\'s identity. Its certificate has to '
        + 'be registered there under the id this node declares.',
    mtls_not_enforced_by_researcher:
        'This node requires mutual TLS but the researcher verifies no node '
        + 'identity. Enable it on the researcher and register this node\'s '
        + 'certificate there, or disable it here.',
    mtls_required_by_researcher:
        'The researcher requires mutual TLS. Enable it here, register the '
        + 'researcher certificate, and have this node\'s certificate '
        + 'registered there.',
    mtls_startup_refused:
        'The node refused to start over its certificates. Resolve the reason '
        + 'above, then start it again.',
    researcher_unavailable:
        'The researcher endpoint did not answer. Check that it runs and that '
        + 'the host and port configured here are the ones it serves.',
}

const formatValue = (value) => {
    if (value === null || value === undefined || value === '') {
        return emptyValue
    }

    return String(value)
}

const formatDateTime = (value) => {
    if (!value) {
        return emptyValue
    }

    const date = new Date(value)

    return Number.isNaN(date.getTime()) ? emptyValue : date.toLocaleString()
}

/** How the recorded connection reads: its wording and its tone. */
const connectionSummary = (connection) => {
    const state = connection?.state

    if (!state) {
        return {
            color: 'default',
            label: 'No connection recorded',
            detail: 'The node has not reported a connection yet.',
        }
    }

    if (state.state !== 'connected') {
        return {
            color: state.state === 'failed' ? 'danger' : 'warning',
            label: state.state === 'failed' ? 'Failed' : 'Disconnected',
            detail: state.reason,
        }
    }

    if (!state.mtls) {
        return {
            color: 'primary',
            label: 'Server-authenticated TLS',
            detail: 'Connected without mutual TLS: the researcher does not '
                + 'verify this node\'s identity.',
        }
    }

    // The node only reaches a connected state under mutual TLS once the researcher
    // has named it from the certificate it presented, so this is not in doubt.
    return {
        color: 'success',
        label: 'Mutual TLS, identity verified',
        detail: 'The researcher verified this node\'s identity.',
    }
}

const DetailItem = ({label, value}) => (
    <div className="node-management-detail-item">
        <span className="node-management-detail-label">{label}</span>
        <span className="node-management-detail-value">{value}</span>
    </div>
)

const ConnectionStatus = ({
    connection,
    connectionError,
    certificateStatus,
    certificateError,
    fetchConnectionState,
    fetchCertificateStatus,
}) => {
    const [refreshing, setRefreshing] = React.useState(false)

    const read = React.useCallback(() => Promise.all([
        fetchConnectionState(),
        fetchCertificateStatus(),
    ]), [fetchCertificateStatus, fetchConnectionState])

    React.useEffect(() => {
        read()
    }, [read])

    const refresh = async () => {
        setRefreshing(true)
        try {
            await read()
        } finally {
            setRefreshing(false)
        }
    }

    const summary = connectionSummary(connection)
    const recorded = connection?.state
    const startupProblems = certificateStatus?.startup_problems || []
    const warnings = certificateStatus?.warnings || []

    return (
        <section className="node-management-card">
            <div className="node-management-section-header">
                <div className="node-management-section-heading">
                    <span className="node-management-section-icon">
                        <EuiIcon type="globe" size="l" />
                    </span>
                    <div>
                        <h2>Connection &amp; Diagnostics</h2>
                        <p>
                            The connection to the researcher as the node
                            recorded it, what it did before, and what would
                            stop it from connecting at all
                        </p>
                    </div>
                </div>
                <div className="node-management-process-header-actions">
                    <EuiButton
                        size="s"
                        iconType="refresh"
                        onClick={refresh}
                        isLoading={refreshing}
                    >
                        Refresh
                    </EuiButton>
                </div>
            </div>

            {[connectionError, certificateError].filter(Boolean).map((message) => (
                <div className="node-management-alert error" key={message}>
                    <EuiIcon type="alert" />
                    <span>{message}</span>
                </div>
            ))}

            <EuiSpacer size="m" />

            <EuiFlexGroup gutterSize="m" alignItems="center" wrap>
                <EuiFlexItem grow={false}>
                    <EuiBadge color={summary.color}>{summary.label}</EuiBadge>
                </EuiFlexItem>
                <EuiFlexItem>
                    <EuiText size="s">{summary.detail}</EuiText>
                </EuiFlexItem>
            </EuiFlexGroup>

            {connection?.stale ? (
                <EuiText size="xs" color="subdued">
                    <p>
                        The node is not running, so this is what it last
                        observed, not what is true now.
                    </p>
                </EuiText>
            ) : null}

            {recorded ? (
                <div className="node-management-details-grid">
                    <DetailItem
                        label="Researcher"
                        value={`${formatValue(recorded.host)}:`
                            + `${formatValue(recorded.port)}`}
                    />
                    <DetailItem
                        label="Researcher id"
                        value={formatValue(recorded.researcher_id)}
                    />
                    <DetailItem
                        label="Since"
                        value={formatDateTime(recorded.started_at)}
                    />
                    <DetailItem
                        label="Last observed"
                        value={formatDateTime(recorded.updated_at)}
                    />
                    <DetailItem
                        label="Last error"
                        value={formatValue(recorded.last_error)}
                    />
                </div>
            ) : null}

            <EuiSpacer size="m" />

            <EuiText size="xs" color="subdued">
                <p>
                    {startupProblems.length || warnings.length
                        || (recorded && fixHints[recorded.operation])
                        ? 'Problems'
                        : 'No problem found'}
                </p>
            </EuiText>

            {startupProblems.map((problem) => (
                <EuiCallOut
                    key={problem}
                    color="danger"
                    iconType="alert"
                    title="The node cannot start"
                    size="s"
                >
                    <p>{problem}</p>
                </EuiCallOut>
            ))}

            {recorded && fixHints[recorded.operation] ? (
                <EuiCallOut
                    color="primary"
                    iconType="help"
                    title="What to do"
                    size="s"
                >
                    <p>{fixHints[recorded.operation]}</p>
                </EuiCallOut>
            ) : null}

            {warnings.map((warning) => (
                <EuiCallOut
                    key={warning}
                    color="warning"
                    iconType="help"
                    title="Check the registry"
                    size="s"
                >
                    <p>{warning}</p>
                </EuiCallOut>
            ))}

            {connection?.history?.length ? (
                <>
                    <EuiSpacer size="s" />
                    <EuiText size="xs" color="subdued">
                        <p>Earlier states, most recent first</p>
                    </EuiText>
                    {connection.history.slice(0, historyShown).map((entry) => (
                        <DetailItem
                            key={`${entry.updated_at}-${entry.operation}`}
                            label={formatDateTime(entry.updated_at)}
                            value={`${entry.state}`
                                + `${entry.operation ? ` - ${entry.operation}` : ''}`}
                        />
                    ))}
                </>
            ) : null}
        </section>
    )
}

const mapStateToProps = (state) => ({
    connection: state.certificates.connection,
    connectionError: state.certificates.connectionError,
    certificateStatus: state.certificates.status,
    certificateError: state.certificates.error,
})

const mapDispatchToProps = (dispatch) => ({
    fetchConnectionState: () => dispatch(fetchConnectionState()),
    fetchCertificateStatus: () => dispatch(fetchCertificateStatus()),
})

export default connect(mapStateToProps, mapDispatchToProps)(ConnectionStatus)
