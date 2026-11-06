import React from 'react'
import {connect} from 'react-redux'
import {
    EuiBadge,
    EuiButton,
    EuiCallOut,
    EuiIcon,
    EuiSpacer,
    EuiText,
} from '@elastic/eui'

import {
    fetchCertificateStatus,
    fetchConnectionState,
} from '../../store/actions/certificatesActions'

// Enough to see a connection settle or flap, without the whole retention window
const historyShown = 5

// What to do about the state the node last recorded, from the troubleshooting
// table of the mutual authentication guide. Keyed by the event the node recorded.
const fixHints = {
    mtls_handshake_failure:
        'The federation server presents a certificate other than the one pinned '
        + 'here. Register its latest certificate; if that is the one already '
        + 'pinned, treat this as a possible man-in-the-middle.',
    mtls_identity_rejected:
        'The federation server rejected this node\'s identity. Request the '
        + 'server to register the node certificate under the id declared here.',
    mtls_not_enforced_by_researcher:
        'This node requires mutual authentication, which the federation server '
        + 'does not enforce. Disable it here, or request the server to enforce '
        + 'it and register the node certificate.',
    mtls_required_by_researcher:
        'The federation server requires mutual authentication. Enable it here, '
        + 'register the server certificate, and request the server to register '
        + 'the node certificate.',
    mtls_startup_refused:
        'The node refused to start because of its certificates. Resolve the '
        + 'problem above and try again.',
    researcher_unavailable:
        'The federation server did not answer. Check that it is running at the '
        + 'host and port configured here.',
    researcher_failed_name_check:
        'The federation server presents a certificate other than the one '
        + 'registered here. Register its latest certificate and restart; if it '
        + 'is already registered, request the server to reissue it for the '
        + 'hosts nodes connect to.',
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
            detail: 'Connected without mutual authentication: the federation '
                + 'server does not verify this node\'s identity.',
        }
    }

    // The node only reaches a connected state under mutual authentication once
    // the federation server has named it from the certificate it presented, so
    // this is not in doubt.
    return {
        color: 'success',
        label: 'Mutual authentication, identity verified',
        detail: 'The node and the federation server verified each other\'s '
            + 'certificates.',
    }
}

const ConnectionStatus = ({
    connection,
    connectionError,
    certificateStatus,
    certificateError,
    DetailItem,
    fetchConnectionState,
    fetchCertificateStatus,
    formatDateTime,
    formatValue,
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
    // The node is the single judge of these: the severity it reports decides how
    // each one is shown, so the two surfaces cannot disagree about what is wrong.
    const diagnostics = certificateStatus?.diagnostics || []
    const startupProblems = diagnostics.filter((d) => d.severity === 'problem')
    const warnings = diagnostics.filter((d) => d.severity === 'warning')

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
                            Connection to the federation server, its history,
                            and what prevents it
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

            {connection?.stale ? (
                <EuiText size="xs" color="subdued">
                    <p>
                        The node is not running, so this is what it last
                        observed, not what is true now.
                    </p>
                </EuiText>
            ) : null}

            <div className="node-management-details-grid">
                <DetailItem
                    icon="check"
                    label="Connection status"
                    valueContent={
                        <EuiBadge color={summary.color}>{summary.label}</EuiBadge>
                    }
                />
                <DetailItem
                    icon="iInCircle"
                    label="Detail"
                    value={summary.detail}
                />
                {recorded ? (
                    <>
                    <DetailItem
                        icon="globe"
                        label="Federation server"
                        value={`${formatValue(recorded.host)}:`
                            + `${formatValue(recorded.port)}`}
                    />
                    <DetailItem
                        icon="tokenKey"
                        label="Federation server id"
                        value={recorded.researcher_id}
                    />
                    <DetailItem
                        icon="calendar"
                        label="Since"
                        value={formatDateTime(recorded.started_at)}
                    />
                    <DetailItem
                        icon="refresh"
                        label="Last observed"
                        value={formatDateTime(recorded.updated_at)}
                    />
                    <DetailItem
                        icon="alert"
                        label="Last error"
                        value={recorded.last_error}
                    />
                    </>
                ) : null}
            </div>

            <EuiSpacer size="m" />

            <h3 className="node-management-subsection-title">
                {startupProblems.length || warnings.length
                    || (recorded && fixHints[recorded.operation])
                    ? 'Problems'
                    : 'No problem found'}
            </h3>

            <div className="node-management-callout-stack">
                {startupProblems.map((problem) => (
                    <EuiCallOut
                        key={problem.message}
                        color="danger"
                        iconType="alert"
                        title="The node cannot start"
                        size="s"
                    >
                        <p>{problem.message}</p>
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
                        key={warning.message}
                        color="warning"
                        iconType="help"
                        title="Worth fixing"
                        size="s"
                    >
                        <p>{warning.message}</p>
                    </EuiCallOut>
                ))}
            </div>

            {connection?.history?.length ? (
                <>
                    <EuiSpacer size="m" />
                    <h3 className="node-management-subsection-title">
                        Earlier states, most recent first
                    </h3>
                    {connection.history.slice(0, historyShown).map((entry) => (
                        <DetailItem
                            key={`${entry.updated_at}-${entry.operation}`}
                            icon="clock"
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
