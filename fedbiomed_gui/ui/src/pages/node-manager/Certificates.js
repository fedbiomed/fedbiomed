import React from 'react'
import {connect} from 'react-redux'
import {
    EuiBadge,
    EuiButton,
    EuiButtonEmpty,
    EuiCallOut,
    EuiConfirmModal,
    EuiFilePicker,
    EuiFlexGroup,
    EuiFlexItem,
    EuiFormRow,
    EuiIcon,
    EuiSpacer,
    EuiSwitch,
    EuiText,
    EuiTextArea,
} from '@elastic/eui'

import {
    deleteCertificate,
    downloadOwnCertificate,
    fetchCertificateStatus,
    fetchConnectionState,
    registerCertificate,
    resetCertificateMessages,
} from '../../store/actions/certificatesActions'
import {writeNodeConfigSection} from '../../store/actions/nodeConfigActions'

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

    return state.identity_verified === true
        ? {
            color: 'success',
            label: 'Mutual TLS, identity verified',
            detail: 'The researcher verified this node\'s identity.',
        }
        : {
            color: 'warning',
            label: 'Mutual TLS, verification unconfirmed',
            detail: 'The researcher certificate is pinned, but whether it '
                + 'verifies node identity could not be determined.',
        }
}

const DetailItem = ({label, value}) => (
    <div className="node-management-detail-item">
        <span className="node-management-detail-label">{label}</span>
        <span className="node-management-detail-value">{value}</span>
    </div>
)

const CertificateDetails = ({certificate}) => (
    <div className="node-management-details-grid">
        <DetailItem label="Subject" value={formatValue(certificate.cert_subject)} />
        <DetailItem label="Issuer" value={formatValue(certificate.cert_issuer)} />
        <DetailItem label="Serial" value={formatValue(certificate.cert_serial)} />
        <DetailItem
            label="Expires"
            value={
                certificate.cert_not_after
                    ? `${formatDateTime(certificate.cert_not_after)}`
                        + ` (${certificate.expires_in_days} days)`
                    : emptyValue
            }
        />
        <DetailItem
            label="Valid for"
            value={
                certificate.san?.length
                    ? certificate.san.join(', ')
                    : 'no host (a client credential)'
            }
        />
        <DetailItem
            label="Fingerprint"
            value={formatValue(certificate.fingerprint)}
        />
    </div>
)

const Certificates = (props) => {
    const {
        certificateStatus,
        connection,
        loading,
        writing,
        error,
        connectionError,
        writeError,
        configWriteError,
        successMessage,
        requiresRestart,
        fetchCertificateStatus,
        fetchConnectionState,
    } = props

    const [certificate, setCertificate] = React.useState('')
    const [conflictingCertificate, setConflictingCertificate] = React.useState(
        null
    )
    const [partyToDelete, setPartyToDelete] = React.useState(null)

    React.useEffect(() => {
        fetchCertificateStatus()
        fetchConnectionState()
    }, [fetchCertificateStatus, fetchConnectionState])

    const refresh = () => {
        props.resetCertificateMessages()
        fetchCertificateStatus()
        fetchConnectionState()
    }

    const readCertificateFile = (files) => {
        const file = files?.[0]
        if (!file) {
            return
        }

        const reader = new FileReader()
        reader.onload = () => setCertificate(String(reader.result))
        reader.readAsText(file)
    }

    const register = async ({upsert = false} = {}) => {
        const registered = await props.registerCertificate(certificate, {upsert})
        if (registered) {
            setCertificate('')
            setConflictingCertificate(null)
        } else if (!upsert) {
            // The party is already registered; replacing it is the user's call
            setConflictingCertificate(certificate)
        }
    }

    const toggleMtls = async (enabled) => {
        props.resetCertificateMessages()
        await props.writeNodeConfigSection(
            'authentication',
            {mutual_authentication: enabled},
            {mutual_authentication: certificateStatus?.mtls_enabled},
        )
        fetchCertificateStatus()
    }

    const summary = connectionSummary(connection)
    const recorded = connection?.state
    const registered = certificateStatus?.registered || []
    const ownCertificate = certificateStatus?.certificate

    return (
        <section className="node-management-card node-management-certificates">
            <div className="node-management-section-header">
                <div className="node-management-section-heading">
                    <span className="node-management-section-icon">
                        <EuiIcon type="lock" size="l" />
                    </span>
                    <div>
                        <h2>Certificates</h2>
                        <p>
                            Mutual TLS with the researcher, and the certificates
                            this node trusts
                        </p>
                    </div>
                </div>
                <div className="node-management-process-header-actions">
                    <EuiButton
                        size="s"
                        iconType="refresh"
                        onClick={refresh}
                        isLoading={loading}
                    >
                        Refresh
                    </EuiButton>
                </div>
            </div>

            {[error, connectionError, writeError, configWriteError]
                .filter(Boolean)
                .map((message) => (
                    <div className="node-management-alert error" key={message}>
                        <EuiIcon type="alert" />
                        <span>{message}</span>
                    </div>
                ))}

            {successMessage ? (
                <div className="node-management-alert info">
                    <EuiIcon type="check" />
                    <span>{successMessage}</span>
                </div>
            ) : null}

            {requiresRestart ? (
                <div className="node-management-alert warning">
                    <EuiIcon type="alert" />
                    <span>
                        The node reads its certificates when it starts, so it has
                        to be restarted for this change to take effect.
                    </span>
                </div>
            ) : null}

            {(certificateStatus?.startup_problems || []).map((problem) => (
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

            {(certificateStatus?.warnings || []).map((warning) => (
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

            <EuiSpacer size="m" />

            <h3>Connection to the researcher</h3>
            <EuiSpacer size="s" />
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

            <EuiSpacer size="l" />

            <h3>Mutual TLS</h3>
            <EuiSpacer size="s" />
            <EuiSwitch
                label="Require and verify certificates on both sides"
                checked={Boolean(certificateStatus?.mtls_enabled)}
                disabled={writing}
                onChange={(event) => toggleMtls(event.target.checked)}
            />

            <EuiSpacer size="l" />

            <h3>This node&apos;s certificate</h3>
            <EuiText size="s" color="subdued">
                <p>
                    Send this to the researcher, which registers it to
                    recognise this node. The private key never leaves the node.
                </p>
            </EuiText>
            <EuiSpacer size="s" />

            {ownCertificate?.error ? (
                <div className="node-management-alert error">
                    <EuiIcon type="alert" />
                    <span>{ownCertificate.error}</span>
                </div>
            ) : ownCertificate ? (
                <>
                    <DetailItem
                        label="Party id"
                        value={formatValue(ownCertificate.party_id)}
                    />
                    <CertificateDetails certificate={ownCertificate} />
                    <EuiButton
                        size="s"
                        iconType="download"
                        onClick={props.downloadOwnCertificate}
                    >
                        Download certificate
                    </EuiButton>
                </>
            ) : null}

            <EuiSpacer size="l" />

            <h3>Registered researcher certificate</h3>
            <EuiText size="s" color="subdued">
                <p>
                    A node registers the certificate of the researcher it
                    connects to, and pins it under mutual TLS.
                </p>
            </EuiText>
            <EuiSpacer size="s" />

            {registered.length ? (
                registered.map((entry) => (
                    <div
                        className="node-management-registered-certificate"
                        key={entry.party_id}
                    >
                        <DetailItem
                            label="Party id"
                            value={`${entry.party_id} (${entry.component})`}
                        />
                        <CertificateDetails certificate={entry} />
                        <EuiButtonEmpty
                            size="s"
                            color="danger"
                            iconType="trash"
                            isDisabled={writing}
                            onClick={() => setPartyToDelete(entry.party_id)}
                        >
                            Delete
                        </EuiButtonEmpty>
                    </div>
                ))
            ) : (
                <EuiText size="s">
                    <p>No certificate is registered.</p>
                </EuiText>
            )}

            <EuiSpacer size="m" />

            <EuiFormRow
                label="Register a certificate"
                helpText="Paste the certificate the researcher sent, or pick the file it came in."
                fullWidth
            >
                <EuiTextArea
                    fullWidth
                    rows={6}
                    placeholder="-----BEGIN CERTIFICATE-----"
                    value={certificate}
                    onChange={(event) => setCertificate(event.target.value)}
                />
            </EuiFormRow>
            <EuiFilePicker
                initialPromptText="Select a .pem file"
                display="default"
                accept=".pem,.crt,.cert"
                onChange={readCertificateFile}
            />
            <EuiSpacer size="s" />
            <EuiButton
                size="s"
                fill
                iconType="plusInCircle"
                isLoading={writing}
                isDisabled={!certificate.trim()}
                onClick={() => register()}
            >
                Register
            </EuiButton>

            {conflictingCertificate ? (
                <EuiConfirmModal
                    title="Replace the registered certificate?"
                    onCancel={() => setConflictingCertificate(null)}
                    onConfirm={() => register({upsert: true})}
                    cancelButtonText="Keep the current one"
                    confirmButtonText="Replace it"
                    buttonColor="danger"
                >
                    <p>
                        This party already has a certificate registered.
                        Replacing it means the node trusts the new one only:
                        do it when the party renewed its certificate, and check
                        that it came from them.
                    </p>
                </EuiConfirmModal>
            ) : null}

            {partyToDelete ? (
                <EuiConfirmModal
                    title="Delete this certificate?"
                    onCancel={() => setPartyToDelete(null)}
                    onConfirm={() => {
                        props.deleteCertificate(partyToDelete)
                        setPartyToDelete(null)
                    }}
                    cancelButtonText="Keep it"
                    confirmButtonText="Delete"
                    buttonColor="danger"
                >
                    <p>
                        With no researcher certificate registered, a node that
                        requires mutual TLS refuses to start.
                    </p>
                </EuiConfirmModal>
            ) : null}
        </section>
    )
}

const mapStateToProps = (state) => ({
    certificateStatus: state.certificates.status,
    connection: state.certificates.connection,
    loading: state.certificates.loading,
    writing: state.certificates.writing,
    error: state.certificates.error,
    connectionError: state.certificates.connectionError,
    writeError: state.certificates.writeError,
    // The mutual-TLS switch writes through the node configuration
    configWriteError: state.node_config.writeError,
    successMessage: state.certificates.successMessage,
    requiresRestart: state.certificates.requiresRestart,
})

const mapDispatchToProps = (dispatch) => ({
    fetchCertificateStatus: () => dispatch(fetchCertificateStatus()),
    fetchConnectionState: () => dispatch(fetchConnectionState()),
    registerCertificate: (certificate, options) => dispatch(
        registerCertificate(certificate, options)
    ),
    deleteCertificate: (partyId) => dispatch(deleteCertificate(partyId)),
    downloadOwnCertificate: () => dispatch(downloadOwnCertificate()),
    resetCertificateMessages: () => dispatch(resetCertificateMessages()),
    writeNodeConfigSection: (section, values, baseValues) => dispatch(
        writeNodeConfigSection(section, values, baseValues)
    ),
})

export default connect(mapStateToProps, mapDispatchToProps)(Certificates)
