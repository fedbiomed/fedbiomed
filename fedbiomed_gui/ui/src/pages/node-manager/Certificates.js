import React from 'react'
import {connect} from 'react-redux'
import {
    EuiButton,
    EuiButtonEmpty,
    EuiCallOut,
    EuiConfirmModal,
    EuiFieldText,
    EuiFilePicker,
    EuiFormRow,
    EuiIcon,
    EuiModal,
    EuiModalBody,
    EuiModalFooter,
    EuiModalHeader,
    EuiModalHeaderTitle,
    EuiSpacer,
    EuiText,
    EuiTextArea,
} from '@elastic/eui'

import {
    deleteCertificate,
    downloadOwnCertificate,
    fetchCertificateStatus,
    generateOwnCertificate,
    registerCertificate,
    replaceOwnCertificate,
    resetCertificateMessages,
} from '../../store/actions/certificatesActions'

const emptyValue = '-'

// The two ways of updating this node's certificate, each behind a confirmation
const ownCertificateActions = {
    generate: 'generate',
    replace: 'replace',
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

/** Reads a picked file as text into a state setter, for a pasted-or-picked field. */
const readFileInto = (setValue) => (files) => {
    const file = files?.[0]
    if (!file) {
        return
    }

    const reader = new FileReader()
    reader.onload = () => setValue(String(reader.result))
    reader.readAsText(file)
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

/** What the last read and the last write have to say, in the order they happen. */
const CertificateMessages = ({error, writeError, successMessage}) => (
    <>
        {[error, writeError].filter(Boolean).map((message) => (
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
    </>
)

/**
 * This node's own certificate: what it currently presents, and the two ways of
 * updating it. Both write to the paths the node configuration already names.
 */
const OwnCertificate = ({
    ownCertificate,
    writing,
    onDownload,
    onGenerate,
    onReplace,
}) => {
    const [certificate, setCertificate] = React.useState('')
    const [privateKey, setPrivateKey] = React.useState('')
    const [confirming, setConfirming] = React.useState(null)

    const replace = async () => {
        setConfirming(null)
        const replaced = await onReplace(certificate, privateKey)
        if (replaced) {
            setCertificate('')
            setPrivateKey('')
        }
    }

    const generate = async () => {
        setConfirming(null)
        await onGenerate()
    }

    return (
        <>
            <h3>What this node presents</h3>
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
                        label="Component id"
                        value={formatValue(ownCertificate.component_id)}
                    />
                    <CertificateDetails certificate={ownCertificate} />
                    {ownCertificate.expiring_soon ? (
                        <EuiCallOut
                            color="warning"
                            iconType="clock"
                            title="This certificate expires soon"
                            size="s"
                        >
                            <p>
                                Update it below, then have the researcher
                                register the new one.
                            </p>
                        </EuiCallOut>
                    ) : null}
                    <EuiSpacer size="s" />
                    <EuiButton
                        size="s"
                        iconType="download"
                        onClick={onDownload}
                    >
                        Download certificate
                    </EuiButton>
                </>
            ) : null}

            <EuiSpacer size="l" />

            <h3>Generate a new one</h3>
            <EuiText size="s" color="subdued">
                <p>
                    The node issues itself a fresh certificate and private key,
                    written where the current ones are.
                </p>
            </EuiText>
            <EuiSpacer size="s" />
            <EuiButton
                size="s"
                iconType="refresh"
                isLoading={writing}
                onClick={() => setConfirming(ownCertificateActions.generate)}
            >
                Regenerate
            </EuiButton>

            <EuiSpacer size="l" />

            <h3>Replace with your own</h3>
            <EuiText size="s" color="subdued">
                <p>
                    For a certificate issued elsewhere. Both the certificate and
                    its private key are required, and are checked together
                    before either replaces what the node has.
                </p>
            </EuiText>
            <EuiSpacer size="s" />

            <EuiFormRow
                label="Certificate"
                helpText="Paste it, or pick the file it came in."
                fullWidth
            >
                <EuiTextArea
                    fullWidth
                    rows={5}
                    placeholder="-----BEGIN CERTIFICATE-----"
                    value={certificate}
                    onChange={(event) => setCertificate(event.target.value)}
                />
            </EuiFormRow>
            <EuiFilePicker
                initialPromptText="Select the certificate (.pem)"
                display="default"
                accept=".pem,.crt,.cert"
                onChange={readFileInto(setCertificate)}
            />
            <EuiSpacer size="s" />

            <EuiFormRow
                label="Private key"
                helpText="The key this certificate was issued for, unencrypted."
                fullWidth
            >
                <EuiTextArea
                    fullWidth
                    rows={5}
                    placeholder="-----BEGIN PRIVATE KEY-----"
                    value={privateKey}
                    onChange={(event) => setPrivateKey(event.target.value)}
                />
            </EuiFormRow>
            <EuiFilePicker
                initialPromptText="Select the private key (.key)"
                display="default"
                accept=".pem,.key"
                onChange={readFileInto(setPrivateKey)}
            />
            <EuiSpacer size="s" />
            <EuiButton
                size="s"
                fill
                iconType="save"
                isLoading={writing}
                isDisabled={!certificate.trim() || !privateKey.trim()}
                onClick={() => setConfirming(ownCertificateActions.replace)}
            >
                Replace certificate
            </EuiButton>

            {confirming ? (
                <EuiConfirmModal
                    title={
                        confirming === ownCertificateActions.generate
                            ? 'Generate a new certificate?'
                            : 'Replace this node\'s certificate?'
                    }
                    onCancel={() => setConfirming(null)}
                    onConfirm={
                        confirming === ownCertificateActions.generate
                            ? generate
                            : replace
                    }
                    cancelButtonText="Keep the current one"
                    confirmButtonText={
                        confirming === ownCertificateActions.generate
                            ? 'Generate it'
                            : 'Replace it'
                    }
                    buttonColor="danger"
                >
                    <p>
                        This node stops presenting the certificate it presents
                        now. Every component holding the old one has to register
                        the new one, and the node has to be restarted to serve
                        it.
                    </p>
                    <p>
                        The pair being replaced is kept alongside it as a
                        timestamped backup.
                    </p>
                </EuiConfirmModal>
            ) : null}
        </>
    )
}

/**
 * The certificates this node has registered, which under mutual TLS is the
 * researcher's and the one it pins.
 */
const ResearcherCertificates = ({
    registered,
    writing,
    onRegister,
    onDelete,
}) => {
    const [certificate, setCertificate] = React.useState('')
    const [componentId, setComponentId] = React.useState('')
    const [conflictingCertificate, setConflictingCertificate] = React.useState(
        null
    )
    const [componentToDelete, setComponentToDelete] = React.useState(null)

    const register = async ({upsert = false} = {}) => {
        const registeredOk = await onRegister(certificate, {
            upsert,
            componentId: componentId.trim() || null,
        })
        if (registeredOk) {
            setCertificate('')
            setComponentId('')
            setConflictingCertificate(null)
        } else if (!upsert) {
            // The component is already registered; replacing it is the user's call
            setConflictingCertificate(certificate)
        }
    }

    return (
        <>
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
                        key={entry.component_id}
                    >
                        <DetailItem
                            label="Component id"
                            value={entry.component_id}
                        />
                        <CertificateDetails certificate={entry} />
                        <EuiButtonEmpty
                            size="s"
                            color="danger"
                            iconType="trash"
                            isDisabled={writing}
                            onClick={() => setComponentToDelete(entry.component_id)}
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
                onChange={readFileInto(setCertificate)}
            />
            <EuiSpacer size="s" />
            <EuiFormRow
                label="Component id"
                helpText="Only for a certificate that does not name the component in its CN= field. Leave it empty otherwise."
                fullWidth
            >
                <EuiFieldText
                    fullWidth
                    placeholder="RESEARCHER_&lt;uuid&gt;"
                    value={componentId}
                    onChange={(event) => setComponentId(event.target.value)}
                />
            </EuiFormRow>
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
                        This component already has a certificate registered.
                        Replacing it means the node trusts the new one only:
                        do it when the component renewed its certificate, and
                        check that it came from them.
                    </p>
                </EuiConfirmModal>
            ) : null}

            {componentToDelete ? (
                <EuiConfirmModal
                    title="Delete this certificate?"
                    onCancel={() => setComponentToDelete(null)}
                    onConfirm={() => {
                        onDelete(componentToDelete)
                        setComponentToDelete(null)
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
        </>
    )
}

/**
 * Shared frame for the certificate windows: the status is read as the window
 * opens, and the messages a write left behind are cleared as it closes.
 */
const CertificateWindow = ({
    title,
    notice,
    onClose,
    error,
    writeError,
    successMessage,
    fetchCertificateStatus,
    resetCertificateMessages,
    children,
}) => {
    React.useEffect(() => {
        fetchCertificateStatus()
    }, [fetchCertificateStatus])

    const close = () => {
        resetCertificateMessages()
        onClose()
    }

    return (
        <EuiModal className="node-certificate-modal" onClose={close}>
            <EuiModalHeader>
                <EuiModalHeaderTitle>{title}</EuiModalHeaderTitle>
            </EuiModalHeader>
            <EuiModalBody>
                <CertificateMessages
                    error={error}
                    writeError={writeError}
                    successMessage={successMessage}
                />
                <EuiCallOut
                    color="primary"
                    iconType="iInCircle"
                    title="Everything here is written straight away"
                    size="s"
                >
                    <p>{notice}</p>
                </EuiCallOut>
                <EuiSpacer size="m" />
                {children}
            </EuiModalBody>
            <EuiModalFooter>
                <EuiButton onClick={close} fill>
                    Close
                </EuiButton>
            </EuiModalFooter>
        </EuiModal>
    )
}

const OwnCertificateWindow = (props) => (
    <CertificateWindow
        title="This node's certificate"
        notice={
            'This window does not take part in the unsaved changes of the '
            + 'configuration page: generating or replacing the pair writes it '
            + 'to disk at once, and Reset there does not undo it. The node '
            + 'reads its certificates when it starts, so restart it to serve '
            + 'a new one.'
        }
        onClose={props.onClose}
        error={props.error}
        writeError={props.writeError}
        successMessage={props.successMessage}
        fetchCertificateStatus={props.fetchCertificateStatus}
        resetCertificateMessages={props.resetCertificateMessages}
    >
        <OwnCertificate
            ownCertificate={props.certificateStatus?.certificate}
            writing={props.writing}
            onDownload={props.downloadOwnCertificate}
            onGenerate={props.generateOwnCertificate}
            onReplace={props.replaceOwnCertificate}
        />
    </CertificateWindow>
)

const ResearcherCertificateWindow = (props) => (
    <CertificateWindow
        title="Researcher certificate"
        notice={
            'Registering or deleting a certificate here writes it to disk at '
            + 'once. The Mutual TLS switch behind this window does not: it '
            + 'applies only once you save the configuration, and Reset there '
            + 'discards it while leaving what you register here in place.'
        }
        onClose={props.onClose}
        error={props.error}
        writeError={props.writeError}
        successMessage={props.successMessage}
        fetchCertificateStatus={props.fetchCertificateStatus}
        resetCertificateMessages={props.resetCertificateMessages}
    >
        <ResearcherCertificates
            registered={props.certificateStatus?.registered || []}
            writing={props.writing}
            onRegister={props.registerCertificate}
            onDelete={props.deleteCertificate}
        />
    </CertificateWindow>
)

const mapStateToProps = (state) => ({
    certificateStatus: state.certificates.status,
    writing: state.certificates.writing,
    error: state.certificates.error,
    writeError: state.certificates.writeError,
    successMessage: state.certificates.successMessage,
})

const mapDispatchToProps = (dispatch) => ({
    fetchCertificateStatus: () => dispatch(fetchCertificateStatus()),
    registerCertificate: (certificate, options) => dispatch(
        registerCertificate(certificate, options)
    ),
    deleteCertificate: (componentId) => dispatch(
        deleteCertificate(componentId)
    ),
    downloadOwnCertificate: () => dispatch(downloadOwnCertificate()),
    generateOwnCertificate: () => dispatch(generateOwnCertificate()),
    replaceOwnCertificate: (certificate, privateKey) => dispatch(
        replaceOwnCertificate(certificate, privateKey)
    ),
    resetCertificateMessages: () => dispatch(resetCertificateMessages()),
})

export const OwnCertificateModal = connect(
    mapStateToProps,
    mapDispatchToProps
)(OwnCertificateWindow)

export const ResearcherCertificateModal = connect(
    mapStateToProps,
    mapDispatchToProps
)(ResearcherCertificateWindow)
