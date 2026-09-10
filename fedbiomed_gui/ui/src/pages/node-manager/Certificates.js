import React from 'react'
import {connect} from 'react-redux'
import {
    EuiButton,
    EuiButtonEmpty,
    EuiButtonIcon,
    EuiCallOut,
    EuiCodeBlock,
    EuiFieldText,
    EuiFilePicker,
    EuiFlexGroup,
    EuiFlexItem,
    EuiFormRow,
    EuiIcon,
    EuiModal,
    EuiModalBody,
    EuiModalFooter,
    EuiModalHeader,
    EuiModalHeaderTitle,
    EuiPopover,
    EuiSpacer,
    EuiText,
    EuiTextArea,
    EuiTitle,
} from '@elastic/eui'

import Popup from '../../components/common/Popup'
import {
    deleteCertificate,
    downloadOwnCertificate,
    fetchCertificateStatus,
    generateOwnCertificate,
    inspectCertificate,
    registerCertificate,
    replaceOwnCertificate,
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
                <Popup
                    icon="alert"
                    iconColor="warning"
                    title={
                        confirming === ownCertificateActions.generate
                            ? 'Generate a new certificate?'
                            : 'Replace this node\'s certificate?'
                    }
                    onClose={() => setConfirming(null)}
                    cancelText="Keep the current one"
                    confirmText={
                        confirming === ownCertificateActions.generate
                            ? 'Generate it'
                            : 'Replace it'
                    }
                    confirmColor="danger"
                    onConfirm={
                        confirming === ownCertificateActions.generate
                            ? generate
                            : replace
                    }
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
                </Popup>
            ) : null}
        </>
    )
}

/**
 * The researcher certificate this node trusts, shown in the box. With none
 * registered, the empty box takes the certificate to register; a registered one
 * is deleted before another is added. A node holds a single researcher
 * certificate: more than one registered is an error, fixed from the command
 * line.
 */
const ResearcherCertificates = ({
    registered,
    loaded,
    writing,
    onInspect,
    onRegister,
    onDelete,
    onDraftChange,
}) => {
    // The certificate being added, from its first change until it is
    // registered; without one, the box shows the registered certificate
    const [draft, setDraft] = React.useState(null)
    const [editing, setEditing] = React.useState(false)
    const [inspecting, setInspecting] = React.useState(false)
    const [inspected, setInspected] = React.useState(null)
    const [inspectError, setInspectError] = React.useState(null)
    const [componentId, setComponentId] = React.useState('')
    const [infoOpen, setInfoOpen] = React.useState(false)
    const [confirmingDelete, setConfirmingDelete] = React.useState(false)

    const current = registered.length === 1 ? registered[0] : null
    const pending = draft !== null
    // With nothing registered the empty box is open for the first certificate,
    // once the status says so: until then, one may still be registered
    const open = loaded && !current && !pending
    const editingNow = editing || open
    // What the header of the box and (i) describe: the draft once inspected,
    // otherwise what is registered
    const described = pending ? inspected : current
    // Recovered from the certificate when Fed-BioMed issued it, typed otherwise.
    // A draft is only inspected once its edit is done.
    const needsComponentId = Boolean(inspected) && !inspected.component_id
    const registerComponentId = inspected?.component_id || componentId.trim()
    const canRegister = Boolean(inspected) && Boolean(registerComponentId)

    // The header of the box: whose certificate it is, whether it is the
    // registered one, and until when it is valid
    const identityTitle = described
        ? described.component_id
            || componentId.trim()
            || 'Component id to enter below'
        : editing ? 'New certificate'
            : loaded ? 'No certificate is registered'
                : 'Reading the registered certificate…'
    const days = described?.expires_in_days
    // A certificate is refused once expired, but one registered earlier expires
    const expired = days < 0
    const expiryDate = described?.cert_not_after
        ? new Date(described.cert_not_after).toLocaleDateString(undefined, {
            day: 'numeric',
            month: 'short',
            year: 'numeric',
        })
        : null
    const identityDetail = !expiryDate
        ? null
        : expired
            ? `Expired ${expiryDate} — request a renewed certificate from `
                + 'the researcher.'
            : `Expires ${expiryDate} · in ${days.toLocaleString()} `
                + `${days === 1 ? 'day' : 'days'}`
                + (described.expiring_soon
                    ? ' — request a renewed certificate from the researcher.'
                    : '')
    // Shown in the page's status pill, so in the page's colours
    const [statusIcon, statusClass] = expired
        ? ['alert', 'danger']
        : described?.expiring_soon
            ? ['clock', 'warning']
            : pending
                ? ['document', 'neutral']
                : ['check', 'success']

    // The window asks before closing on a certificate not registered yet; an
    // empty box holds nothing to lose
    const hasDraft = Boolean(draft?.trim())
    React.useEffect(() => {
        onDraftChange(hasDraft)
    }, [onDraftChange, hasDraft])

    // Any change to the draft makes it one that is not checked yet
    const edit = (text) => {
        setDraft(text)
        setInspected(null)
        setInspectError(null)
        setInfoOpen(false)
        setEditing(true)
    }

    // Done and picking a file end the edit the same way: the node reads the text
    // back, and names the component when the certificate carries it
    const done = async (text) => {
        setDraft(text)
        setInspectError(null)
        setInspecting(true)
        const result = await onInspect(text)
        setInspecting(false)
        if (result.error) {
            setInspectError(result.error)
            setEditing(true)
            return
        }

        setInspected(result)
        setEditing(false)
    }

    const register = async () => {
        const registeredOk = await onRegister(draft, {
            componentId: inspected.component_id ? null : registerComponentId,
        })
        if (registeredOk) {
            // The box shows the registered certificate again, now this one
            setDraft(null)
            setInspected(null)
            setComponentId('')
        }
    }

    if (registered.length > 1) {
        return (
            <EuiCallOut
                color="danger"
                iconType="alert"
                title="More than one certificate is registered"
                size="s"
            >
                <p>
                    A node holds a single researcher certificate, but{' '}
                    {registered.map((entry) => entry.component_id).join(', ')}
                    {' '}are registered. Remove the extra ones with{' '}
                    <code>fedbiomed node certificate delete</code>, then reopen
                    this window.
                </p>
            </EuiCallOut>
        )
    }

    return (
        <>
            <EuiText size="s" color="subdued">
                <p>
                    The certificate this node trusts for its researcher. Changes
                    are written at once; restart the node to use them.
                </p>
            </EuiText>
            <EuiSpacer size="m" />

            {described ? (
                <div
                    className={
                        'node-certificate-identity'
                        + (expired
                            ? ' danger'
                            : described.expiring_soon ? ' warning' : '')
                    }
                >
                    <EuiFlexGroup
                        gutterSize="s"
                        alignItems="center"
                        responsive={false}
                    >
                        <EuiFlexItem>
                            <EuiTitle size="xxs">
                                <h4>{identityTitle}</h4>
                            </EuiTitle>
                            <EuiSpacer size="xs" />
                            <EuiFlexGroup
                                gutterSize="s"
                                alignItems="center"
                                responsive={false}
                                wrap
                            >
                                <EuiFlexItem grow={false}>
                                    <span
                                        className={
                                            'node-management-status-pill '
                                            + `${statusClass} `
                                            + 'node-certificate-identity-state'
                                        }
                                    >
                                        <EuiIcon type={statusIcon} size="s" />
                                        {pending
                                            ? 'Not registered yet'
                                            : 'Registered'}
                                    </span>
                                </EuiFlexItem>
                                {identityDetail ? (
                                    <EuiFlexItem>
                                        <EuiText size="xs">
                                            {identityDetail}
                                        </EuiText>
                                    </EuiFlexItem>
                                ) : null}
                            </EuiFlexGroup>
                        </EuiFlexItem>
                        <EuiFlexItem grow={false}>
                            <EuiPopover
                                button={
                                    <EuiButtonIcon
                                        iconType="iInCircle"
                                        aria-label="Certificate details"
                                        title="Certificate details"
                                        onClick={() => setInfoOpen(!infoOpen)}
                                    />
                                }
                                isOpen={infoOpen}
                                closePopover={() => setInfoOpen(false)}
                                anchorPosition="leftUp"
                            >
                                <EuiCodeBlock
                                    language="json"
                                    fontSize="s"
                                    paddingSize="s"
                                    overflowHeight={400}
                                    isCopyable
                                >
                                    {/* The text itself is in the box */}
                                    {JSON.stringify(
                                        {...described, certificate: undefined},
                                        null,
                                        2
                                    )}
                                </EuiCodeBlock>
                            </EuiPopover>
                        </EuiFlexItem>
                    </EuiFlexGroup>
                </div>
            ) : (
                <EuiTitle size="xxs">
                    <h4>{identityTitle}</h4>
                </EuiTitle>
            )}
            <EuiSpacer size="s" />
            <EuiFormRow
                isInvalid={Boolean(inspectError)}
                error={inspectError}
                fullWidth
            >
                <EuiTextArea
                    className="node-certificate-pem"
                    fullWidth
                    rows={8}
                    readOnly={!pending && !open}
                    isInvalid={Boolean(inspectError)}
                    placeholder={
                        editingNow
                            ? 'Paste the certificate, or load its .pem file'
                            : undefined
                    }
                    value={pending ? draft : current?.certificate || ''}
                    onChange={(event) => edit(event.target.value)}
                />
            </EuiFormRow>

            {needsComponentId ? (
                <EuiFormRow
                    label="Component id"
                    helpText="This certificate does not name its component: enter the researcher's id."
                    fullWidth
                >
                    <EuiFieldText
                        fullWidth
                        placeholder="RESEARCHER_&lt;uuid&gt;"
                        value={componentId}
                        onChange={(event) => setComponentId(event.target.value)}
                    />
                </EuiFormRow>
            ) : null}
            <EuiSpacer size="s" />

            <EuiFlexGroup
                gutterSize="s"
                alignItems="center"
                responsive={false}
                wrap
            >
                {editingNow ? (
                    <>
                        <EuiFlexItem>
                            <EuiFilePicker
                                compressed
                                display="default"
                                initialPromptText="Load a .pem file"
                                accept=".pem,.crt,.cert"
                                onChange={readFileInto(done)}
                            />
                        </EuiFlexItem>
                        <EuiFlexItem grow={false}>
                            <EuiButton
                                size="s"
                                fill
                                iconType="check"
                                isLoading={inspecting}
                                isDisabled={!draft?.trim()}
                                onClick={() => done(draft)}
                            >
                                Done
                            </EuiButton>
                        </EuiFlexItem>
                    </>
                ) : pending ? (
                    <EuiFlexItem grow={false}>
                        <EuiButton
                            size="s"
                            fill
                            iconType="plusInCircle"
                            isLoading={writing}
                            isDisabled={!canRegister}
                            onClick={register}
                        >
                            Register
                        </EuiButton>
                    </EuiFlexItem>
                ) : current ? (
                    <EuiFlexItem grow={false}>
                        <EuiButton
                            size="s"
                            color="danger"
                            iconType="trash"
                            isDisabled={writing}
                            onClick={() => setConfirmingDelete(true)}
                        >
                            Delete
                        </EuiButton>
                    </EuiFlexItem>
                ) : null}
            </EuiFlexGroup>

            {confirmingDelete ? (
                <Popup
                    icon="trash"
                    iconColor="danger"
                    title="Delete the researcher certificate?"
                    onClose={() => setConfirmingDelete(false)}
                    cancelText="Keep it"
                    confirmText="Delete"
                    confirmColor="danger"
                    onConfirm={() => {
                        onDelete(current.component_id)
                        setConfirmingDelete(false)
                    }}
                >
                    <p>
                        The node stops trusting the certificate of{' '}
                        <code>{current.component_id}</code>. A running node
                        keeps trusting it until it is restarted.
                    </p>
                    <p>
                        With no researcher certificate registered, a node that
                        requires mutual authentication refuses to start.
                    </p>
                </Popup>
            ) : null}
        </>
    )
}

/**
 * Shared frame for the certificate windows. The status is read as the window
 * opens; what a write did is reported in the global result popup. Closing on a
 * draft that is not written yet asks first.
 */
const CertificateWindow = ({
    title,
    notice,
    onClose,
    hasDraft = false,
    error,
    fetchCertificateStatus,
    children,
}) => {
    const [confirmingClose, setConfirmingClose] = React.useState(false)

    React.useEffect(() => {
        fetchCertificateStatus()
    }, [fetchCertificateStatus])

    const close = () => (hasDraft ? setConfirmingClose(true) : onClose())

    return (
        <EuiModal className="node-certificate-modal" onClose={close}>
            <EuiModalHeader>
                <EuiModalHeaderTitle>{title}</EuiModalHeaderTitle>
            </EuiModalHeader>
            <EuiModalBody>
                {error ? (
                    <div className="node-management-alert error">
                        <EuiIcon type="alert" />
                        <span>{error}</span>
                    </div>
                ) : null}
                {notice ? (
                    <>
                        <EuiCallOut
                            color="primary"
                            iconType="iInCircle"
                            title="Everything here is written straight away"
                            size="s"
                        >
                            <p>{notice}</p>
                        </EuiCallOut>
                        <EuiSpacer size="m" />
                    </>
                ) : null}
                {children}
            </EuiModalBody>
            <EuiModalFooter>
                <EuiButtonEmpty onClick={close}>Close</EuiButtonEmpty>
            </EuiModalFooter>

            {confirmingClose ? (
                <Popup
                    icon="alert"
                    iconColor="warning"
                    title="Discard the certificate you have not registered?"
                    onClose={() => setConfirmingClose(false)}
                    cancelText="Keep editing"
                    confirmText="Discard"
                    confirmColor="danger"
                    onConfirm={onClose}
                />
            ) : null}
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
        fetchCertificateStatus={props.fetchCertificateStatus}
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

const ResearcherCertificateWindow = (props) => {
    const [hasDraft, setHasDraft] = React.useState(false)

    return (
        <CertificateWindow
            title="Researcher certificate"
            onClose={props.onClose}
            hasDraft={hasDraft}
            error={props.error}
            fetchCertificateStatus={props.fetchCertificateStatus}
        >
            <ResearcherCertificates
                registered={props.certificateStatus?.registered || []}
                loaded={Boolean(props.certificateStatus)}
                writing={props.writing}
                onInspect={props.inspectCertificate}
                onRegister={props.registerCertificate}
                onDelete={props.deleteCertificate}
                onDraftChange={setHasDraft}
            />
        </CertificateWindow>
    )
}

const mapStateToProps = (state) => ({
    certificateStatus: state.certificates.status,
    writing: state.certificates.writing,
    error: state.certificates.error,
})

const mapDispatchToProps = (dispatch) => ({
    fetchCertificateStatus: () => dispatch(fetchCertificateStatus()),
    inspectCertificate: (certificate) => dispatch(
        inspectCertificate(certificate)
    ),
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
})

export const OwnCertificateModal = connect(
    mapStateToProps,
    mapDispatchToProps
)(OwnCertificateWindow)

export const ResearcherCertificateModal = connect(
    mapStateToProps,
    mapDispatchToProps
)(ResearcherCertificateWindow)
