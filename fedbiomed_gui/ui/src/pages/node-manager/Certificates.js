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

// The two ways of updating this node's certificate, each behind a confirmation
const ownCertificateActions = {
    generate: 'generate',
    replace: 'replace',
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

/**
 * The header of a certificate box: whose certificate it is, its state in the
 * page's status pill, and until when it is valid, with the details behind (i).
 * With no certificate to describe, just the title.
 */
const CertificateHeader = ({
    title,
    certificate,
    label,
    neutral = false,
    renewHint,
}) => {
    const [infoOpen, setInfoOpen] = React.useState(false)

    // Details left open do not carry over to the next certificate shown
    React.useEffect(() => {
        setInfoOpen(false)
    }, [certificate])

    if (!certificate) {
        return (
            <EuiTitle size="xxs">
                <h4>{title}</h4>
            </EuiTitle>
        )
    }

    const days = certificate.expires_in_days
    // A certificate is refused once expired, but one written earlier expires
    const expired = days < 0
    const expiryDate = certificate.cert_not_after
        ? new Date(certificate.cert_not_after).toLocaleDateString(undefined, {
            day: 'numeric',
            month: 'short',
            year: 'numeric',
        })
        : null
    const detail = !expiryDate
        ? null
        : expired
            ? `Expired ${expiryDate} — ${renewHint}`
            : `Expires ${expiryDate} · in ${days.toLocaleString()} `
                + `${days === 1 ? 'day' : 'days'}`
                + (certificate.expiring_soon ? ` — ${renewHint}` : '')
    // Shown in the page's status pill, so in the page's colours
    const [statusIcon, statusClass] = expired
        ? ['alert', 'danger']
        : certificate.expiring_soon
            ? ['clock', 'warning']
            : neutral
                ? ['document', 'neutral']
                : ['check', 'success']

    return (
        <div
            className={
                'node-certificate-identity'
                + (expired
                    ? ' danger'
                    : certificate.expiring_soon ? ' warning' : '')
            }
        >
            <EuiFlexGroup gutterSize="s" alignItems="center" responsive={false}>
                <EuiFlexItem>
                    <EuiTitle size="xxs">
                        <h4>{title}</h4>
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
                                {label}
                            </span>
                        </EuiFlexItem>
                        {detail ? (
                            <EuiFlexItem>
                                <EuiText size="xs">{detail}</EuiText>
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
                                {...certificate, certificate: undefined},
                                null,
                                2
                            )}
                        </EuiCodeBlock>
                    </EuiPopover>
                </EuiFlexItem>
            </EuiFlexGroup>
        </div>
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

    const title = described
        ? described.component_id
            || componentId.trim()
            || 'Component id to enter below'
        : editing ? 'New certificate'
            : loaded ? 'No certificate is registered'
                : 'Reading the registered certificate…'

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

            <CertificateHeader
                title={title}
                certificate={described}
                label={pending ? 'Not registered yet' : 'Registered'}
                neutral={pending}
                renewHint="request a renewed certificate from the researcher."
            />
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
 * draft that is not written yet asks first. `action` is the window's own button,
 * shown beside Close.
 */
const CertificateWindow = ({
    title,
    onClose,
    hasDraft = false,
    action,
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
                {children}
            </EuiModalBody>
            <EuiModalFooter>
                <EuiButtonEmpty onClick={close}>Close</EuiButtonEmpty>
                {action}
            </EuiModalFooter>

            {confirmingClose ? (
                <Popup
                    icon="alert"
                    iconColor="warning"
                    title="Discard the certificate you have not saved?"
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

/**
 * The certificate this node presents, shown in the box, and the two ways of
 * updating it: the node issues itself a new pair, or takes a certificate and
 * key issued elsewhere. Both write to the paths the node configuration names.
 */
const OwnCertificateWindow = ({
    certificateStatus,
    writing,
    error,
    onClose,
    fetchCertificateStatus,
    downloadOwnCertificate,
    generateOwnCertificate,
    replaceOwnCertificate,
}) => {
    // From Replace until the pair is written, the box takes the certificate
    // replacing the current one, and a second box its private key
    const [replacing, setReplacing] = React.useState(false)
    const [certificate, setCertificate] = React.useState('')
    const [privateKey, setPrivateKey] = React.useState('')
    const [confirming, setConfirming] = React.useState(null)

    const ownCertificate = certificateStatus?.certificate
    const loaded = Boolean(certificateStatus)
    const readable = Boolean(ownCertificate) && !ownCertificate.error
    const title = replacing
        ? 'New certificate'
        : readable
            ? ownCertificate.component_id
            : loaded ? 'No readable certificate' : 'Reading the node certificate…'
    // Closing asks first once there is something typed to lose
    const hasDraft = replacing
        && Boolean(certificate.trim() || privateKey.trim())

    const replace = async () => {
        setConfirming(null)
        const replaced = await replaceOwnCertificate(certificate, privateKey)
        if (replaced) {
            // The box shows the certificate the node presents again, now this one
            setReplacing(false)
            setCertificate('')
            setPrivateKey('')
        }
    }

    const generate = async () => {
        setConfirming(null)
        await generateOwnCertificate()
    }

    return (
        <CertificateWindow
            title="Node certificate"
            onClose={onClose}
            hasDraft={hasDraft}
            action={replacing ? (
                <EuiButton
                    fill
                    iconType="save"
                    isLoading={writing}
                    isDisabled={!certificate.trim() || !privateKey.trim()}
                    onClick={() => setConfirming(ownCertificateActions.replace)}
                >
                    Replace certificate
                </EuiButton>
            ) : null}
            error={error}
            fetchCertificateStatus={fetchCertificateStatus}
        >
            <EuiText size="s" color="subdued">
                <p>
                    The certificate this node presents to the researcher, which
                    registers it. The private key never leaves the node. Changes
                    are written at once; restart the node to use them.
                </p>
            </EuiText>
            <EuiSpacer size="m" />

            {ownCertificate?.error ? (
                <>
                    <div className="node-management-alert error">
                        <EuiIcon type="alert" />
                        <span>{ownCertificate.error}</span>
                    </div>
                    <EuiSpacer size="s" />
                </>
            ) : null}

            <CertificateHeader
                title={title}
                certificate={readable && !replacing ? ownCertificate : null}
                label="In use"
                renewHint={
                    'regenerate or replace it, and request the researcher to '
                    + 'register the new one.'
                }
            />
            <EuiSpacer size="s" />

            {replacing ? (
                <>
                    <EuiFormRow label="Certificate" fullWidth>
                        <EuiTextArea
                            className="node-certificate-pem"
                            fullWidth
                            rows={6}
                            placeholder="Paste the certificate, or load its .pem file"
                            value={certificate}
                            onChange={(event) => setCertificate(event.target.value)}
                        />
                    </EuiFormRow>
                    <EuiFilePicker
                        compressed
                        display="default"
                        initialPromptText="Load the certificate (.pem)"
                        accept=".pem,.crt,.cert"
                        onChange={readFileInto(setCertificate)}
                    />
                    <EuiSpacer size="m" />
                    <EuiFormRow label="Private key" fullWidth>
                        <EuiTextArea
                            className="node-certificate-pem"
                            fullWidth
                            rows={6}
                            placeholder="Paste its private key, or load its .key file"
                            value={privateKey}
                            onChange={(event) => setPrivateKey(event.target.value)}
                        />
                    </EuiFormRow>
                    <EuiFilePicker
                        compressed
                        display="default"
                        initialPromptText="Load the private key (.key)"
                        accept=".pem,.key"
                        onChange={readFileInto(setPrivateKey)}
                    />
                </>
            ) : (
                <>
                    <EuiTextArea
                        className="node-certificate-pem"
                        fullWidth
                        rows={8}
                        readOnly
                        value={readable ? ownCertificate.certificate : ''}
                    />
                    <EuiSpacer size="s" />
                    {loaded ? (
                        <EuiFlexGroup
                            gutterSize="s"
                            alignItems="center"
                            responsive={false}
                            wrap
                        >
                            {readable ? (
                                <EuiFlexItem grow={false}>
                                    <EuiButton
                                        size="s"
                                        iconType="download"
                                        onClick={downloadOwnCertificate}
                                    >
                                        Download
                                    </EuiButton>
                                </EuiFlexItem>
                            ) : null}
                            <EuiFlexItem grow={false}>
                                <EuiButton
                                    size="s"
                                    iconType="refresh"
                                    isLoading={writing}
                                    onClick={() => setConfirming(ownCertificateActions.generate)}
                                >
                                    Regenerate
                                </EuiButton>
                            </EuiFlexItem>
                            <EuiFlexItem grow={false}>
                                <EuiButton
                                    size="s"
                                    iconType="pencil"
                                    isDisabled={writing}
                                    onClick={() => setReplacing(true)}
                                >
                                    Replace
                                </EuiButton>
                            </EuiFlexItem>
                        </EuiFlexGroup>
                    ) : null}
                </>
            )}

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
        </CertificateWindow>
    )
}

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
