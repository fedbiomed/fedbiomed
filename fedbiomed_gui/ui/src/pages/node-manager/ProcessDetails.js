import React from 'react'
import {
    EuiButton,
    EuiIcon,
    EuiModal,
    EuiModalBody,
    EuiModalFooter,
    EuiModalHeader,
    EuiModalHeaderTitle,
    EuiToolTip,
} from '@elastic/eui'

const ProcessDetails = ({
    actor,
    actorName,
    currentState,
    DetailItem,
    formatDateTime,
    formatValue,
    isActorModalVisible,
    lastRefresh,
    onCloseActorModal,
    onShowActorModal,
    processState,
    runningFor,
    stateTone,
    statusMessage,
    StatusPill,
    SummaryCard,
}) => {
    return (
        <>
            <section className="node-management-card node-management-process">
                <div className="node-management-section-header">
                    <div className="node-management-section-heading">
                        <span className="node-management-section-icon">
                            <EuiIcon type="stats" size="l" />
                        </span>
                        <div>
                            <h2>Process Details</h2>
                            <p>Current process information and status</p>
                        </div>
                    </div>
                    <div className="node-management-process-header-actions">
                        <EuiToolTip
                            position="bottom"
                            title="Last Action Actor"
                            content={(
                                <div className="node-management-actor-tooltip">
                                    <span>
                                        <strong>Name:</strong>{' '}
                                        {formatValue(
                                            actorName || actor.local_username
                                        )}
                                    </span>
                                    <span>
                                        <strong>Email:</strong>{' '}
                                        {formatValue(actor.email)}
                                    </span>
                                    <span>
                                        <strong>Role:</strong>{' '}
                                        {formatValue(actor.role)}
                                    </span>
                                    <span>
                                        <strong>Source:</strong>{' '}
                                        {formatValue(actor.source)}
                                    </span>
                                    <span>
                                        <strong>User ID:</strong>{' '}
                                        {formatValue(actor.user_id)}
                                    </span>
                                </div>
                            )}
                        >
                            <EuiButton
                                size="s"
                                iconType="user"
                                onClick={onShowActorModal}
                            >
                                View actor
                            </EuiButton>
                        </EuiToolTip>
                        <StatusPill state={currentState} />
                    </div>
                </div>

                <div className="node-management-process-content">
                    <aside className="node-management-summary">
                        <SummaryCard
                            className="uptime"
                            label="Uptime"
                            value={runningFor}
                            description="Running for"
                        />
                        <SummaryCard
                            className="pid"
                            label="PID"
                            value={processState?.pid}
                            description="Process ID"
                        />
                    </aside>

                    <div className="node-management-details-grid">
                        <DetailItem
                            icon="tokenKey"
                            label="Node ID"
                            value={processState?.node_id}
                        />
                        <DetailItem
                            icon="calendar"
                            label="Started At"
                            value={formatDateTime(processState?.started_at)}
                        />
                        <DetailItem
                            icon="user"
                            label="Node Name"
                            value={processState?.node_name}
                        />
                        <DetailItem
                            icon="refresh"
                            label="Last Refresh"
                            value={formatDateTime(lastRefresh)}
                        />
                        <DetailItem
                            icon="iInCircle"
                            label="State"
                            valueContent={<StatusPill state={currentState} />}
                        />
                        <DetailItem
                            icon="calendar"
                            label="Updated At"
                            value={formatDateTime(processState?.updated_at)}
                        />
                        <DetailItem
                            icon="clock"
                            label="Running For"
                            value={runningFor}
                        />
                        <DetailItem
                            icon="clock"
                            label="Stopped At"
                            value={formatDateTime(processState?.stopped_at)}
                        />
                        <DetailItem
                            icon="play"
                            label="Action"
                            value={processState?.action}
                        />
                        <DetailItem
                            icon="console"
                            label="Exit Code"
                            value={processState?.exit_code}
                        />
                        <DetailItem
                            icon="flag"
                            label="Reason"
                            value={processState?.reason}
                        />
                        <DetailItem
                            icon="compute"
                            label="GPU"
                            value={processState?.node_args?.gpu}
                        />
                        <DetailItem
                            icon="number"
                            label="GPU Number"
                            value={processState?.node_args?.gpu_num}
                        />
                        <DetailItem
                            icon="check"
                            label="GPU Only"
                            value={processState?.node_args?.gpu_only}
                        />
                        <DetailItem
                            icon="bug"
                            label="Debug"
                            value={processState?.node_args?.debug}
                        />
                        <DetailItem
                            icon="desktop"
                            label="Background"
                            value={processState?.background}
                        />
                    </div>
                </div>

                <div className={`node-management-status-banner ${
                    stateTone(currentState)
                }`}>
                    <span className="node-management-status-banner-icon">
                        <EuiIcon type="iInCircle" size="l" />
                    </span>
                    <div>
                        <strong>Process Status</strong>
                        <p>{statusMessage}</p>
                    </div>
                </div>
            </section>

            {isActorModalVisible ? (
                <EuiModal
                    className="node-management-actor-modal"
                    onClose={onCloseActorModal}
                >
                    <EuiModalHeader>
                        <EuiModalHeaderTitle>
                            Last Action Actor
                        </EuiModalHeaderTitle>
                    </EuiModalHeader>
                    <EuiModalBody>
                        <p className="node-management-actor-description">
                            User or process responsible for the latest action
                        </p>
                        <div className="node-management-actor-grid">
                            <DetailItem
                                icon="user"
                                label="Name"
                                value={actorName || actor.local_username}
                            />
                            <DetailItem
                                icon="email"
                                label="Email"
                                value={actor.email}
                            />
                            <DetailItem
                                icon="users"
                                label="Role"
                                value={actor.role}
                            />
                            <DetailItem
                                icon="inputOutput"
                                label="Source"
                                value={actor.source}
                            />
                            <DetailItem
                                icon="tokenKey"
                                label="User ID"
                                value={actor.user_id}
                            />
                        </div>
                    </EuiModalBody>
                    <EuiModalFooter>
                        <EuiButton fill onClick={onCloseActorModal}>
                            Close
                        </EuiButton>
                    </EuiModalFooter>
                </EuiModal>
            ) : null}
        </>
    )
}

export default ProcessDetails
