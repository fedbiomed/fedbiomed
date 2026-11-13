import React from 'react'
import {
    EuiButton,
    EuiButtonEmpty,
    EuiFlexGroup,
    EuiFlexItem,
    EuiFormRow,
    EuiIcon,
    EuiSelect,
    EuiSpacer,
    EuiText,
} from '@elastic/eui'

const ApplicationLogs = ({
    downloadNodeLogFile,
    formatDateTime,
    formatValue,
    handleLogScroll,
    loadLatestNodeLogs,
    loadOlderNodeLogs,
    logError,
    logFileName,
    logFileOptions,
    logFileSize,
    logFilesError,
    logFilesLoading,
    logHasMore,
    logItems,
    logLastRefresh,
    logLoading,
    logPageSize,
    logScrollRef,
    onLogFileChange,
    onLogPageSizeChange,
}) => {
    return (
        <section className="node-management-card node-management-logs">
            <div className="node-management-section-header">
                <div className="node-management-section-heading">
                    <span className="node-management-section-icon">
                        <EuiIcon type="console" size="l" />
                    </span>
                    <div>
                        <h2>Application Logs</h2>
                        <p>Node runtime log entries from application.log</p>
                    </div>
                </div>
                <div className="node-management-process-header-actions">
                    <EuiButton
                        size="s"
                        iconType="download"
                        onClick={() => downloadNodeLogFile(logFileName)}
                    >
                        Download raw log
                    </EuiButton>
                    <EuiButton
                        size="s"
                        iconType="refresh"
                        onClick={() => loadLatestNodeLogs()}
                        isLoading={logLoading}
                    >
                        Refresh
                    </EuiButton>
                </div>
            </div>

            {logError ? (
                <div className="node-management-alert error">
                    <EuiIcon type="alert" />
                    <span>{logError}</span>
                </div>
            ) : null}

            {logFilesError ? (
                <div className="node-management-alert error">
                    <EuiIcon type="alert" />
                    <span>{logFilesError}</span>
                </div>
            ) : null}

            <div className="node-management-log-controls">
                <EuiFlexGroup gutterSize="m" alignItems="center" wrap>
                    <EuiFlexItem grow={false}>
                        <EuiFormRow label="Log file">
                            <EuiSelect
                                value={logFileName}
                                options={logFileOptions}
                                isDisabled={logFilesLoading}
                                onChange={(event) => (
                                    onLogFileChange(event.target.value)
                                )}
                            />
                        </EuiFormRow>
                    </EuiFlexItem>
                    <EuiFlexItem grow={false}>
                        <EuiFormRow label="Lines per load">
                            <EuiSelect
                                value={String(logPageSize)}
                                options={[50, 100, 200, 500].map((value) => ({
                                    value: String(value),
                                    text: String(value),
                                }))}
                                onChange={(event) => (
                                    onLogPageSizeChange(
                                        Number.parseInt(
                                            event.target.value,
                                            10
                                        )
                                    )
                                )}
                            />
                        </EuiFormRow>
                    </EuiFlexItem>
                    <EuiFlexItem grow={false}>
                        <EuiText size="s">
                            <span>
                                Loaded lines: {logItems.length}
                                {' | '}
                                File size: {formatValue(logFileSize)} bytes
                            </span>
                        </EuiText>
                    </EuiFlexItem>
                </EuiFlexGroup>
            </div>

            <EuiSpacer size="m" />

            <EuiFlexGroup
                justifyContent="spaceBetween"
                alignItems="center"
                gutterSize="m"
                wrap
            >
                <EuiFlexItem grow={false}>
                    <EuiText size="s">
                        <span>
                            Last refresh: {formatDateTime(logLastRefresh)}
                        </span>
                    </EuiText>
                </EuiFlexItem>
                <EuiFlexItem grow={false}>
                    <EuiText size="s">
                        <span>
                            {logHasMore
                                ? 'Scroll up to load older logs'
                                : 'Start of log file reached'}
                        </span>
                    </EuiText>
                </EuiFlexItem>
            </EuiFlexGroup>

            <EuiSpacer size="m" />

            <div
                className="node-management-log-viewer"
                ref={logScrollRef}
                onScroll={handleLogScroll}
            >
                <div className="node-management-log-load-more">
                    {logHasMore ? (
                        <EuiButtonEmpty
                            size="xs"
                            onClick={() => loadOlderNodeLogs()}
                            isLoading={logLoading}
                        >
                            Load older logs
                        </EuiButtonEmpty>
                    ) : (
                        <span>Start of log file</span>
                    )}
                </div>
                {logItems.length ? (
                    logItems.map((item) => (
                        <div
                            className="node-management-log-row"
                            key={item.id || item.offset}
                        >
                            <span className="node-management-log-timestamp">
                                {formatValue(item.timestamp)}
                            </span>
                            <span className="node-management-log-line">
                                {formatValue(item.raw)}
                            </span>
                        </div>
                    ))
                ) : (
                    <div className="node-management-log-empty">
                        {logLoading ? 'Loading logs...' : 'No logs found'}
                    </div>
                )}
            </div>
        </section>
    )
}

export default ApplicationLogs
