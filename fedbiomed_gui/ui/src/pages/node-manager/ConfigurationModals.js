import React from 'react'
import {
    EuiButton,
    EuiCallOut,
    EuiModal,
    EuiModalBody,
    EuiModalFooter,
    EuiModalHeader,
    EuiModalHeaderTitle,
    EuiSpacer,
    EuiText,
} from '@elastic/eui'

import {formatValue, labelFor} from './configFormat'

/**
 * The changes about to be written, field by field, as the last step before a
 * write. Saving from here is the only way a change reaches config.ini.
 */
export const ReviewChangesModal = ({
    changes,
    isRunning,
    writing,
    actionInProgress,
    onCancel,
    onSave,
    onSaveAndRestart,
}) => (
    <EuiModal onClose={onCancel}>
        <EuiModalHeader>
            <EuiModalHeaderTitle>
                Review changes ({changes.length})
            </EuiModalHeaderTitle>
        </EuiModalHeader>
        <EuiModalBody>
            <table className="node-config-review-table">
                <thead>
                    <tr>
                        <th>Field</th>
                        <th>Current</th>
                        <th>New</th>
                    </tr>
                </thead>
                <tbody>
                    {changes.map((change) => (
                        <tr key={`${change.section}.${change.key}`}>
                            <td>
                                <strong>{change.label}</strong>
                                <small>{change.sectionLabel}</small>
                            </td>
                            <td className="node-config-review-current">
                                {formatValue(change.current)}
                            </td>
                            <td className="node-config-review-next">
                                {formatValue(change.next)}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
            {isRunning ? (
                <>
                    <EuiSpacer size="m" />
                    <EuiCallOut
                        color="warning"
                        iconType="refresh"
                        size="s"
                        title={
                            'The node is running. Changes apply after '
                            + 'the node restarts.'
                        }
                    />
                </>
            ) : null}
        </EuiModalBody>
        <EuiModalFooter>
            <EuiButton onClick={onCancel}>
                Back
            </EuiButton>
            <EuiButton
                fill
                iconType="save"
                onClick={onSave}
                isLoading={writing}
                isDisabled={actionInProgress}
            >
                Save
            </EuiButton>
            {isRunning ? (
                <EuiButton
                    color="warning"
                    fill
                    iconType="refresh"
                    onClick={onSaveAndRestart}
                    isLoading={actionInProgress}
                    isDisabled={actionInProgress}
                >
                    Save &amp; Restart
                </EuiButton>
            ) : null}
        </EuiModalFooter>
    </EuiModal>
)

/**
 * What to do about a config.ini that changed under the form. Refusing the write
 * is the default; overwriting it is the user's explicit call.
 */
export const WriteConflictModal = ({
    conflictItems,
    restartAfterWrite,
    actionInProgress,
    onCancel,
    onRefresh,
    onOverwrite,
}) => (
    <EuiModal onClose={onCancel}>
        <EuiModalHeader>
            <EuiModalHeaderTitle>
                Configuration file changed
            </EuiModalHeaderTitle>
        </EuiModalHeader>
        <EuiModalBody>
            <EuiText size="s">
                <p>
                    The configuration file was modified after this page
                    loaded. Review the current file values, refresh the
                    form, or overwrite them.
                </p>
                {conflictItems.map((item) => (
                    <p key={`${item.section}.${item.key}`}>
                        <strong>
                            {labelFor(item.section)} /{' '}
                            {labelFor(item.key)}:
                        </strong>{' '}
                        shown value "{formatValue(item.base)}", file
                        value "{formatValue(item.current)}",
                        requested value "{formatValue(item.requested)}"
                    </p>
                ))}
            </EuiText>
        </EuiModalBody>
        <EuiModalFooter>
            <EuiButton onClick={onCancel}>
                Cancel
            </EuiButton>
            <EuiButton onClick={onRefresh}>
                Refresh latest
            </EuiButton>
            <EuiButton
                color="warning"
                fill
                onClick={onOverwrite}
                isLoading={actionInProgress}
            >
                {restartAfterWrite ? 'Overwrite & Restart' : 'Overwrite'}
            </EuiButton>
        </EuiModalFooter>
    </EuiModal>
)
