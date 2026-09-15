import React from 'react'
import {
    EuiButton,
    EuiButtonEmpty,
    EuiFlexGroup,
    EuiFlexItem,
    EuiIcon,
    EuiModal,
    EuiModalBody,
    EuiModalFooter,
    EuiModalHeader,
    EuiText,
    EuiTitle,
} from '@elastic/eui'

/**
 * A small window that reports something or asks for a decision: an icon beside
 * a title, an optional body, and one or two buttons. An EUI modal, so it stacks
 * above the EUI window that opened it and takes the keyboard from it.
 *
 * `onClose` runs on Esc, a click outside, and the cancel button; the confirm
 * button runs `onConfirm`, or closes when there is nothing to confirm.
 */
export const Popup = ({
    icon,
    iconColor,
    title,
    onClose,
    cancelText,
    confirmText = 'OK',
    confirmColor = 'primary',
    onConfirm,
    children,
}) => (
    <EuiModal className="popup" onClose={onClose} maxWidth={480}>
        <EuiModalHeader>
            <EuiFlexGroup gutterSize="s" alignItems="center" responsive={false}>
                <EuiFlexItem grow={false}>
                    <EuiIcon type={icon} color={iconColor} size="m" />
                </EuiFlexItem>
                <EuiFlexItem>
                    <EuiTitle size="xs">
                        <h2>{title}</h2>
                    </EuiTitle>
                </EuiFlexItem>
            </EuiFlexGroup>
        </EuiModalHeader>
        {children ? (
            <EuiModalBody>
                <EuiText size="s">{children}</EuiText>
            </EuiModalBody>
        ) : null}
        <EuiModalFooter>
            {cancelText ? (
                <EuiButtonEmpty size="s" onClick={onClose}>
                    {cancelText}
                </EuiButtonEmpty>
            ) : null}
            <EuiButton
                size="s"
                fill
                color={confirmColor}
                onClick={onConfirm || onClose}
            >
                {confirmText}
            </EuiButton>
        </EuiModalFooter>
    </EuiModal>
)

export default Popup
