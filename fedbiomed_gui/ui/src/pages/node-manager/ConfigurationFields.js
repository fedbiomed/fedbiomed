import React from 'react'
import {
    EuiCallOut,
    EuiFieldNumber,
    EuiFieldText,
    EuiSelect,
    EuiSpacer,
} from '@elastic/eui'

import {formatValue, labelFor} from './configFormat'

// What a setting actually does, where the key alone does not say it
const fieldDescriptions = {
    'security.training_plan_approval':
        'Require manual approval before a training plan can run.',
    'security.allow_default_training_plans':
        'Permit built-in training plans without extra registration.',
    'security.secure_aggregation':
        'Enable secure aggregation for compatible experiments.',
    'security.force_secure_aggregation':
        'Reject jobs that do not use secure aggregation.',
    'security.secagg_insecure_validation':
        'Allow insecure validation mode for secure aggregation setup.',
    'security.allow_preproc':
        'Allow preprocessing steps before model training starts.',
    'security.allow_federated_analytics':
        'Permit analytics queries that do not train a model.',
}

const ChangeHint = ({savedValue}) => (
    <span className="node-config-change-hint">
        was {formatValue(savedValue)}
    </span>
)

/** The control a field is edited through, whatever its type. */
const FieldControl = ({section, fieldKey, field, value, append, onChange}) => {
    const disabled = !field.editable
    const handleChange = (event) => onChange(section, fieldKey, event.target.value)

    if (field.type === 'boolean') {
        const checked = Boolean(value)

        return (
            <div className="node-config-segmented">
                <button
                    type="button"
                    className="true"
                    aria-pressed={checked}
                    disabled={disabled}
                    onClick={() => onChange(section, fieldKey, true)}
                >
                    True
                </button>
                <button
                    type="button"
                    className="false"
                    aria-pressed={!checked}
                    disabled={disabled}
                    onClick={() => onChange(section, fieldKey, false)}
                >
                    False
                </button>
            </div>
        )
    }

    if (field.type === 'integer') {
        return (
            <EuiFieldNumber
                className="node-config-input"
                disabled={disabled}
                min={field.min ?? 0}
                value={value}
                onChange={handleChange}
            />
        )
    }

    if (field.type === 'enum') {
        return (
            <EuiSelect
                className="node-config-input"
                disabled={disabled}
                value={value}
                options={(field.options || []).map((option) => ({
                    value: option,
                    text: option,
                }))}
                onChange={handleChange}
            />
        )
    }

    return (
        <EuiFieldText
            className="node-config-input"
            disabled={disabled}
            value={value}
            title={disabled ? String(value) : undefined}
            onChange={handleChange}
            append={append}
        />
    )
}

/** One setting: what it is on the left, what it is set to on the right. */
const ConfigField = ({
    section,
    fieldKey,
    field,
    value,
    savedValue,
    changed,
    append,
    onChange,
}) => {
    const description = fieldDescriptions[`${section}.${fieldKey}`]

    return (
        <div className={`node-config-row ${changed ? 'changed' : ''}`}>
            <div className="node-config-row-label">
                <div className="node-config-row-name">
                    {field.label || labelFor(fieldKey)}
                </div>
                {description ? (
                    <div className="node-config-row-help">{description}</div>
                ) : null}
                {changed ? <ChangeHint savedValue={savedValue} /> : null}
            </div>
            <div className="node-config-row-control">
                <FieldControl
                    section={section}
                    fieldKey={fieldKey}
                    field={field}
                    value={value}
                    append={append}
                    onChange={onChange}
                />
            </div>
        </div>
    )
}

/** The fields of one config.ini section, in the order the file lists them. */
const ConfigSection = ({
    section,
    sectionInfo,
    showHeading,
    actions,
    draftValues,
    savedValues,
    changedKeys,
    onChange,
    appendFor,
}) => {
    const fields = sectionInfo.fields || {}
    const fieldKeys = Object.keys(fields)

    if (!fieldKeys.length) {
        return null
    }

    return (
        <div className="node-config-subsection">
            {showHeading || actions ? (
                <div className="node-config-subsection-header">
                    {showHeading ? (
                        <h3 className="node-config-subsection-heading">
                            {sectionInfo.label || labelFor(section)}
                        </h3>
                    ) : null}
                    {actions}
                </div>
            ) : null}

            <div className="node-config-rows">
                {fieldKeys.map((key) => (
                    <ConfigField
                        key={key}
                        section={section}
                        fieldKey={key}
                        field={fields[key]}
                        value={draftValues[section]?.[key] ?? ''}
                        savedValue={savedValues[section]?.[key]}
                        changed={changedKeys.has(`${section}.${key}`)}
                        append={appendFor(section, key)}
                        onChange={onChange}
                    />
                ))}
            </div>
        </div>
    )
}

/**
 * One concern of the configuration, however many config.ini sections it spans.
 * The tab above names the group, so each section speaks for itself and carries
 * whatever it can be acted on with.
 */
export const ConfigGroup = ({
    group,
    sections,
    sectionActions = {},
    ...fieldProps
}) => (
    <section className="node-config-section">
        {group.sections.map((section) => (
            <ConfigSection
                key={section}
                section={section}
                sectionInfo={sections[section] || {}}
                showHeading={group.sections.length > 1}
                actions={sectionActions[section]}
                {...fieldProps}
            />
        ))}
    </section>
)

/** Callouts driven by a list, so the page shows only what currently applies. */
export const Notices = ({notices}) => (
    <>
        {notices.filter((notice) => notice.title).map((notice) => (
            <React.Fragment key={notice.title}>
                <EuiSpacer size="m" />
                <EuiCallOut
                    color={notice.color}
                    iconType={notice.icon}
                    title={notice.title}
                >
                    {notice.body ? <p>{notice.body}</p> : null}
                </EuiCallOut>
            </React.Fragment>
        ))}
    </>
)
