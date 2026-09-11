/** Formatting the configuration page and its dialogs both read values through. */

/** A config key as a heading, for a field the schema gives no label. */
export const labelFor = (key) => key
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')

/** A config value as it reads to someone comparing it with config.ini. */
export const formatValue = (value) => {
    if (typeof value === 'boolean') {
        return value ? 'True' : 'False'
    }

    return value === '' || value === null || value === undefined
        ? '(empty)'
        : String(value)
}
