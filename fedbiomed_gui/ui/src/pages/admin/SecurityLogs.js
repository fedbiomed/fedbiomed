import React from 'react'
import axios from 'axios'
import {
  EuiBasicTable,
  EuiButton,
  EuiButtonEmpty,
  EuiFlexGroup,
  EuiFlexItem,
  EuiFieldNumber,
  EuiFieldSearch,
  EuiFieldText,
  EuiFormRow,
  EuiSelect,
  EuiSpacer,
  EuiSwitch,
  EuiText,
} from '@elastic/eui'
import moment from 'moment'

import { EP_SECURITY_LOGS } from '../../constants'

function normalizeItem(it) {
  if (!it || typeof it !== 'object') return { details: '' }

  const excluded = new Set([
    'timestamp',
    'operation',
    'status',
    'researcher_id',
    'caller_module',
    'caller_function',
    'caller_file',
    'caller_line',
    'details',
  ])

  const keys = Object.keys(it).filter((k) => !excluded.has(k))

  const messageIdx = keys.indexOf('message')
  if (messageIdx >= 0) {
    keys.splice(messageIdx, 1)
    keys.unshift('message')
  }

  const otherKeys = keys.filter((k) => k !== 'message').sort((a, b) => a.localeCompare(b))
  const orderedKeys = keys[0] === 'message' ? ['message'].concat(otherKeys) : otherKeys

  const parts = orderedKeys.map((k) => {
    const v = it[k]
    if (v === null || v === undefined) return `${k}=`
    if (typeof v === 'string') return `${k}=${v}`
    try {
      return `${k}=${JSON.stringify(v)}`
    } catch {
      return `${k}=${String(v)}`
    }
  })

  return { ...it, details: parts.join(' | ') }
}

const SecurityLogs = () => {
  const [contains, setContains] = React.useState('')
  const [operation, setOperation] = React.useState('')
  const [status, setStatus] = React.useState('')
  const [researcherId, setResearcherId] = React.useState('')

  const [startTs, setStartTs] = React.useState('')
  const [endTs, setEndTs] = React.useState('')
  const [maxTotal, setMaxTotal] = React.useState('2000')

  const [showDetails, setShowDetails] = React.useState(false)
  const [showCallerInfo, setShowCallerInfo] = React.useState(false)

  const [operationOptions, setOperationOptions] = React.useState([])
  const [statusOptions, setStatusOptions] = React.useState([])
  const [researcherOptions, setResearcherOptions] = React.useState([])

  // Server-side paging state
  const [items, setItems] = React.useState([])
  const [loading, setLoading] = React.useState(false)
  const [currentPage, setCurrentPage] = React.useState(0)
  const [pageSize, setPageSize] = React.useState(20)
  const [lastBatchSize, setLastBatchSize] = React.useState(0)

  const onStartTsChange = React.useCallback(
    (e) => {
      const value = e.target.value
      setStartTs(value)

      if (!value || !endTs) return
      const start = moment(value)
      const end = moment(endTs)
      if (start.isValid() && end.isValid() && end.isBefore(start)) {
        setEndTs(value)
      }
    },
    [endTs],
  )

  const onEndTsChange = React.useCallback(
    (e) => {
      const value = e.target.value
      if (!value) {
        setEndTs(value)
        return
      }
      if (startTs) {
        const start = moment(startTs)
        const end = moment(value)
        if (start.isValid() && end.isValid() && end.isBefore(start)) {
          setEndTs(startTs)
          return
        }
      }
      setEndTs(value)
    },
    [startTs],
  )

  const rangeLabel = React.useMemo(() => {
    const start = startTs ? moment(startTs) : null
    const end = endTs ? moment(endTs) : null
    const startTxt = start && start.isValid() ? start.format('YYYY-MM-DD HH:mm') : ''
    const endTxt = end && end.isValid() ? end.format('YYYY-MM-DD HH:mm') : ''
    return `${startTxt} - ${endTxt}`
  }, [startTs, endTs])

  const buildParams = React.useCallback(() => {
    const params = {
      page_size: pageSize,
      current_page: currentPage,
    }

    if (contains) params.contains = contains
    if (operation) params.operation = operation
    if (status) params.status = status
    if (researcherId) params.researcher_id = researcherId

    if (startTs) {
      const m = moment(startTs)
      if (m.isValid()) params.start_ts = m.toISOString()
    }
    if (endTs) {
      const m = moment(endTs)
      if (m.isValid()) params.end_ts = m.toISOString()
    }

    // Apply a cap only when date filtering is used (optional)
    const hasDateFilter = Boolean(startTs || endTs)
    const mt = parseInt(maxTotal, 10)
    if (hasDateFilter && Number.isFinite(mt) && mt > 0) {
      params.max_num_of_logs = mt
    }

    return params
  }, [pageSize, currentPage, contains, operation, status, researcherId, startTs, endTs, maxTotal])

  const buildRangeParams = React.useCallback(() => {
    const params = { page_size: 2000, current_page: 0 }

    if (startTs) {
      const m = moment(startTs)
      if (m.isValid()) params.start_ts = m.toISOString()
    }
    if (endTs) {
      const m = moment(endTs)
      if (m.isValid()) params.end_ts = m.toISOString()
    }

    const hasDateFilter = Boolean(startTs || endTs)
    const mt = parseInt(maxTotal, 10)
    if (hasDateFilter && Number.isFinite(mt) && mt > 0) {
      params.max_num_of_logs = mt
    }

    return params
  }, [startTs, endTs, maxTotal])

  const loadFacets = React.useCallback(async () => {
    const res = await axios.get(EP_SECURITY_LOGS, { params: buildRangeParams() })
    const payload = res?.data?.result || {}
    const facetItems = payload.items || []

    const ops = new Set()
    const sts = new Set()
    const rids = new Set()

    facetItems.forEach((it) => {
      if (!it) return
      if (it.operation) ops.add(String(it.operation))
      if (it.status) sts.add(String(it.status))

      if (it.researcher_id === null || it.researcher_id === undefined || it.researcher_id === '') {
        rids.add('__none__')
      } else {
        rids.add(String(it.researcher_id))
      }
    })

    setOperationOptions(Array.from(ops).sort((a, b) => a.localeCompare(b)))
    setStatusOptions(Array.from(sts).sort((a, b) => a.localeCompare(b)))
    setResearcherOptions(
      Array.from(rids).sort((a, b) => {
        if (a === '__none__') return 1
        if (b === '__none__') return -1
        return a.localeCompare(b)
      }),
    )
  }, [buildRangeParams])

  const loadLogs = React.useCallback(async () => {
    setLoading(true)
    try {
      const res = await axios.get(EP_SECURITY_LOGS, { params: buildParams() })
      const payload = res?.data?.result || {}
      const rawItems = payload.items || []
      const newItems = rawItems.map((it) => normalizeItem(it))
      setItems(newItems)
      setLastBatchSize(newItems.length)
    } finally {
      setLoading(false)
    }
  }, [buildParams])

  // Refresh dropdown options when date range changes
  React.useEffect(() => {
    loadFacets()
  }, [startTs, endTs, loadFacets])

  // When filters change, reset to first page
  React.useEffect(() => {
    setCurrentPage(0)
  }, [contains, operation, status, researcherId, startTs, endTs])

  // Load logs when current page / page size / filters change
  React.useEffect(() => {
    loadLogs()
  }, [loadLogs])

  const operationSelectOptions = [{ value: '', text: 'All operations' }].concat(
    operationOptions.map((op) => ({ value: op, text: op })),
  )
  const statusSelectOptions = [{ value: '', text: 'All statuses' }].concat(
    statusOptions.map((st) => ({ value: st, text: st })),
  )
  const researcherSelectOptions = [{ value: '', text: 'All researchers' }].concat(
    researcherOptions.map((rid) => ({
      value: rid,
      text: rid === '__none__' ? '(none)' : rid,
    })),
  )

  const cellNoWrap = { whiteSpace: 'nowrap' }
  const cellWrap = { whiteSpace: 'normal', wordBreak: 'break-word' }

  const columns = [
    {
      field: 'timestamp',
      name: 'Timestamp',
      truncateText: false,
      render: (ts) => {
        if (!ts) return ''
        const m = moment(ts)
        const txt = m.isValid() ? m.format('YYYY-MM-DD HH:mm:ss') : String(ts)
        return <span style={cellNoWrap}>{txt}</span>
      },
    },
    {
      field: 'operation',
      name: 'Operation',
      truncateText: false,
      render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
    },
    {
      field: 'status',
      name: 'Status',
      truncateText: false,
      render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
    },
    {
      field: 'researcher_id',
      name: 'Researcher',
      truncateText: false,
      render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
    },
  ]

  if (showDetails) {
    columns.push({
      field: 'details',
      name: 'Details',
      truncateText: false,
      render: (v) => <div style={cellWrap}>{String(v ?? '')}</div>,
    })
  }

  if (showCallerInfo) {
    columns.push(
      {
        field: 'caller_module',
        name: 'Caller module',
        truncateText: false,
        render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
      },
      {
        field: 'caller_function',
        name: 'Caller function',
        truncateText: false,
        render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
      },
      {
        field: 'caller_file',
        name: 'Caller file',
        truncateText: false,
        render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
      },
      {
        field: 'caller_line',
        name: 'Caller line',
        truncateText: false,
        render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
      },
    )
  }

  const canGoPrev = currentPage > 0
  const canGoNext = lastBatchSize === pageSize

  return (
    <>
      <EuiText>
        <h2>Security Logs</h2>
        <p>{rangeLabel}</p>
      </EuiText>
      <EuiSpacer size="m" />

      <EuiFlexGroup gutterSize="m" wrap>
        <EuiFlexItem grow={false} style={{ minWidth: 260 }}>
          <EuiFormRow label="Contains">
            <EuiFieldSearch
              value={contains}
              onChange={(e) => setContains(e.target.value)}
              placeholder="Search in entry"
            />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 220 }}>
          <EuiFormRow label="Operation">
            <EuiSelect options={operationSelectOptions} value={operation} onChange={(e) => setOperation(e.target.value)} />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 180 }}>
          <EuiFormRow label="Status">
            <EuiSelect options={statusSelectOptions} value={status} onChange={(e) => setStatus(e.target.value)} />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 240 }}>
          <EuiFormRow label="Researcher ID">
            <EuiSelect options={researcherSelectOptions} value={researcherId} onChange={(e) => setResearcherId(e.target.value)} />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 240 }}>
          <EuiFormRow label="From">
            <EuiFieldText type="datetime-local" value={startTs} max={endTs || undefined} onChange={onStartTsChange} />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 240 }}>
          <EuiFormRow label="To">
            <EuiFieldText type="datetime-local" value={endTs} min={startTs || undefined} onChange={onEndTsChange} />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 220 }}>
          <EuiFormRow label="Max logs">
            <EuiFieldNumber min={1} value={maxTotal} onChange={(e) => setMaxTotal(e.target.value)} />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 220 }}>
          <EuiFormRow label="Rows per page">
            <EuiSelect
              value={String(pageSize)}
              options={[20, 50, 100, 200].map((n) => ({ value: String(n), text: String(n) }))}
              onChange={(e) => {
                setPageSize(parseInt(e.target.value, 10))
                setCurrentPage(0)
              }}
            />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 220 }}>
          <EuiFormRow label="Caller info">
            <EuiSwitch label="Show caller fields" checked={showCallerInfo} onChange={(e) => setShowCallerInfo(e.target.checked)} />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 220 }}>
          <EuiFormRow label="Details">
            <EuiSwitch label="Show details column" checked={showDetails} onChange={(e) => setShowDetails(e.target.checked)} />
          </EuiFormRow>
        </EuiFlexItem>
      </EuiFlexGroup>

      <EuiSpacer size="m" />

      <EuiFlexGroup justifyContent="spaceBetween" alignItems="center" gutterSize="m" wrap>
        <EuiFlexItem grow={false}>
          <EuiButton size="s" onClick={() => loadLogs()} isLoading={loading}>
            Refresh
          </EuiButton>
        </EuiFlexItem>

        <EuiFlexItem grow={false}>
          <EuiFlexGroup gutterSize="s" alignItems="center">
            <EuiFlexItem grow={false}>
              <EuiButtonEmpty size="s" onClick={() => setCurrentPage((p) => Math.max(0, p - 1))} isDisabled={!canGoPrev || loading}>
                Prev
              </EuiButtonEmpty>
            </EuiFlexItem>
            <EuiFlexItem grow={false}>
              <EuiText size="s">
                <span>Page {currentPage + 1}</span>
              </EuiText>
            </EuiFlexItem>
            <EuiFlexItem grow={false}>
              <EuiButtonEmpty size="s" onClick={() => setCurrentPage((p) => p + 1)} isDisabled={!canGoNext || loading}>
                Next
              </EuiButtonEmpty>
            </EuiFlexItem>
          </EuiFlexGroup>
        </EuiFlexItem>
      </EuiFlexGroup>

      <EuiSpacer size="m" />
      <div style={{ overflowX: 'auto' }}>
        <div style={{ minWidth: showCallerInfo ? 3600 : showDetails ? 2200 : 1200 }}>
          <EuiBasicTable items={items} columns={columns} loading={loading} tableLayout="auto" />
        </div>
      </div>
    </>
  )
}

export default SecurityLogs