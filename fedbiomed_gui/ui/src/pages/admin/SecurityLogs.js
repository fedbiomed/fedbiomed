import React from 'react'
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

import { connect } from 'react-redux'

import { fetchSecurityLogs, fetchSecurityLogsFacets } from '../../store/actions/securityLogsActions'

const SecurityLogs = (props) => {
  const [contains, setContains] = React.useState('')
  const [operation, setOperation] = React.useState('')
  const [status, setStatus] = React.useState('')
  const [researcherId, setResearcherId] = React.useState('')

  const [startTs, setStartTs] = React.useState('')
  const [endTs, setEndTs] = React.useState('')
  const [maxTotal, setMaxTotal] = React.useState('2000')

  const [showDetails, setShowDetails] = React.useState(false)
  const [showCallerInfo, setShowCallerInfo] = React.useState(false)

  const [currentPage, setCurrentPage] = React.useState(0)
  const [pageSize, setPageSize] = React.useState(20)

  const {
    operationOptions,
    statusOptions,
    researcherOptions,
    items,
    loading,
    lastBatchSize,
    fetchSecurityLogs,
    fetchSecurityLogsFacets,
  } = props

  const loadFacets = React.useCallback(() => {
    fetchSecurityLogsFacets({ startTs, endTs, maxTotal })
  }, [fetchSecurityLogsFacets, startTs, endTs, maxTotal])

  const loadLogs = React.useCallback(() => {
    fetchSecurityLogs({
      contains,
      operation,
      status,
      researcherId,
      startTs,
      endTs,
      maxTotal,
      currentPage,
      pageSize,
    })
  }, [fetchSecurityLogs, contains, operation, status, researcherId, startTs, endTs, maxTotal, currentPage, pageSize])

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

const mapStateToProps = (state) => {
  return {
    operationOptions: state.security_logs.operationOptions,
    statusOptions: state.security_logs.statusOptions,
    researcherOptions: state.security_logs.researcherOptions,
    items: state.security_logs.items,
    loading: state.security_logs.loading,
    lastBatchSize: state.security_logs.lastBatchSize,
  }
}

const mapDispatchToProps = (dispatch) => {
  return {
    fetchSecurityLogs: (args) => dispatch(fetchSecurityLogs(args)),
    fetchSecurityLogsFacets: (args) => dispatch(fetchSecurityLogsFacets(args)),
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(SecurityLogs)