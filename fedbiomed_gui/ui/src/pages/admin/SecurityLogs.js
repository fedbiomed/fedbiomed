import React from 'react'
import axios from 'axios'
import {
  EuiBasicTable,
  EuiButton,
  EuiFlexGroup,
  EuiFlexItem,
  EuiFieldNumber,
  EuiFieldSearch,
  EuiFieldText,
  EuiFormRow,
  EuiSelect,
  EuiSpacer,
  EuiSwitch,
  EuiTablePagination,
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
    // internal/computed
    'details',
  ])

  const keys = Object.keys(it).filter((k) => !excluded.has(k))

  // Prefer showing "message" first if present, then remaining keys sorted.
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
    } catch (e) {
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

  const onStartTsChange = React.useCallback((e) => {
    const value = e.target.value
    setStartTs(value)

    // Keep the range valid: if From moves past To, clamp To to From.
    if (!value || !endTs) return
    const start = moment(value)
    const end = moment(endTs)
    if (start.isValid() && end.isValid() && end.isBefore(start)) {
      setEndTs(value)
    }
  }, [endTs])

  const onEndTsChange = React.useCallback((e) => {
    const value = e.target.value

    // Prevent selecting/typing a To earlier than From.
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
  }, [startTs])

  const [showDetails, setShowDetails] = React.useState(true)
  const [showCallerInfo, setShowCallerInfo] = React.useState(false)

  const [operationOptions, setOperationOptions] = React.useState([])
  const [statusOptions, setStatusOptions] = React.useState([])
  const [researcherOptions, setResearcherOptions] = React.useState([])

  const [items, setItems] = React.useState([])
  const [nextSkip, setNextSkip] = React.useState(0)
  const [loading, setLoading] = React.useState(false)

  const [pageIndex, setPageIndex] = React.useState(0)
  const [pageSize, setPageSize] = React.useState(20)
  const [sortField, setSortField] = React.useState('timestamp')
  const [sortDirection, setSortDirection] = React.useState('desc')

  const rangeLabel = React.useMemo(() => {
    const start = startTs ? moment(startTs) : null
    const end = endTs ? moment(endTs) : null
    const startTxt = start && start.isValid() ? start.format('YYYY-MM-DD HH:mm') : ''
    const endTxt = end && end.isValid() ? end.format('YYYY-MM-DD HH:mm') : ''
    return `${startTxt} - ${endTxt}`
  }, [startTs, endTs])

  const buildParams = React.useCallback((skip) => {
    const params = {
      limit: 200,
      skip: skip || 0,
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

    // Apply a default cap for date-range browsing (server also enforces a default).
    const hasDateFilter = Boolean(startTs || endTs)
    const mt = parseInt(maxTotal, 10)
    if (hasDateFilter && Number.isFinite(mt) && mt > 0) {
      params.max_total = mt
    }

    return params
  }, [contains, operation, status, researcherId, startTs, endTs, maxTotal])

  const buildRangeParams = React.useCallback(() => {
    const params = { limit: 2000, skip: 0 }

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
      params.max_total = mt
    }

    return params
  }, [startTs, endTs, maxTotal])

  const loadFacets = React.useCallback(async () => {
    // Fetch a larger unfiltered sample to populate dropdowns.
    // This keeps the implementation simple and avoids a DB.
    const res = await axios.get(EP_SECURITY_LOGS, {
      params: buildRangeParams(),
    })

    const payload = res?.data?.result || {}
    const facetItems = payload.items || []

    const ops = new Set()
    const sts = new Set()
    const rids = new Set()
    facetItems.forEach((it) => {
      if (!it) return
      if (it.operation) ops.add(String(it.operation))
      if (it.status) sts.add(String(it.status))

      // researcher_id may be null in JSON; use a special value for filtering.
      if (it.researcher_id === null || it.researcher_id === undefined || it.researcher_id === '') {
        rids.add('__none__')
      } else {
        rids.add(String(it.researcher_id))
      }
    })

    const opsSorted = Array.from(ops).sort((a, b) => a.localeCompare(b))
    const stsSorted = Array.from(sts).sort((a, b) => a.localeCompare(b))

    const ridsSorted = Array.from(rids).sort((a, b) => {
      if (a === '__none__') return 1
      if (b === '__none__') return -1
      return a.localeCompare(b)
    })

    setOperationOptions(opsSorted)
    setStatusOptions(stsSorted)
    setResearcherOptions(ridsSorted)
  }, [buildRangeParams])

  const loadLogs = React.useCallback(async ({ reset = true } = {}) => {
    setLoading(true)
    try {
      const res = await axios.get(EP_SECURITY_LOGS, { params: buildParams(0) })
      const payload = res?.data?.result || {}
      const rawItems = payload.items || []
      const newItems = rawItems.map((it) => normalizeItem(it))
      setItems(newItems)
      setNextSkip(payload.next_skip || newItems.length)
      if (reset) setPageIndex(0)
    } finally {
      setLoading(false)
    }
  }, [buildParams, normalizeItem])

  const loadMore = React.useCallback(async () => {
    setLoading(true)
    try {
      const res = await axios.get(EP_SECURITY_LOGS, { params: buildParams(nextSkip) })
      const payload = res?.data?.result || {}
      const rawMore = payload.items || []
      const more = rawMore.map((it) => normalizeItem(it))
      setItems((prev) => prev.concat(more))
      setNextSkip(payload.next_skip || (nextSkip + more.length))
    } finally {
      setLoading(false)
    }
  }, [buildParams, nextSkip, normalizeItem])

  // Refresh dropdown options when date range changes
  React.useEffect(() => {
    loadFacets()
  }, [startTs, endTs, loadFacets])

  // Reload when file or filters change
  React.useEffect(() => {
    loadLogs({ reset: true })
  }, [contains, operation, status, researcherId, startTs, endTs, loadLogs])

  const operationSelectOptions = [{ value: '', text: 'All operations' }].concat(
    operationOptions.map((op) => ({ value: op, text: op }))
  )

  const statusSelectOptions = [{ value: '', text: 'All statuses' }].concat(
    statusOptions.map((st) => ({ value: st, text: st }))
  )

  const maxTotalNumber = Number.isFinite(parseInt(maxTotal, 10)) ? parseInt(maxTotal, 10) : null
  const hasMore = maxTotalNumber ? nextSkip < maxTotalNumber : true

  const researcherSelectOptions = [{ value: '', text: 'All researchers' }].concat(
    researcherOptions.map((rid) => ({
      value: rid,
      text: rid === '__none__' ? '(none)' : rid,
    }))
  )

  const cellNoWrap = { whiteSpace: 'nowrap' }
  const cellWrap = { whiteSpace: 'normal', wordBreak: 'break-word' }

  const sortedItems = React.useMemo(() => {
    const arr = items.slice()

    const dir = sortDirection === 'asc' ? 1 : -1
    const getVal = (it) => {
      const v = it?.[sortField]
      if (sortField === 'timestamp') {
        const m = moment(v)
        return m.isValid() ? m.valueOf() : 0
      }
      if (typeof v === 'number') return v
      return String(v ?? '')
    }

    arr.sort((a, b) => {
      const av = getVal(a)
      const bv = getVal(b)
      if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir
      return String(av).localeCompare(String(bv)) * dir
    })

    return arr
  }, [items, sortField, sortDirection])

  const pageCount = Math.max(1, Math.ceil(sortedItems.length / pageSize))
  const safePageIndex = Math.min(pageIndex, pageCount - 1)
  const pagedItems = React.useMemo(() => {
    const start = safePageIndex * pageSize
    return sortedItems.slice(start, start + pageSize)
  }, [sortedItems, safePageIndex, pageSize])

  const onTableChange = React.useCallback(({ sort }) => {
    if (sort && sort.field) {
      setSortField(sort.field)
      setSortDirection(sort.direction)
      setPageIndex(0)
    }
  }, [])

  const columns = [
    {
      field: 'timestamp',
      name: 'Timestamp',
      width: '180px',
      truncateText: false,
      sortable: true,
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
      width: '220px',
      truncateText: false,
      sortable: true,
      render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
    },
    {
      field: 'status',
      name: 'Status',
      width: '140px',
      truncateText: false,
      sortable: true,
      render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
    },
    {
      field: 'researcher_id',
      name: 'Researcher',
      width: '280px',
      truncateText: false,
      sortable: true,
      render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
    },
  ]

  if (showDetails) {
    columns.push({
      field: 'details',
      name: 'Details',
      width: '1100px',
      truncateText: false,
      render: (v) => <div style={cellWrap}>{String(v ?? '')}</div>,
    })
  }

  if (showCallerInfo) {
    columns.push(
      {
        field: 'caller_module',
        name: 'Caller module',
        width: '220px',
        truncateText: false,
        sortable: true,
        render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
      },
      {
        field: 'caller_function',
        name: 'Caller function',
        width: '220px',
        truncateText: false,
        sortable: true,
        render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
      },
      {
        field: 'caller_file',
        name: 'Caller file',
        width: '520px',
        truncateText: false,
        sortable: true,
        render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
      },
      {
        field: 'caller_line',
        name: 'Caller line',
        width: '120px',
        truncateText: false,
        sortable: true,
        render: (v) => <span style={cellNoWrap}>{String(v ?? '')}</span>,
      },
    )
  }

  return (
    <React.Fragment>
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
            <EuiSelect
              options={operationSelectOptions}
              value={operation}
              onChange={(e) => setOperation(e.target.value)}
            />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 180 }}>
          <EuiFormRow label="Status">
            <EuiSelect
              options={statusSelectOptions}
              value={status}
              onChange={(e) => setStatus(e.target.value)}
            />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 240 }}>
          <EuiFormRow label="Researcher ID">
            <EuiSelect
              options={researcherSelectOptions}
              value={researcherId}
              onChange={(e) => setResearcherId(e.target.value)}
            />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 240 }}>
          <EuiFormRow label="From">
            <EuiFieldText
              type="datetime-local"
              value={startTs}
              max={endTs || undefined}
              onChange={onStartTsChange}
            />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 240 }}>
          <EuiFormRow label="To">
            <EuiFieldText
              type="datetime-local"
              value={endTs}
              min={startTs || undefined}
              onChange={onEndTsChange}
            />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 220 }}>
          <EuiFormRow label="Max logs">
            <EuiFieldNumber
              min={1}
              value={maxTotal}
              onChange={(e) => setMaxTotal(e.target.value)}
            />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 220 }}>
          <EuiFormRow label="Caller info">
            <EuiSwitch
              label="Show caller fields"
              checked={showCallerInfo}
              onChange={(e) => setShowCallerInfo(e.target.checked)}
            />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 220 }}>
          <EuiFormRow label="Details">
            <EuiSwitch
              label="Show details column"
              checked={showDetails}
              onChange={(e) => setShowDetails(e.target.checked)}
            />
          </EuiFormRow>
        </EuiFlexItem>
      </EuiFlexGroup>

      <EuiFlexGroup gutterSize="s" justifyContent="flexEnd">
        <EuiFlexItem grow={false}>
          <EuiButton size="s" onClick={() => loadLogs({ reset: true })} isLoading={loading}>
            Refresh
          </EuiButton>
        </EuiFlexItem>
      </EuiFlexGroup>

      <EuiSpacer size="m" />
      <div style={{ overflowX: 'auto' }}>
        <div style={{ minWidth: showCallerInfo ? 3600 : showDetails ? 2200 : 1200 }}>
          <EuiBasicTable
            items={pagedItems}
            columns={columns}
            loading={loading}
            tableLayout="auto"
            sorting={{ sort: { field: sortField, direction: sortDirection } }}
            onChange={onTableChange}
          />
        </div>
      </div>

      <EuiSpacer size="m" />
      <EuiFlexGroup justifyContent="spaceBetween" alignItems="center" gutterSize="m" wrap>
        <EuiFlexItem grow={false}>
          <EuiTablePagination
            activePage={safePageIndex}
            pageCount={pageCount}
            onChangePage={(page) => setPageIndex(page)}
            itemsPerPage={pageSize}
            onChangeItemsPerPage={(size) => {
              setPageSize(size)
              setPageIndex(0)
            }}
            itemsPerPageOptions={[20, 50, 100]}
          />
        </EuiFlexItem>

        <EuiFlexItem grow={false}>
          <EuiButton onClick={loadMore} isLoading={loading} isDisabled={!hasMore}>
            Load more
          </EuiButton>
        </EuiFlexItem>
      </EuiFlexGroup>
    </React.Fragment>
  )
}

export default SecurityLogs
