import React from 'react'
import axios from 'axios'
import {
  EuiButton,
  EuiFlexGroup,
  EuiFlexItem,
  EuiFieldSearch,
  EuiFormRow,
  EuiInMemoryTable,
  EuiSelect,
  EuiSpacer,
  EuiSwitch,
  EuiText,
} from '@elastic/eui'
import moment from 'moment'

import { EP_SECURITY_LOG_FILES, EP_SECURITY_LOGS } from '../../constants'


const SecurityLogs = () => {
  const [files, setFiles] = React.useState([])
  const [selectedFile, setSelectedFile] = React.useState('')

  const [contains, setContains] = React.useState('')
  const [operation, setOperation] = React.useState('')
  const [status, setStatus] = React.useState('')
  const [researcherId, setResearcherId] = React.useState('')

  const [showCallerInfo, setShowCallerInfo] = React.useState(false)

  const [operationOptions, setOperationOptions] = React.useState([])
  const [statusOptions, setStatusOptions] = React.useState([])
  const [researcherOptions, setResearcherOptions] = React.useState([])

  const [items, setItems] = React.useState([])
  const [nextSkip, setNextSkip] = React.useState(0)
  const [loading, setLoading] = React.useState(false)

  const loadFiles = React.useCallback(async () => {
    const res = await axios.get(EP_SECURITY_LOG_FILES)
    const list = res?.data?.result || []
    setFiles(list)

    // Default selection to newest file if none selected
    if (!selectedFile && list.length > 0) {
      setSelectedFile(list[0].name)
    }
  }, [selectedFile])

  const buildParams = React.useCallback((skip) => {
    const params = {
      limit: 200,
      skip: skip || 0,
    }

    if (selectedFile) params.file = selectedFile
    if (contains) params.contains = contains
    if (operation) params.operation = operation
    if (status) params.status = status
    if (researcherId) params.researcher_id = researcherId

    return params
  }, [selectedFile, contains, operation, status, researcherId])

  const loadFacets = React.useCallback(async () => {
    if (!selectedFile) return

    // Fetch a larger unfiltered sample to populate dropdowns.
    // This keeps the implementation simple and avoids a DB.
    const res = await axios.get(EP_SECURITY_LOGS, {
      params: {
        file: selectedFile,
        limit: 2000,
        skip: 0,
      },
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
  }, [selectedFile])

  const loadLogs = React.useCallback(async ({ reset = true } = {}) => {
    setLoading(true)
    try {
      const res = await axios.get(EP_SECURITY_LOGS, { params: buildParams(0) })
      const payload = res?.data?.result || {}
      const newItems = payload.items || []
      setItems(newItems)
      setNextSkip(payload.next_skip || newItems.length)
    } finally {
      setLoading(false)
    }
  }, [buildParams])

  const loadMore = React.useCallback(async () => {
    setLoading(true)
    try {
      const res = await axios.get(EP_SECURITY_LOGS, { params: buildParams(nextSkip) })
      const payload = res?.data?.result || {}
      const more = payload.items || []
      setItems((prev) => prev.concat(more))
      setNextSkip(payload.next_skip || (nextSkip + more.length))
    } finally {
      setLoading(false)
    }
  }, [buildParams, nextSkip])

  React.useEffect(() => {
    loadFiles()
  }, [loadFiles])

  // Refresh dropdown options when the file changes
  React.useEffect(() => {
    loadFacets()
  }, [selectedFile, loadFacets])

  // Reload when file or filters change
  React.useEffect(() => {
    if (!selectedFile) return
    loadLogs({ reset: true })
  }, [selectedFile, contains, operation, status, researcherId, loadLogs])

  const fileOptions = files.map((f) => ({
    value: f.name,
    text: `${f.name} (${Math.round((f.size || 0) / 1024)} KB)`,
  }))

  const operationSelectOptions = [{ value: '', text: 'All operations' }].concat(
    operationOptions.map((op) => ({ value: op, text: op }))
  )

  const statusSelectOptions = [{ value: '', text: 'All statuses' }].concat(
    statusOptions.map((st) => ({ value: st, text: st }))
  )

  const researcherSelectOptions = [{ value: '', text: 'All researchers' }].concat(
    researcherOptions.map((rid) => ({
      value: rid,
      text: rid === '__none__' ? '(none)' : rid,
    }))
  )

  const columns = [
    {
      field: 'timestamp',
      name: 'Timestamp',
      truncateText: true,
      render: (ts) => {
        if (!ts) return ''
        const m = moment(ts)
        return m.isValid() ? m.format('YYYY-MM-DD HH:mm:ss') : String(ts)
      },
    },
    { field: 'operation', name: 'Operation', truncateText: true },
    { field: 'status', name: 'Status', truncateText: true },
    { field: 'researcher_id', name: 'Researcher', truncateText: true },
    { field: 'message', name: 'Message', truncateText: true },
  ]

  if (showCallerInfo) {
    columns.push(
      { field: 'caller_module', name: 'Caller module', truncateText: true },
      { field: 'caller_function', name: 'Caller function', truncateText: true },
      { field: 'caller_file', name: 'Caller file', truncateText: true },
      { field: 'caller_line', name: 'Caller line', truncateText: true },
    )
  }

  return (
    <React.Fragment>
      <EuiText>
        <h2>Security Logs</h2>
      </EuiText>
      <EuiSpacer size="m" />

      <EuiFlexGroup gutterSize="m" wrap>
        <EuiFlexItem grow={false} style={{ minWidth: 320 }}>
          <EuiFormRow label="Log file">
            <EuiSelect
              options={fileOptions}
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
            />
          </EuiFormRow>
        </EuiFlexItem>

        <EuiFlexItem grow={false} style={{ minWidth: 260 }}>
          <EuiFormRow label="Contains">
            <EuiFieldSearch
              value={contains}
              onChange={(e) => setContains(e.target.value)}
              placeholder="Search in message"
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

        <EuiFlexItem grow={false} style={{ minWidth: 220 }}>
          <EuiFormRow label="Caller info">
            <EuiSwitch
              label="Show caller fields"
              checked={showCallerInfo}
              onChange={(e) => setShowCallerInfo(e.target.checked)}
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
      <EuiInMemoryTable
        items={items}
        columns={columns}
        loading={loading}
        pagination={{ initialPageSize: 20, pageSizeOptions: [20, 50, 100] }}
        sorting={true}
      />

      <EuiSpacer size="m" />
      <EuiFlexGroup justifyContent="center">
        <EuiFlexItem grow={false}>
          <EuiButton onClick={loadMore} isLoading={loading}>
            Load more
          </EuiButton>
        </EuiFlexItem>
      </EuiFlexGroup>
    </React.Fragment>
  )
}

export default SecurityLogs
