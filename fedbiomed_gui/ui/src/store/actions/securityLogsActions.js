import axios from 'axios'
import moment from 'moment'

import { EP_SECURITY_LOGS } from '../../constants'
import {
  SECURITY_LOGS_SET_FACETS,
  SECURITY_LOGS_SET_ITEMS,
  SECURITY_LOGS_SET_LAST_BATCH_SIZE,
  SECURITY_LOGS_SET_LOADING,
} from './actions'

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

const buildRangeParams = ({ startTs, endTs, maxTotal }) => {
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
}

const buildParams = ({
  contains,
  operation,
  status,
  researcherId,
  startTs,
  endTs,
  maxTotal,
  currentPage,
  pageSize,
}) => {
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

  const hasDateFilter = Boolean(startTs || endTs)
  const mt = parseInt(maxTotal, 10)
  if (hasDateFilter && Number.isFinite(mt) && mt > 0) {
    params.max_num_of_logs = mt
  }

  return params
}

export const fetchSecurityLogsFacets = ({ startTs, endTs, maxTotal }) => {
  return async (dispatch) => {
    try {
      const res = await axios.get(EP_SECURITY_LOGS, { params: buildRangeParams({ startTs, endTs, maxTotal }) })
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

      dispatch({
        type: SECURITY_LOGS_SET_FACETS,
        payload: {
          operationOptions: Array.from(ops).sort((a, b) => a.localeCompare(b)),
          statusOptions: Array.from(sts).sort((a, b) => a.localeCompare(b)),
          researcherOptions: Array.from(rids).sort((a, b) => {
            if (a === '__none__') return 1
            if (b === '__none__') return -1
            return a.localeCompare(b)
          }),
        },
      })
    } catch (error) {
      // keep the page functional even if facets fail; errors are shown via global modal
      if (error?.response?.data?.message) {
        dispatch({ type: 'ERROR_MODAL', payload: `Error while loading security log facets: ${error.response.data.message}` })
      } else {
        dispatch({ type: 'ERROR_MODAL', payload: `Unexpected error while loading security log facets: ${String(error)}` })
      }
    }
  }
}

export const fetchSecurityLogs = (args) => {
  return async (dispatch) => {
    dispatch({ type: SECURITY_LOGS_SET_LOADING, payload: true })
    try {
      const res = await axios.get(EP_SECURITY_LOGS, { params: buildParams(args) })
      const payload = res?.data?.result || {}
      const rawItems = payload.items || []
      const newItems = rawItems.map((it) => normalizeItem(it))

      dispatch({ type: SECURITY_LOGS_SET_ITEMS, payload: newItems })
      dispatch({ type: SECURITY_LOGS_SET_LAST_BATCH_SIZE, payload: newItems.length })
    } catch (error) {
      if (error?.response?.data?.message) {
        dispatch({ type: 'ERROR_MODAL', payload: `Error while loading security logs: ${error.response.data.message}` })
      } else {
        dispatch({ type: 'ERROR_MODAL', payload: `Unexpected error while loading security logs: ${String(error)}` })
      }
    } finally {
      dispatch({ type: SECURITY_LOGS_SET_LOADING, payload: false })
    }
  }
}
