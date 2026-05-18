import React from 'react'
import axios from 'axios'
import {
  EuiButton,
  EuiButtonEmpty,
  EuiCallOut,
  EuiCode,
  EuiDescriptionList,
  EuiFieldNumber,
  EuiFlexGroup,
  EuiFlexItem,
  EuiFormRow,
  EuiPanel,
  EuiSpacer,
  EuiSwitch,
  EuiText,
  EuiTitle,
} from '@elastic/eui'
import { useDispatch } from 'react-redux'

import {
  EP_NODE_LIFECYCLE_RESTART,
  EP_NODE_LIFECYCLE_START,
  EP_NODE_LIFECYCLE_STATUS,
  EP_NODE_LIFECYCLE_STOP,
} from '../constants'

const DEFAULT_STATUS = {
  node_id: '',
  node_name: '',
  state: 'stopped',
  pid: null,
  action: null,
  reason: null,
  updated_at: null,
  started_at: null,
  stopped_at: null,
  exit_code: null,
  managed_by_current_process: false,
}

const buildNodeArgs = ({ gpu, gpuNum, gpuOnly, debug }) => ({
  gpu: gpu || gpuOnly,
  gpu_num: Math.max(1, parseInt(gpuNum, 10) || 1),
  gpu_only: gpuOnly,
  debug,
})

const formatValue = (value) => {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  return String(value)
}

const NodeLifecycle = () => {
  const dispatch = useDispatch()
  const [status, setStatus] = React.useState(DEFAULT_STATUS)
  const [loading, setLoading] = React.useState(false)
  const [gpu, setGpu] = React.useState(false)
  const [gpuOnly, setGpuOnly] = React.useState(false)
  const [gpuNum, setGpuNum] = React.useState(1)
  const [debug, setDebug] = React.useState(false)

  const isRunning = status.state === 'running'
  const isTransitioning = status.state === 'starting' || status.state === 'stopping'
  const canStart = !loading && !isRunning && !isTransitioning
  const canStop = !loading && isRunning
  const canRestart = !loading && !isTransitioning

  const loadStatus = React.useCallback(async () => {
    setLoading(true)
    try {
      const res = await axios.get(EP_NODE_LIFECYCLE_STATUS)
      setStatus({ ...DEFAULT_STATUS, ...(res?.data?.result || {}) })
    } catch (error) {
      const message = error?.response?.data?.message || String(error)
      dispatch({ type: 'ERROR_MODAL', payload: `Error while loading node status: ${message}` })
    } finally {
      setLoading(false)
    }
  }, [dispatch])

  React.useEffect(() => {
    loadStatus()
  }, [loadStatus])

  const runAction = async (endpoint, args, successMessage) => {
    setLoading(true)
    dispatch({
      type: 'SET_LOADING',
      payload: { status: true, launcher: 'node-lifecycle', text: 'Processing node lifecycle request' },
    })

    try {
      const res = await axios.post(endpoint, args)
      setStatus({ ...DEFAULT_STATUS, ...(res?.data?.result || {}) })
      dispatch({ type: 'SUCCESS_MODAL', payload: successMessage })
    } catch (error) {
      const message = error?.response?.data?.message || String(error)
      dispatch({ type: 'ERROR_MODAL', payload: message })
    } finally {
      setLoading(false)
      dispatch({ type: 'SET_LOADING', payload: { status: false, launcher: 'node-lifecycle' } })
    }
  }

  const nodeArgs = buildNodeArgs({ gpu, gpuNum, gpuOnly, debug })

  const statusItems = [
    { title: 'Node ID', description: formatValue(status.node_id) },
    { title: 'Node name', description: formatValue(status.node_name) },
    { title: 'State', description: <EuiCode>{formatValue(status.state)}</EuiCode> },
    { title: 'PID', description: formatValue(status.pid) },
    { title: 'Last action', description: formatValue(status.action) },
    { title: 'Reason', description: formatValue(status.reason) },
    { title: 'Updated at', description: formatValue(status.updated_at) },
    { title: 'Started at', description: formatValue(status.started_at) },
    { title: 'Stopped at', description: formatValue(status.stopped_at) },
    { title: 'Exit code', description: formatValue(status.exit_code) },
    {
      title: 'Managed here',
      description: formatValue(status.managed_by_current_process),
    },
  ]

  return (
    <>
      <EuiTitle>
        <h2>Node Lifecycle</h2>
      </EuiTitle>
      <EuiSpacer size="m" />

      <EuiFlexGroup gutterSize="l" alignItems="flexStart" wrap>
        <EuiFlexItem grow={2} style={{ minWidth: 360 }}>
          <EuiPanel hasShadow={false} hasBorder>
            <EuiText>
              <h3>Status</h3>
            </EuiText>
            <EuiSpacer size="m" />
            <EuiDescriptionList type="responsiveColumn" listItems={statusItems} />
            {!status.managed_by_current_process && status.state === 'running' ? (
              <>
                <EuiSpacer size="m" />
                <EuiCallOut
                  size="s"
                  color="warning"
                  title="The persisted state is running, but this GUI server does not own a live process handle."
                />
              </>
            ) : null}
          </EuiPanel>
        </EuiFlexItem>

        <EuiFlexItem grow={1} style={{ minWidth: 320 }}>
          <EuiPanel hasShadow={false} hasBorder>
            <EuiText>
              <h3>Startup options</h3>
            </EuiText>
            <EuiSpacer size="m" />

            <EuiFormRow label="GPU">
              <EuiSwitch label="Enable GPU" checked={gpu} onChange={(e) => setGpu(e.target.checked)} />
            </EuiFormRow>

            <EuiFormRow label="GPU only">
              <EuiSwitch
                label="Train only with GPU resources"
                checked={gpuOnly}
                onChange={(e) => {
                  setGpuOnly(e.target.checked)
                  if (e.target.checked) setGpu(true)
                }}
              />
            </EuiFormRow>

            <EuiFormRow label="GPU number">
              <EuiFieldNumber min={1} value={gpuNum} onChange={(e) => setGpuNum(e.target.value)} />
            </EuiFormRow>

            <EuiFormRow label="Debug">
              <EuiSwitch label="Enable debug mode" checked={debug} onChange={(e) => setDebug(e.target.checked)} />
            </EuiFormRow>
          </EuiPanel>
        </EuiFlexItem>
      </EuiFlexGroup>

      <EuiSpacer size="m" />

      <EuiFlexGroup gutterSize="s" wrap>
        <EuiFlexItem grow={false}>
          <EuiButton iconType="play" onClick={() => runAction(EP_NODE_LIFECYCLE_START, nodeArgs, 'Node start request has been processed')} isDisabled={!canStart}>
            Start
          </EuiButton>
        </EuiFlexItem>
        <EuiFlexItem grow={false}>
          <EuiButton iconType="stop" color="danger" onClick={() => runAction(EP_NODE_LIFECYCLE_STOP, {}, 'Node stop request has been processed')} isDisabled={!canStop}>
            Stop
          </EuiButton>
        </EuiFlexItem>
        <EuiFlexItem grow={false}>
          <EuiButton iconType="refresh" onClick={() => runAction(EP_NODE_LIFECYCLE_RESTART, nodeArgs, 'Node restart request has been processed')} isDisabled={!canRestart}>
            Restart
          </EuiButton>
        </EuiFlexItem>
        <EuiFlexItem grow={false}>
          <EuiButtonEmpty iconType="refresh" onClick={loadStatus} isLoading={loading}>
            Refresh
          </EuiButtonEmpty>
        </EuiFlexItem>
      </EuiFlexGroup>
    </>
  )
}

export default NodeLifecycle
