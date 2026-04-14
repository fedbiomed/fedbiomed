import React from 'react'
import axios from 'axios'
import {
  EuiButton,
  EuiCallOut,
  EuiFlexGroup,
  EuiFlexItem,
  EuiHealth,
  EuiSpacer,
  EuiText,
  EuiTitle,
} from '@elastic/eui'
import {
  EP_NODE_RESTART,
  EP_NODE_START,
  EP_NODE_STATUS,
  EP_NODE_STOP,
} from '../constants'


const NodeManagement = () => {
  const [currentStatus, setCurrentStatus] = React.useState('unknown')
  const [lastMessage, setLastMessage] = React.useState('')
  const [loadingAction, setLoadingAction] = React.useState('')

  const statusColor = React.useMemo(() => {
    if (currentStatus === 'running') {
      return 'success'
    }

    if (currentStatus === 'stopped') {
      return 'danger'
    }

    return 'subdued'
  }, [currentStatus])

  const canStart = currentStatus !== 'running' && loadingAction === ''
  const canStop = currentStatus === 'running' && loadingAction === ''
  const canRestart = currentStatus === 'running' && loadingAction === ''
  const canRefreshStatus = loadingAction === ''

  const loadStatus = React.useCallback(async () => {
    setLoadingAction('status')
    try {
      const res = await axios.get(EP_NODE_STATUS)
      setCurrentStatus(res?.data?.result?.status || 'unknown')
      setLastMessage(res?.data?.message || 'Node status retrieved')
    } catch (error) {
      setLastMessage(error?.response?.data?.message || `Unable to get node status: ${String(error)}`)
    } finally {
      setLoadingAction('')
    }
  }, [])

  React.useEffect(() => {
    loadStatus()
  }, [loadStatus])

  const callNodeRoute = async (action, endpoint) => {
    setLoadingAction(action)
    try {
      const res = await axios.post(endpoint)
      setLastMessage(res?.data?.message || `Node ${action} command sent`)
      await loadStatus()
    } catch (error) {
      setLastMessage(error?.response?.data?.message || `Unable to ${action} node: ${String(error)}`)
      setLoadingAction('')
    }
  }

  return (
    <>
      <EuiTitle>
        <h2>Node Management</h2>
      </EuiTitle>
      <EuiSpacer size="m" />

      <EuiText>
        <p>Use these controls to call the draft backend routes for node lifecycle management.</p>
        <p>
          <strong>Current status:</strong>{' '}
          <EuiHealth color={statusColor}>{currentStatus}</EuiHealth>
        </p>
      </EuiText>

      <EuiSpacer size="m" />

      <EuiFlexGroup wrap gutterSize="m">
        <EuiFlexItem grow={false}>
          <EuiButton
            fill
            color="primary"
            onClick={() => callNodeRoute('start', EP_NODE_START)}
            isLoading={loadingAction === 'start'}
            isDisabled={!canStart}
          >
            Start
          </EuiButton>
        </EuiFlexItem>

        <EuiFlexItem grow={false}>
          <EuiButton
            fill
            color="primary"
            onClick={() => callNodeRoute('stop', EP_NODE_STOP)}
            isLoading={loadingAction === 'stop'}
            isDisabled={!canStop}
          >
            Stop
          </EuiButton>
        </EuiFlexItem>

        <EuiFlexItem grow={false}>
          <EuiButton
            fill
            color="primary"
            onClick={() => callNodeRoute('restart', EP_NODE_RESTART)}
            isLoading={loadingAction === 'restart'}
            isDisabled={!canRestart}
          >
            Restart
          </EuiButton>
        </EuiFlexItem>

        <EuiFlexItem grow={false}>
          <EuiButton
            fill
            color="primary"
            onClick={loadStatus}
            isLoading={loadingAction === 'status'}
            isDisabled={!canRefreshStatus}
          >
            Status
          </EuiButton>
        </EuiFlexItem>
      </EuiFlexGroup>

      <EuiSpacer size="m" />

      {lastMessage ? (
        <EuiCallOut title={lastMessage} iconType="iInCircle" />
      ) : null}
    </>
  )
}


export default NodeManagement