import React from 'react'
import axios from 'axios'
import {
  EuiButton,
  EuiCallOut,
  EuiFlexGroup,
  EuiFlexItem,
  EuiPanel,
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

      <EuiPanel paddingSize="l">
        <EuiText>
          <p>Use these controls to call the draft backend routes for node lifecycle management.</p>
          <p><strong>Current status:</strong> {currentStatus}</p>
        </EuiText>

        <EuiSpacer size="m" />

        <EuiFlexGroup wrap gutterSize="m">
          <EuiFlexItem grow={false}>
            <EuiButton
              fill
              onClick={() => callNodeRoute('start', EP_NODE_START)}
              isLoading={loadingAction === 'start'}
            >
              Start
            </EuiButton>
          </EuiFlexItem>

          <EuiFlexItem grow={false}>
            <EuiButton
              color="warning"
              onClick={() => callNodeRoute('stop', EP_NODE_STOP)}
              isLoading={loadingAction === 'stop'}
            >
              Stop
            </EuiButton>
          </EuiFlexItem>

          <EuiFlexItem grow={false}>
            <EuiButton
              color="primary"
              onClick={() => callNodeRoute('restart', EP_NODE_RESTART)}
              isLoading={loadingAction === 'restart'}
            >
              Restart
            </EuiButton>
          </EuiFlexItem>

          <EuiFlexItem grow={false}>
            <EuiButton
              color="success"
              onClick={loadStatus}
              isLoading={loadingAction === 'status'}
            >
              Status
            </EuiButton>
          </EuiFlexItem>
        </EuiFlexGroup>

        <EuiSpacer size="m" />

        {lastMessage ? (
          <EuiCallOut title={lastMessage} iconType="iInCircle" />
        ) : null}
      </EuiPanel>
    </>
  )
}


export default NodeManagement