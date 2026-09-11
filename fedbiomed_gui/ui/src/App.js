import React from 'react'
import './App.css';
import '@elastic/eui/dist/eui_theme_light.css';
import { EuiProvider } from '@elastic/eui';
import Popup from "./components/common/Popup";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useOutletContext,
} from "react-router-dom";

import Home from './pages/Home'
import Repository from './pages/repository';
import Datasets from './pages/datasets';
import AddDataset from "./pages/datasets/AddDataset"
import DatasetPreview from './pages/datasets/DatasetPreview';
import {connect, useDispatch} from 'react-redux'
import CommonStandards from "./pages/datasets/CommonStandards";
import MedicalFolderDataset from "./pages/datasets/medical.folder.dataset";
import TrainingPlans from "./pages/training-plan/TrainingPlans";
import SingleModel from "./pages/training-plan/SingleTrainingPlan";
import Login from "./pages/authentication/Login";
import Register from "./pages/authentication/Register";
import {LoginProtected, AdminProtected} from "./components/layout/ProtectedRoutes";
import PasswordChange from "./pages/authentication/PasswordChange";
import UserInfo from "./pages/authentication/UserInfo";
import UserManagement from "./pages/admin/UserManagement";
import AccountRequestManagement from "./pages/admin/AccountRequestManagement";
import SecurityLogs from "./pages/admin/SecurityLogs";
import UserAccount from './pages/authentication/UserAccount';
import NodeManagement, {NodeManagementEnabled} from './pages/node-manager/NodeManagement';

// Reads the flag passed down via <Outlet context={...}/> from LoginProtected,
// which only fetches it once the user is authenticated.
const NodeManagementRoute = () => {
  const {nodeManagementEnabled} = useOutletContext();
  return nodeManagementEnabled ? <NodeManagementEnabled/> : <NodeManagement/>;
}

function App(props) {

  const dispatch = useDispatch();
  const onResultModalClose = () => dispatch({type:'RESET_GLOBAL_MODAL'});


  return (
    <EuiProvider colorMode="light">
      <div className="App" >
        <Router>
              <Routes>
                <Route path="/login/" element={<Login/>} />
                <Route path="/register/" element={<Register/>} />
                <Route path="/" element ={<LoginProtected/>} >
                  <Route path="/" element={<Home/>} />
                  <Route path="/user-account/" element={<UserAccount/>}>
                      <Route index element={<UserInfo/>} />
                      <Route path={"info"} element={<UserInfo/>} />
                      <Route path={"change-password"} element={<PasswordChange/>} />
                      <Route path={"user-management"} element={<AdminProtected redirect_to={'/user-account'}><UserManagement/></AdminProtected>}/>
                      <Route path={"account-requests"} element={<AdminProtected redirect_to={'/user-account'}><AccountRequestManagement/></AdminProtected>}/>
                      <Route path={"security-logs"} element={<AdminProtected redirect_to={'/user-account'}><SecurityLogs/></AdminProtected>}/>
                  </Route>
                  <Route path="/repository/" element={<Repository/>} />
                  <Route path="/node-management/" element={<NodeManagementRoute/>} />
                  <Route path="/training-plans/" element={<TrainingPlans/>} />
                  <Route path="/training-plans/preview/:training_plan_id" element={<SingleModel />} />
                  <Route path="/datasets/" element={<Datasets/>} />
                  <Route path="/datasets/preview/:dataset_id" element={<DatasetPreview />} />
                  <Route path="/datasets/add-dataset/" element={<AddDataset/>} >
                    <Route index element={<Navigate to="common-standards" replace />} />
                    <Route path="common-standards" element={<CommonStandards/>} />
                    <Route path="medical-folder-dataset" element={<MedicalFolderDataset/>} />
                  </Route>
                  <Route
                        path="*"
                        status={404}
                        element={
                          <main style={{ padding: "1rem" }}>
                            <p>Error 404: there is nothing here</p>
                          </main>
                        }
                      />

                </Route>
            </Routes>
        </Router>

        {props.result.show ? (
          <Popup
            icon={props.result.error ? 'alert' : 'checkInCircleFilled'}
            iconColor={props.result.error ? 'danger' : 'success'}
            title={props.result.error ? 'Error' : 'Success'}
            onClose={onResultModalClose}
          >
            <p>{props.result.message}</p>
          </Popup>
        ) : null}
      </div>
      <div className={`loader-frame ${props.result.loading ?  'active' : ''}`}>
          <div style={{width:"100%"}}>
              <div className="lds-ring">
                    <div></div>
                    <div></div>
                    <div></div>
                    <div></div>
              </div>
              <span style={{textAlign: "center", display:"block"}}>{props.result.text}</span>
          </div>
      </div>
    </EuiProvider>
  );
}


const mapStateToProps = (state) => {
  return {
    result : state.resultModal,
    auth   : {...state.auth},
    first_connection : state.first_connection
  }
}

export default connect(mapStateToProps,null)(App);
