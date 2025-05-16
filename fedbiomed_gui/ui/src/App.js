import React from 'react'
import './App.css';
import '@elastic/eui/dist/eui_theme_light.css';
import { EuiProvider } from '@elastic/eui';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";

import Home from './pages/Home'
import Configuration from './pages/Configuration';
import Repository from './pages/repository';
import Datasets from './pages/datasets';
import AddDataset from "./pages/datasets/AddDataset"
import DatasetPreview from './pages/datasets/DatasetPreview';
import Modal from "./components/common/Modal"
import {connect, useDispatch} from 'react-redux'
import Button, {ButtonsWrapper} from "./components/common/Button";
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
import UserAccount from './pages/authentication/UserAccount';


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
                  <Route path="/configuration/" element={<Configuration/>} />
                  <Route path="/user-account/" element={<UserAccount/>}>
                      <Route index element={<UserInfo/>} />
                      <Route path={"info"} element={<UserInfo/>} />
                      <Route path={"change-password"} element={<PasswordChange/>} />
                      <Route path={"user-management"} element={<AdminProtected redirect_to={'/user-account'}><UserManagement/></AdminProtected>}/>
                      <Route path={"account-requests"} element={<AdminProtected redirect_to={'/user-account'}><AccountRequestManagement/></AdminProtected>}/>
                  </Route>
                  <Route path="/repository/" element={<Repository/>} />
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

        <Modal show={props.result.show} class="info-box" id="message" onModalClose={onResultModalClose}>
            <Modal.Header>
              { props.result.error ? (
                  "Error"
              ) : "Success"}
            </Modal.Header>
          <Modal.Content>
              {props.result.message}
          </Modal.Content>
          <Modal.Footer>
                  <ButtonsWrapper alignment={"right"}>
                          <Button onClick={onResultModalClose}>Close</Button>
                  </ButtonsWrapper>
          </Modal.Footer>
        </Modal>
        <Modal show={false} class="token-expired" id="msg-token-expired" onModalClose={onResultModalClose}>
            <Modal.Header>
              {
                  "Error"
                }
            </Modal.Header>
            <Modal.Content>
                {props.result.message}
            </Modal.Content>
            <Modal.Footer>
                  <ButtonsWrapper alignment={"right"}>
                            <Button onClick={onResultModalClose}>Close</Button>
                    </ButtonsWrapper>
            </Modal.Footer>
          </Modal>
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
