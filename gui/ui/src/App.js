import React, { useCallback, useEffect } from 'react'
import './App.css';
import '@elastic/eui/dist/eui_theme_light.css';
import { EuiProvider } from '@elastic/eui';
import {
  BrowserRouter as Router,
  Routes,
  Route
} from "react-router-dom";

import Home from './pages/Home'
import Configuration from './pages/Configuration';
import Repository from './pages/repository';
import SideNav from './components/layout/SideNav';
import Datasets from './pages/datasets';
import AddDataset from "./pages/datasets/AddDataset"
import DatasetPreview from './pages/datasets/DatasetPreview';
import Modal from "./components/common/Modal"
import {connect, useDispatch} from 'react-redux'
import Button, {ButtonsWrapper} from "./components/common/Button";
import CommonStandards from "./pages/datasets/CommonStandards";
import MedicalFolderDataset from "./pages/datasets/MedicalFolderDataset";
import Models from "./pages/models/Models";
import SingleModel from "./pages/models/SingleModel";
import Login from "./pages/authentication/Login";
import Register from "./pages/authentication/Register";
import useToken from './pages/authentication/useToken';
import PocEndpoints from './pages/authentication/PocEndpoints'; //for testing purposes
import LogUserLayout from './components/layout/LogUserLayout';
import UserAccount from './pages/authentication/UserAccount';


function App(props) {

  //const history = new createBrowserHistory();
  const { accessToken, removeToken, setToken, getAccessToken, checkIsTokenActive, readToken } = useToken();

  const dispatch = useDispatch();
  // const logOut = useCallback(() => {
  //   dispatch(Logout());
  // }, [dispatch]);


  const onResultModalClose = () => {
    dispatch({type:'RESET_GLOBAL_MODAL'});
  }

// useEffect(() => {
//   console.log("into use effect")
//   console.log(checkIsTokenActive())
//   EventBus.on("logOut", () => { logOut()});
//   return () => {EventBus.remove("logOut");};
// });

  console.log( readToken())
  console.log("APP")
  let style = {
  //   display: "none"
  };

  return (
    <EuiProvider colorMode="light">
      <div className="App" >
        <Router>
              <Routes>
                <Route path="/login/" element={<Login setToken={setToken}/>} />
                <Route path="/register/" element={<Register/>} />
                <Route path="/" element ={<LogUserLayout/>} >
                  <Route path="/" element={<Home/>} />
                  <Route path="/configuration/" element={<Configuration/>} />
                  <Route path="/user-account/" element={<UserAccount/>}>
                    {/* add here other pages for admin management */}
                  </Route>
                  <Route path="/repository/" element={<Repository/>} />
                  <Route path="/models/" element={<Models/>} />
                  <Route path="/models/preview/:model_id" element={<SingleModel />} />
                  <Route path="/datasets/" element={<Datasets/>} />
                  <Route path="/datasets/preview/:dataset_id" element={<DatasetPreview />} />
                  <Route path="/datasets/add-dataset/" element={<AddDataset/>} >
                    <Route index element={<CommonStandards/>} />
                    <Route path="medical-folder-dataset" element={<MedicalFolderDataset/>} />
                  </Route>
                  {/* Go to pocEndpoints page with the access token set */}
                  <Route path="/pocEndpoints/" element={<PocEndpoints accessToken={accessToken}
                                                                    removeToken={removeToken} 
                                                                    setToken={setToken}/>} />
                  {/* Dealing with inexistant paths TODO: render message into a erro box + add redirection*/}
                  {/*path="*" stands for all others routes */}
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
                {/* <Route path="/login/" element={<Login/>} /> */}
                {/* <Route path="/pocEndpoints/" element = {<pocEndpoints/>} /> */}
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
    auth   : state.auth
  }
}

export default connect(mapStateToProps,null)(App);
