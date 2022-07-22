import React from 'react'
import './App.css';
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
import useToken from './pages/authentication/useToken';
import PocEndpoints from './pages/authentication/PocEndpoints';
import Logout from "./pages/authentication/Logout";


function App(props) {

  const { accessToken, removeToken, setToken } = useToken();

  const logOut = useCallback(() => {
    dispatch(logout());
  }, [dispatch]);

  const dispatch = useDispatch()

  const onResultModalClose = () => {
    dispatch({type:'RESET_GLOBAL_MODAL'})
  }
  let style = {
  //   display: "none"
  };

  return (
    <React.Fragment>
    <div className="App">
      <Router>
        <div className="layout-wrapper">
          <div className="main-side-bar" id="#my_id" style={style}>
            <SideNav/>
          </div>
          <div className="main-frame">
              <div className="router-frame">
                <div className="inner"> 
                  {/* If the user is not logged in, redirect towards login page */}
                  {!accessToken && accessToken!=="" && accessToken!== undefined?  
                  <Login setToken={setToken} />
                  :(
                    <>
                      <Routes>
                        <Route exact path="/" element={<Home/>} />
                        {/* <Route path="/login/" element={<Login/>} /> */}
                        {/* <Route path="/pocEndpoints/" element = {<pocEndpoints/>} /> */}
                        <Route path="/configuration/" element={<Configuration/>} />
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
                      </Routes>
                    </>
                  )}
                  {/* <Routes>
                    <Route exact path="/" element={<Home/>} />
                    <Route path="/login/" element={<Login/>} />
                    <Route path="/pocEndpoints/" element = {<pocEndpoints/>} />
                    <Route path="/configuration/" element={<Configuration/>} />
                    <Route path="/repository/" element={<Repository/>} />
                    <Route path="/models/" element={<Models/>} />
                    <Route path="/models/preview/:model_id" element={<SingleModel />} />
                    <Route path="/datasets/" element={<Datasets/>} />
                    <Route path="/datasets/preview/:dataset_id" element={<DatasetPreview />} />
                    <Route path="/datasets/add-dataset/" element={<AddDataset/>} >
                        <Route index element={<CommonStandards/>} />
                        <Route path="medical-folder-dataset" element={<MedicalFolderDataset/>} />
                    </Route>
                  </Routes> */}
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
                </div>
              </div>
          </div>
        </div>
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
    </div>
    <div>
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

    </React.Fragment>
  );
}


const mapStateToProps = (state) => {
  return {
    result : state.resultModal
  }
}

export default connect(mapStateToProps,null)(App);
