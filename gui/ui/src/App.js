import React from 'react'
import './App.css';
import {
  BrowserRouter as Router,
  Routes,
  Route
} from "react-router-dom";

import Home from './components/pages/Home'
import Configuration from './components/pages/Configuration';
import Repository from './components/pages/Repository';
import Header from './components/layout/Header';
import SideNav from './components/layout/SideNav';
import Datasets from './components/pages/Datasets';
import AddDataset from './components/pages/AddDataset';
import DatasetPreview from './components/pages/DatasetPreview';
import Modal from "./components/common/Modal"
import {connect, useDispatch} from 'react-redux'
import Button, {ButtonsWrapper} from "./components/common/Button";

function App(props) {

  const dispatch = useDispatch()

  const [headerName, setHeaderName] = React.useState('Home')

  /**
   * Change header based on active item
   * @param {string} val
   */
  const setHeader = (val) => {
      setHeaderName(val)
  }


  const onResultModalClose = () => {
    dispatch({type:'RESET_GLOBAL_MODAL'})
  }

  return (
    <div className="App">
      <Router>
        <div className="layout-wrapper">
          <div className="main-side-bar">
            <SideNav activePage={setHeader}/>
          </div>
          <div className="main-frame">
            <Header text={headerName}/>
              <div className="router-frame">
                <div className="inner"> 
                  <Routes>
                    <Route exact path="/" element={<Home/>} />
                    <Route path="/configuration" element={<Configuration/>} />
                    <Route path="/repository" element={<Repository/>} />
                    <Route path="/datasets" element={<Datasets/>} />
                    <Route path="/datasets/add-dataset" element={<AddDataset/>} />
                    <Route path="/datasets/preview/:dataset_id" element={<DatasetPreview setHeader={setHeader}/>} />
                  </Routes>
                  <div className={`loader-frame ${props.result.loading ?  'active' : ''}`}>
                    <div className="lds-ring">
                      <div></div>
                      <div></div>
                      <div></div>
                      <div></div>
                    </div>
                  </div>
                </div>
              </div>
          </div>
        </div>
      </Router>
      <Modal show={props.result.show} class="info-box" onModalClose={onResultModalClose}>
         <Modal.Header>
           { props.result.error ? (
               "Error"
           ) : "Success"}
         </Modal.Header>
        <Modal.Content>
            {props.result.message}
        </Modal.Content>
        <Modal.Footer>
               <ButtonsWrapper className={"float-right"}>
                        <Button onClick={onResultModalClose}>Close</Button>
                </ButtonsWrapper>
        </Modal.Footer>
      </Modal>
    </div>
  );
}


const mapStateToProps = (state) => {
  return {
    result : state.resultModal
  }
}

export default connect(mapStateToProps,null)(App);
