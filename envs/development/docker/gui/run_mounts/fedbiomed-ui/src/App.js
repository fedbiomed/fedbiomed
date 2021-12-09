import React from 'react'
import './App.css';
import {
  BrowserRouter as Router,
  Routes,
  Route
} from "react-router-dom";

import Home from './pages/Home' 
import Configuration from './pages/Configuration';
import Repository from './pages/Repository';
import Header from './components/Header';
import SideNav from './components/SideNav';
import Datasets from './pages/Datasets';
import AddDataset from './pages/AddDataset';
import DatasetPreview from './pages/DatasetPreview';

function App() {

  const [headerName, setHeaderName] = React.useState('Home')

  /**
   * Change header based on active item
   * @param {string} val
   */
  const setHeader = (val) => {
      setHeaderName(val)
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
                </div>
              </div>
          </div>
        </div>
      </Router>
    </div>
  );
}

export default App;
