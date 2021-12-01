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

function App() {
  return (
    <div className="App">
      <Router>
        <div className="layout-wrapper">
          <div className="main-side-bar">
            <SideNav />
          </div>
          <div className="main-router">
            <Header/>
            
                <Routes>
                  <Route exact path="/" element={<Home/>} />
                  <Route path="/configuration" element={<Configuration/>} />
                  <Route path="/repository" element={<Repository/>} />
                </Routes>
            
          </div>
        </div>
      </Router>
    </div>
  );
}

export default App;
