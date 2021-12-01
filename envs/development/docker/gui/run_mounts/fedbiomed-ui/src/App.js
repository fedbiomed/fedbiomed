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


function App() {
  return (
    <div className="App">
        <Header/>
        <Router>
            <Routes>
              <Route exact path="/" element={<Home/>} />
              <Route path="/configuration" element={<Configuration/>} />
              <Route path="/repository" element={<Repository/>} />
            </Routes>
        </Router>
    </div>
  );
}

export default App;
