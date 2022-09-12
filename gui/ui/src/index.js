import React from 'react';
import './style.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import {createStore, applyMiddleware} from 'redux'
import {Provider} from 'react-redux'
import thunk from "redux-thunk" 
import RootReducer from './store/index'
import { createRoot } from 'react-dom/client';
import {setupAxios} from "./AxiosErrorHandler";

export const store = createStore(RootReducer, applyMiddleware(thunk))


const container = document.getElementById('root');
const root = createRoot(container);
setupAxios()

root.render(
    <Provider store={store}>
        <App />
    </Provider>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
