import React from 'react'
import { EuiHorizontalRule } from '@elastic/eui';

export const Modal = (props) => {

    const [show, setShow] = React.useState(props.show)

    /**
     * Method for closıng modal window
     */
    const closeModal = () => {

        // Close modal windows
        setShow(false)

        // Update close actıon if it is set in the parent component
        if (props.onModalClose){
            props.onModalClose()
        } 
    }

    // Catch close action triggered by the parent components
    React.useEffect( () => { setShow(props.show) }, [props.show])


    return (
        <div style={{
                    display : show === true ? 'flex' : 'none'
                    }} 
            className="modal-overlay">

            <div style={{width:props.width ? props.width : 'unset'}}
                className={`modal-inner shadow ${props.class ? props.class : ''}`}>

                <div className="close" onClick={closeModal}>X</div>

                {props.children} 

            </div>         
        </div>
    )
}


export const Header = (props) => {

    return (
        <div className="model-header">
             {props.children}
             <EuiHorizontalRule />
        </div>
    )
}


export const Content = (props) => {
    return (
        <div className="modal-content">
             {props.children}
        </div>
    )
}


export const Footer = (props) => {
    return (
        <div className={`modal-footer ${props.spread ? props.spread : ''}`}>
            {props.children}
        </div>
    )
}


export const Action = (props) => {
    return (
        <div className="modal-actions">
            {props.children}
        </div>
    )
}


export default Object.assign(Modal, {
        Action,
        Footer,
        Header,
        Content
})