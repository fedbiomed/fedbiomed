import React from 'react';
import {connect} from "react-redux"
import {Link} from "react-router-dom"
import { listDatasets,  removeDataset, addDefaultDataset } from '../store/actions/datasetsActions';
import {ReactComponent as GarbageLogo} from '../assets/img/garbage.svg'
import {ReactComponent as LaunchIcon} from '../assets/img/launch.svg'
import Modal from '../components/Modal';
import Button from '../components/Button'

//import {Modal, Button} from "react-bootstrap"

export const Datasets = (props) => {

    const listDatasetsAction = props.listDataSetsAction
    const [modal, setModal] = React.useState({show : false, header:null, content:null, approve:null })

    React.useEffect(() => {
        // Get list of datasets
        listDatasetsAction()
    }, [])


    /**
     * Hook for default dataset result
     */
    React.useEffect(() => {

        if(props.datasets.default_dataset.success){
            //Popup success
            listDatasetsAction()
        }else if(props.datasets.default_dataset.waiting){
            //Display loading
        }

    }, [
                props.datasets.default_dataset.success,
                props.datasets.default_dataset.waiting,
    ])



    /**
     * Parse tags array to div of tags
     * @param tags
     * @returns List[{JSX.Element}]
     */
    const parseTags = (tags) => {
        
        let tag_result = tags.map((item) => {
            return (
            <div className="tag">
                {item}
            </div> 
            )
        })

        return tag_result
    }

    /**
     * Handles modal window close action
     */
    const handleClose = () => {
        setModal({...modal, show:false})
    }

    /**
     * Function that opens modal window when clicking any 
     * the delete icon
     * @param {event} e 
     * @param {object} data 
     */
    const openModalOnDelete = (e, data) => {
        setModal({
            show : true,
            header: 'Delete Dataset',
            content: 'Dataset will be removed. Are you sure?',
            approve: 'Remove',
            action : 'remove',
            data   : data
        })
    }

    /**
     * When the condition is approved in modal window
     */
    const handleModalApprove = () => {
        switch (modal.action) {
            case 'remove':  
                props.removeDatasetAction(modal.data)
            default:
                handleClose()
        }
    }

    /**
     * Checks is there any default dataset registered
     * in the list of datasets
     * @returns {boolean}
     */
    const defaultDataStatus = () => {
        console.log(props.datasets.datasets)
        if(props.datasets.datasets){
            let dlist = props.datasets.datasets
            let index = dlist.findIndex( x =>  x.data_type === 'default')

            if(index > -1){
                return true
            }else{
                return false
            }
        }
    }

    /**
     * Handle on add default dataset button
     * clicked. Send request to add MNIST dataset
     */
    const onAddDefaultDataset = () => {

        // Send empty data by default
        // it will be MNIST
        props.addDefaultDataset({})
    }

    return (
        <React.Fragment>
            <div className="frame-header">
                <div className={`row`}>
                    <p>List of datasets loaded in the node</p>
                    {
                        defaultDataStatus() === false ? (
                            <Button style={{'margin-left':'auto'}} onClick={onAddDefaultDataset}>
                                Add MNIST Dataset
                            </Button>
                        ) : null
                    }
                </div>
            </div>
            <hr/>
            <div className="frame-content">

                <table className="datasets">
                    <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Tags</th>
                        <th>Description</th>
                        <th class="center">Action</th>
                    </tr>
                    { props.datasets.datasets && props.datasets.datasets.map( (item,key) => {
                        return (    
                            <tr key={key}>
                                <td> {item.name}</td>
                                <td> {item.data_type}</td>
                                <td> {parseTags(item.tags)}</td>
                                <td> {item.description}</td>
                                <td class="center">
                                    <div className="action-buttons">
                                        <div className="icon" >
                                            <Link to={{pathname: `preview/${item.dataset_id}`}}>
                                                <LaunchIcon/>
                                            </Link>
                                        </div>
                                        <div className="icon delete" onClick={(event) => openModalOnDelete(event, item)}>
                                            <GarbageLogo/>
                                        </div>
                                    </div>
                                </td>
                            </tr>
                        )
                    })}

                </table>

            </div>
            <div className="frame-footer">
                 <Link to="/datasets/add-dataset">
                    <Button variant="primary" style={{float:'right'}}>Add Dataset</Button>
                </Link>
            </div>
            <Modal show={modal.show} width="35%" onModalClose={handleClose} >
                <Modal.Header>
                    {modal.header}
                </Modal.Header>
                <Modal.Content>{modal.content}</Modal.Content>
                <Modal.Footer>
                    <Button type="negative" onClick={handleClose}>
                        Cancel
                    </Button>
                    <Button onClick={handleModalApprove}>
                                {modal.approve}
                    </Button>
                </Modal.Footer>
            </Modal>
        </React.Fragment>
    );
}


/**
 * Pass action to props of component
 * @param {function} dispatch 
 * @returns 
 */
const mapDispatchToProps = (dispatch) => {
    return {
        listDataSetsAction : (data) => dispatch(listDatasets(data)),
        removeDatasetAction : (data) => dispatch(removeDataset(data)),
        addDefaultDataset: (data) => dispatch(addDefaultDataset(data))
    }
}

/**
 * Map global state
 * @param {*} state 
 * @returns 
 */
const mapStateToProps = (state) => {
    return {
        datasets : state.datasets
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(Datasets);