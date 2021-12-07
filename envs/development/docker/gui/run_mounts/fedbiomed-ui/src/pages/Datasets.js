import React from 'react';
import {connect} from "react-redux"
import {Link} from "react-router-dom"
import { listDatasets } from '../store/actions/datasetsActions';
import { removeDataset } from '../store/actions/datasetsActions';
import {ReactComponent as GarbageLogo} from '../assets/img/garbage.svg'
import {ReactComponent as PlusLogo} from '../assets/img/plus.svg'
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
            content: 'Dataset will be remvoed. Are you sure?',
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

    return (
        <React.Fragment>
            <div className="frame-header">
                <p>List of datasets loaded in the node</p>
                <Link to="/datasets/add-dataset">
                    <Button variant="primary" style={{float:'right'}}>Add Dataset</Button>
                </Link>
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
                    { props.datasets.datasets.map( (item,key) => {
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
        removeDatasetAction : (data) => dispatch(removeDataset(data))
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