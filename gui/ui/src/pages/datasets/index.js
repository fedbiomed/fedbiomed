import React from 'react';
import {connect} from "react-redux"
import {Link} from "react-router-dom"
import { listDatasets,  removeDataset, addDefaultDataset, searchDataset } from '../../store/actions/datasetsActions';
import {ReactComponent as GarbageLogo} from '../../assets/img/garbage.svg'
import {ReactComponent as LaunchIcon} from '../../assets/img/launch.svg'
import Modal from '../../components/common/Modal';
import Button from '../../components/common/Button'
import {Text} from '../../components/common/Inputs'
import {EntryTable, TableBody} from "../../components/common/Tables";


export const Datasets = (props) => {

    const listDatasetsAction = props.listDataSetsAction
    const searchDatasetAction = props.searchDatasetAction
    const [modal, setModal] = React.useState({show : false, header:null, content:null, approve:null })
    const [search, setSearch] = React.useState('')
    const [searchRT, setSearchRT] = React.useState('')
    const [timeoutObject, setTimeoutObject] = React.useState({
                                                                        name: '',
                                                                        typing: false,
                                                                        typingTimeout: 0
                                                                    })

    React.useEffect(() => {
        // Get list of datasets
        listDatasetsAction({})
    }, [listDatasetsAction])


    /**
     * Hook for default dataset result
     */
    React.useEffect(() => {

        if(props.datasets.default_dataset.success){
            //Popup success
            listDatasetsAction({})
        }else if(props.datasets.default_dataset.waiting){
            //Display loading
        }

    }, [
                props.datasets.default_dataset.success,
                props.datasets.default_dataset.waiting,
                listDatasetsAction
    ])


    /**
     * Search Dataset Action
     */
    const searchDataset = (e) => {
        let search = e.target.value
        if (timeoutObject.typingTimeout) {
           clearTimeout(timeoutObject.typingTimeout);
        }
        setSearchRT(search)
        setTimeoutObject({
           name: e.target.value,
           typing: false,
           typingTimeout: setTimeout(function () {
               setSearch(search)
               searchDatasetAction({search:search});
             }, 600)
        });
    }

    const clearSearch = () => {
        setSearch('')
        setSearchRT('')
    }

    /**
     * Parse tags array to div of tags
     * @param tags
     * @returns List[{JSX.Element}]
     */
    const parseTags = (tags) => {
        
        let tag_result = tags.map((item, key) => {
            return (
            <div key={key} className="tag">
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
                handleClose()
                break;
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
        props.addDefaultDataset({tags : ["#MNIST", "#dataset"], name : "MNIST", desc: "Default MNIST dataset"})
    }

    return (
        <React.Fragment>
            <div className="frame-header">
                <div className={`row`}>
                    <p>This page displays loaded datasets in the node.
                    </p>
                    {
                        defaultDataStatus() === false ? (
                            <Button wrapperStyle={{marginLeft:'auto'}} onClick={onAddDefaultDataset}>
                                Add MNIST Dataset
                            </Button>
                        ) : null
                    }
                </div>
                <div className={"row"}>
                    <div className={"note"} style={{width:'80%'}}>
                        <div>
                        Please click on
                        <div style={{display:'inline-block', marginLeft:10, marginRight: 10}} className="icon"><LaunchIcon width={"15px"} height={"15px"}/></div>
                        button to display details of the dataset. To remove the dataset, please click on
                        <div style={{display:'inline-block', marginLeft:10, marginRight: 10}} className="icon"><GarbageLogo width={"15px"} height={"15px"}/></div>
                        button.
                        </div>
                    </div>
                    <div className="form-control with-button" style={{width:'20%'}}>
                        <Text
                            placeholder={'Search in datasets'}
                            onChange={searchDataset}
                            value={searchRT}
                        />
                        <div className={"input-clear"}
                             style={{ visibility: searchRT === "" ? 'hidden' : 'visible'}}
                             onClick={clearSearch}
                        >X</div>
                    </div>
                </div>
            </div>
            <div className="frame-content">
                <EntryTable>
                    <TableBody>
                        <tr>
                            <th>Name</th>
                            <th>Type</th>
                            <th>Tags</th>
                            <th>Description</th>
                            <th className="center">Action</th>
                        </tr>
                        { search === '' ? (
                            props.datasets.datasets && props.datasets.datasets.map( (item,key) => {
                            return (
                                <tr key={key}>
                                    <td> {item.name}</td>
                                    <td> {item.data_type}</td>
                                    <td> {parseTags(item.tags)}</td>
                                    <td> {item.description}</td>
                                    <td className="center">
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
                                </tr>)
                        })) : (
                            props.datasets.search && props.datasets.search.map( (item,key) => {
                                return (
                                    <tr key={key}>
                                        <td> {item.name}</td>
                                        <td> {item.data_type}</td>
                                        <td> {parseTags(item.tags)}</td>
                                        <td> {item.description}</td>
                                        <td className="center">
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
                                    </tr> )
                        }))}

                    </TableBody>
                </EntryTable>
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
        addDefaultDataset: (data) => dispatch(addDefaultDataset(data)),
        searchDatasetAction: (data) => dispatch(searchDataset(data))
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