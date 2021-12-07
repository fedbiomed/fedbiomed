import React from 'react';
import {connect} from 'react-redux'
import { getFilesFromRepository } from '../store/actions/repositoryActions';
import RepositoryItem from '../components/RepositoryItem';
import RepositoryBar from '../components/RepositoryBar';


const Repository = (props) => {

    React.useEffect(() => {

        props.getFiles({path : []})
        console.log(props.repository)
    }, [props.repository.files.length])


    const onItemClick = (type, path) => {
        
        console.log(type)
        console.log(path)
        if(type === "dir"){
            console.log('clicked')
            props.getFiles({path : path})
        }
    }


    const onAddActionClick = (event, type, path) => {
        console.log('Inner')
        event.stopPropagation()
    }


    return (
        <React.Fragment>
            <div className="frame-header">
            </div>
            <div className="frame-content">
                <div className="main-repository">
                    {Object.keys(props.repository.files).map( (item, key) => {
                        return (
                            <RepositoryBar key={`bat-${key}`}>            
                                {props.repository.files[item].map( (item,key) => {
                                
                                    return (
                                        <RepositoryItem 
                                            key={`item-${key}`}
                                            name={item.name}
                                            path={item.path}
                                            type={item.type}
                                            registered={item.registered}
                                            onItemClick={onItemClick}
                                            onAddActionClick={onAddActionClick}
                                        />
                                    )
                                })}
                            </RepositoryBar> 
                        )
                    })
                    } 
                </div>  
            </div> 
            <div className="frame-footer">

            </div>
        </React.Fragment>
    );
}

const mapStateToProps = (state) => {
    return {
        repository : state.repository
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        getFiles: (data) => dispatch(getFilesFromRepository(data))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(Repository);