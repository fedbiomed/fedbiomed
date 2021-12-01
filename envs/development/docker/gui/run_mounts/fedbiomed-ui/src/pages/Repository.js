import React from 'react';
import {connect} from 'react-redux'
import { getFilesFromRepository } from '../store/actions/repository';


const Repository = (props) => {

    React.useEffect(() => {

        props.createProject({path : []})
    }, [props])

    return (
        <div>
            Reposirtory
        </div>
    );
}

const mapStateToProps = (state) => {
    return {
        repository : state.repository
    }
}

const mapDispatchToProps = (dispatch) => {
    return {
        createProject: (data) => dispatch(getFilesFromRepository(data))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(Repository);