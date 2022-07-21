import { EP_PROTECTED, EP_ADMIN } from '../../constants';
import axios from 'axios';

const PocEndpoints = (props) => {

    const token = props.accessToken; // sessionStorage.getItem('accessToken');
    const get_protected_data = () => {
        axios.get(EP_PROTECTED, {
            // TODO: Find a way to add headers automatically to all requests
            headers: {
                'Authorization': `Bearer ${token}` 
            }
        })
            .then( response => {
                console.log(response) 
            })
            .catch( (error) => {
                alert(error.response.data.message)
            })
    }

    const get_admin_data = () => {
        axios.get(EP_ADMIN, {
            headers: {
                'Authorization': `Bearer ${token}` 
            }
        })
            .then( response => {
                console.log(response) 
            })
            .catch( (error) => {
                alert(error.response.data.message)
            })
    }

    return (
        <div>
            <div>
                <button onClick={get_protected_data}>Get Protected Data</button>
            </div>
            <div>
                <button onClick={get_admin_data}>Get Admin Data</button>
            </div>
        </div>
      );
}

export default PocEndpoints;