import React, {useState} from 'react';
import {EuiButton,
        EuiToolTip,
        EuiFlexItem,
        EuiFlexGroup,
        EuiForm,
        EuiFormRow,
        EuiFieldPassword,
        EuiFieldText,
        EuiSpacer,
        EuiIcon,
        EuiToast
} from '@elastic/eui';


import { EP_REGISTER} from '../../constants';
import {useNavigate} from "react-router-dom";
import {useDispatch} from "react-redux";
import axios from "axios";

const initialRegisterForm = {name: '', surname: '', email: '', password: '', confirm: ''}


const RegisterForm = (props) => {

    const navigate = useNavigate();
    const [registerForm, setRegisterForm] = useState(initialRegisterForm)
    const dispatch = useDispatch();
    const [error, setError] = useState({show : false, message : ''})
    const errorClose = () => setError({show:false, message:''})

    /**
     * Registration form action
     * @param event
     */
    const register = (event) => {

        // Prevent form submit
        event.preventDefault()
        let data = {
            email: registerForm.email,
            password: registerForm.password,
            name: registerForm.name,
            surname: registerForm.surname,
            confirm: registerForm.confirm
        }
        axios.post(EP_REGISTER, data).then((response) => {
            if (response.status === 201) {
              dispatch({type :'SUCCESS_MODAL', payload:'A request has been sent to an administrator to validate you registration.'})
            }
            navigate('/login')
        }).catch((error) => {
            if (error.response) {
                setError({show:true, message: error.response.data.message})
            }else{
                setError({show:true, message: error.toString()})
            }
        })
    }

    /**
     * Hadnler for form input changes
     * @param event
     */
    const handleChange = (event) => {
        let {value, name} = event.target
        setRegisterForm(prevNote => ({
          ...prevNote, [name]: value})
        )

    }

    return (
            <EuiForm component="form"  onSubmit={register} >
                 <EuiFlexGroup direction="column" >
                     <EuiFlexItem grow={false}>
                         {error.show ? (
                             <React.Fragment>
                                 <EuiSpacer size="l" />
                                 <EuiToast
                                        title="Error"
                                        color="danger"
                                        iconType="alert"
                                        onClose={errorClose}
                                        style={{width:400}}
                                      >
                                     <p>{error.message}</p>
                                 </EuiToast>
                             </React.Fragment>

                         ) : null}
                     </EuiFlexItem>
                      <EuiFlexItem grow={false}>
                        <EuiFormRow label={"Name"} hasEmptyLabelSpace>
                              <EuiFieldText
                                  onChange={handleChange}
                                  name='name'
                                  value={registerForm.name}
                               />
                        </EuiFormRow>
                     </EuiFlexItem >
                     <EuiFlexItem grow={false}>
                        <EuiFormRow label={"Surname"} hasEmptyLabelSpace>
                              <EuiFieldText
                                  onChange={handleChange}
                                  name='surname'
                                  value={registerForm.surname}
                               />
                        </EuiFormRow>
                     </EuiFlexItem >
                     <EuiFlexItem grow={false}>
                        <EuiFormRow label={"Enter your e-mail"} hasEmptyLabelSpace>
                              <EuiFieldText
                                  onChange={handleChange}
                                  text={registerForm.email}
                                  name='email'
                                  prepend={<EuiIcon type="email" />}
                                  value={registerForm.email}
                               />
                        </EuiFormRow>
                     </EuiFlexItem >
                     <EuiFlexItem grow={false}>
                          <EuiFormRow label={"Password"} hasEmptyLabelSpace>
                            <EuiToolTip
                                  display={"block"}
                                  position="right"
                                  title={"Attention!"}
                                  content="Password should be at least 8 character long,
                                  with at least one uppercase letter, one lowercase letter and one number"
                            >
                                 <EuiFieldPassword
                                        type='dual'
                                        name={"password"}
                                        value={registerForm.password}
                                        onChange={handleChange}
                                  />
                            </EuiToolTip>
                         </EuiFormRow>
                     </EuiFlexItem>
                     <EuiFlexItem grow={false}>
                          <EuiFormRow label={"Confirm Password"} hasEmptyLabelSpace>
                            <EuiToolTip
                                  display={"block"}
                                  position="right"
                                  title={"Attention!"}
                                  content="Password should be at least 8 character long,
                                  with at least one uppercase letter, one lowercase letter and one number"
                            >
                                 <EuiFieldPassword
                                        type='dual'
                                        name={"confirm"}
                                        value={registerForm.confirm}
                                        onChange={handleChange}
                                  />
                            </EuiToolTip>
                         </EuiFormRow>
                     </EuiFlexItem>
                     <EuiFlexItem grow={false} >
                         <EuiFormRow display="center">
                            <EuiButton type="submit" fill>
                                Register
                            </EuiButton>
                         </EuiFormRow>
                     </EuiFlexItem>
                 </EuiFlexGroup>
            </EuiForm>
    );
};

export default RegisterForm;