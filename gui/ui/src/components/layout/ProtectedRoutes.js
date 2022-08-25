import React from 'react';
import SideNav from './SideNav'
import { Navigate, Outlet, useNavigate, NavLink} from "react-router-dom";
import {decodeToken, removeToken, setUser} from "../../store/actions/authActions";
import {useDispatch, useSelector, shallowEqual, connect} from "react-redux";
import {
    EuiHeaderSectionItemButton,
    EuiAvatar,
    EuiHeader,
    EuiHeaderSectionItem,
    EuiHeaderSection,
  EuiFlexGroup,
  EuiFlexItem,
  EuiLink,
  EuiPopover,
  EuiSpacer,
  EuiText,
  useGeneratedHtmlId,

} from '@elastic/eui'
import logo from "../../assets/img/fedbiomed-logo-small.png";


const mapStateToProps = (state) => {
    return {
        user : state.auth
    }
}

export const LoginProtected = connect(mapStateToProps, null)((props) => {

    const dispatch = useDispatch()
    let user = decodeToken()

    if(user && !props.user.is_auth){
        dispatch(setUser(user))
    }


    if(user) {
        return(
            <React.Fragment>
                <EuiHeader position={'fixed'}>
                    <EuiHeaderSection grow={false}>
                        <EuiHeaderSectionItem border="right">
                            <img alt="fedbiomed-logo" src={logo} style={{marginRight:10, width:30}}/>
                            Fed-BioMed
                        </EuiHeaderSectionItem>
                    </EuiHeaderSection>
                    <EuiHeaderSection >
                        <HeaderUserMenu name={`${props.user.name} - ${props.user.surname}`}/>
                    </EuiHeaderSection>
                </EuiHeader>
                <div className="layout-wrapper">
                    <div className="main-side-bar">
                        <SideNav/>
                    </div>
                    <div className="main-frame">
                        <div className="router-frame">
                            <div className="inner">
                                <Outlet />
                            </div>
                        </div>
                    </div>
                </div>
            </React.Fragment>
        )

    }else{
        return(
            <Navigate to="/login/" />
        )
    }

})



export const AdminProtected = (props) => {

    const {role} = useSelector((state) => state.auth, shallowEqual)

    if (role == "Admin"){
        return props.children
    }else{
        return(
            <Navigate to="/user-account/" />
        )
    }
};


const HeaderUserMenu = (props) => {
  const headerUserPopoverId = useGeneratedHtmlId({
    prefix: 'headerUserPopover',
  });
  const [isOpen, setIsOpen] = React.useState(false);
  const navigate = useNavigate()

  const onMenuButtonClick = () => {
    setIsOpen(!isOpen);
  };

  const closeMenu = () => {
    setIsOpen(false);
  };

  const button = (
    <EuiHeaderSectionItemButton
      aria-controls={headerUserPopoverId}
      aria-expanded={isOpen}
      aria-haspopup="true"
      aria-label="Account menu"
      onClick={onMenuButtonClick}
    >
      <EuiAvatar name={props.name} size="m" />
    </EuiHeaderSectionItemButton>
  );

  return (
    <EuiPopover
      id={headerUserPopoverId}
      button={button}
      isOpen={isOpen}
      anchorPosition="downRight"
      closePopover={closeMenu}
      panelPaddingSize="none"
    >
      <div style={{ width: 300 }}>
        <EuiFlexGroup
          gutterSize="m"
          className="euiHeaderProfile"
          responsive={false}
        >
          <EuiFlexItem grow={false}>
            <EuiAvatar name={props.name} size="xl" />
          </EuiFlexItem>

          <EuiFlexItem>
            <EuiText>
              <p>{props.name}</p>
            </EuiText>
            <EuiSpacer size="m" />
            <EuiFlexGroup>
              <EuiFlexItem>
                <EuiFlexGroup justifyContent="spaceBetween">
                  <EuiFlexItem grow={false}>
                    <EuiLink onClick={() => navigate('user-account')}>Account</EuiLink>
                  </EuiFlexItem>

                  <EuiFlexItem grow={false}>
                    <EuiLink onClick={() => removeToken()} >Log out</EuiLink>
                  </EuiFlexItem>
                </EuiFlexGroup>
              </EuiFlexItem>
            </EuiFlexGroup>
          </EuiFlexItem>
        </EuiFlexGroup>
      </div>
    </EuiPopover>
  );
};
