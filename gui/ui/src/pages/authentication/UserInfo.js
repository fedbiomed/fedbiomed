import React from 'react';
import {useSelector} from 'react-redux'
import {EuiDescriptionList} from "@elastic/eui";

const UserInfo = () => {

    const user = useSelector( (state) => state.auth)

    return (
        <React.Fragment>
                <EuiDescriptionList
                  listItems={[
                      {
                          title: "Role",
                          description: user.role ? user.role : 'Unknown'
                      },
                      {
                          title: "Name",
                          description: user.user_name ? user.user_name : 'Unknown'
                      },
                      {
                          title: "Surname",
                          description: user.user_surname ? user.user_surname : 'Unknown'
                      },
                      {
                          title: "Email",
                          description: user.email ? user.email : 'Unknown'
                      }
                  ]}
                  type="column"
                  compressed
                />
        </React.Fragment>
    );
};

export default UserInfo;