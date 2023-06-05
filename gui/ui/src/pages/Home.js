import React from 'react'
import {Link} from 'react-router-dom'
import {ReactComponent as DocIcon} from "../assets/img/doc.svg";
import {ReactComponent as FileIcon} from "../assets/img/file.svg";
import {ReactComponent as DataIcon} from "../assets/img/database.svg";
import {ReactComponent as PlusIcon} from "../assets/img/plus.svg";
import {ReactComponent as ConfIcon} from "../assets/img/configuration.svg";
import {EuiTitle, EuiText, EuiTextAlign} from '@elastic/eui'


const Home = (props) => {

    const boxes = [
        [{
             title : 'Documentation',
             icon : DocIcon,
             link : 'https://fedbiomed.org/',
             text: 'Visit our documentation page to know more about Fed-BioMed Nodes and management.',
             internal: false,
         },
         {
             title : 'List of Files',
             icon : FileIcon,
             link : '/repository',
             text: 'List data files stored in the server.',
             internal: true,
         },
         {
             title : 'Dataset Management',
             icon : DataIcon,
             link : '/datasets',
             text: 'Manage datasets that are deployed on the node.',
             internal: true,
         }
        ],
        [{
            title : 'Load Datasets',
            icon : PlusIcon,
            link : '/datasets/add-dataset',
            text: 'Select dataset and deploy on the node',
            internal: true,
        },
        {
            title : 'Configuration',
            icon : ConfIcon,
            link : '/configuration',
            text: 'Display details of node configurations',
            internal: true,
        }
        ]

    ]

    return (
        <React.Fragment>
            <EuiTitle>
                <EuiTextAlign textAlign={"center"}>
                    <h2>Fed-BioMed Node GUI</h2>
                </EuiTextAlign>
            </EuiTitle>
            <EuiText>
                <EuiTextAlign textAlign={"center"}>
                    <p>Welcome to Fed-BioMed Node application. In this application, you can manage your data
                        files that are deployed in the node or load new datasets into the node.  </p>
                </EuiTextAlign>
            </EuiText>
            <div className="frame-content">
                {boxes.map( (row, key1) => {
                    return (
                        <div key={key1} className={'row'} style={{justifyContent:'center'}}>
                            {row.map((item, key2) => {
                                return(
                                   item.internal ? (
                                       <Link key={key2} className="fed-box-link" to={{ pathname: item.link}}>
                                            <div className={'fed-box'}>
                                                <div className={"title"}>
                                                    {item.title}
                                                </div>
                                                <div className={"icon"}>
                                                    {<item.icon/>}
                                                </div>
                                                <div className={"text"}>
                                                    {item.text}
                                                </div>
                                            </div>
                                       </Link>
                                   ) : (
                                      <a key={key2} className="fed-box-link" href={item.link}  rel="noopener noreferrer" target='_blank'>
                                            <div className={'fed-box'}>
                                                <div className={"title"}>
                                                    {item.title}
                                                </div>
                                                <div className={"icon"}>
                                                    {<item.icon/>}
                                                </div>
                                                <div className={"text"}>
                                                    {item.text}
                                                </div>
                                            </div>
                                        </a>
                                   )
                                )
                             })}
                        </div>
                    )
                })}
            </div>
            <div className="frame-footer">

            </div>
        </React.Fragment>
    )
}

export default Home