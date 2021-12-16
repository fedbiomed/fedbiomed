import React from 'react';

export const TableInfo = (props) => {

    return (
        <div className="table">
            <table className="info">
                <tbody>
                    {Object.keys(props.info).map((item, key) => {
                        if(props.info[item].editable && props.edit){
                            return (
                                <tr key={key}>
                                    <td className="title">{item}</td>
                                    <td>
                                        {props.info[item].input}
                                    </td>
                                </tr>
                            )
                        }else{
                            return (
                                <tr key={key}>
                                    <td className="title">{item}</td>
                                    <td>{props.info[item].value.toString()}</td>
                                </tr>
                            )
                        }

                    })}
                </tbody>
            </table>
        </div>
    );
}


export const TableData = (props) => {


    /**
     * Return DOM <td> elements
     * @returns {*[]}
     */
    const return_rows = () => {
        let result = []
        for(let i=0; i<props.table.index.length; i++){
            let row = []
            props.table.data[i].forEach( (item, key) => {
                row.push( <td>{item}</td>)
            })           
            result.push(<tr>{row}</tr>)
        }

        return result
    }



    return (
        <React.Fragment>
            <div className="table">
                <table className="data">
                    <tbody>
                        <tr>
                            {props.table.columns.map((item, key) => {
                                return (
                                    <th key={key}>{item}</th> 
                                )
                            })}
                        </tr>
                        { return_rows() }
                    </tbody>
                </table>
            </div>
        </React.Fragment>
    );
}