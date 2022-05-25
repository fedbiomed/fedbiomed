import React from 'react';
import styles from './Tables.module.css'

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
                row.push( <td key={`td-${key}`}>{item.toString().substring(0,12)}</td>)
            })           
            result.push(<tr key={`tr-${i}`}>{row}</tr>)
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



export const SelectiveTable = (props) => {
    const tableRef = React.createRef()
    const [hoverColIndex, setHoverColIndex] = React.useState(null)

    React.useEffect( () => {
        if(tableRef.current){
            tableRef.current.addEventListener('mouseover', handleTableHover);
            tableRef.current.addEventListener('mouseout', handleTableUnHover);
            tableRef.current.addEventListener('mousedown', handleTableColumnClick);
        }

        return () => {
          setHoverColIndex(null)
            if(tableRef.current){
              tableRef.current.removeEventListener('mouseover', handleTableHover);
              tableRef.current.removeEventListener('mouseout', handleTableHover);
            }
        };

    }, [])


    const handleTableHover = (event) => {
        let index  = getIndex(event)
        setHoverColIndex(index)
    };

    const handleTableColumnClick = (event) => {
        let index  = getIndex(event)
        if(index !== props.selectedColIndex){
            if(props.onSelect){
                props.onSelect(index)
            }
        }
    };

    const handleTableUnHover = (event) => {
        setHoverColIndex(null)
    };

    const getIndex = (event) => {
        let target = event.target
        let index = [...target.parentElement.children].indexOf(target)
        return index
    }

    return (
        <React.Fragment>
            <div style={props.style} className={styles.table_wrapper}>
                <div className={styles.wrapper_inner}>
                    <table ref={tableRef} className={styles.dataTable}>
                            <thead className={props.theadClassName} style={props.theadStyle}>
                                <TableHead
                                    table={props.table}
                                    hoverColumns={true}
                                    hoverColIndex={hoverColIndex}
                                    activeColIndex={props.selectedColIndex}
                                    selectedLabel={props.selectedLabel}
                                />
                            </thead>
                            <tbody className={props.tbodyClassName} style={props.tbodyStyle} >
                                <TableRows
                                    table={props.table}
                                    hoverColumns={true}
                                    hoverColIndex={hoverColIndex}
                                    activeColIndex={props.selectedColIndex}
                                />
                            </tbody>
                    </table>

                </div>
            </div>
            <span className={styles.displayNote}>
                 Displays: {props.table.samples < props.table.displays ?
                                    props.table.samples : props.table.displays} / {props.table.samples}
            </span>
        </React.Fragment>
    );
};

/**
 * Component to build Table headers
 * @param props
 * @returns {JSX.Element}
 * @constructor
 */
const TableHead = (props) => {
    return (
            <tr>
                {props.table.columns.map((item, key) => {

                    if(props.hoverColumns){
                        return <th className={props.hoverColIndex === key ||
                                                    props.activeColIndex === key ?
                                                        styles.activeCol : null}
                                    key={key}>
                                    <React.Fragment>
                                        {props.activeColIndex === key ? (
                                            <span className={styles.selectedLabel}>
                                                {props.selectedLabel ? props.selectedLabel : 'Selected'}
                                            </span>
                                        ) : null }
                                        {item}
                                    </React.Fragment>
                                </th>

                    }else{
                       return <th key={key}>{item}</th>
                    }

                })}
            </tr>
    )
}


/**
 * Component to build table rows
 * @param props
 * @returns {JSX.Element}
 * @constructor
 */
const TableRows = (props) => {


    return (
        <React.Fragment>
            {props.table.data.map((row, key) => {
                    return(
                        <tr className={props.className}>

                                <React.Fragment>
                                    {row.map((col, key_col) => {
                                        if(props.hoverColumns){
                                            return(
                                                <td className={props.hoverColIndex === key_col ||
                                                                props.activeColIndex === key_col ?
                                                                    styles.activeCol : null}
                                                    key={`td-${key_col}`}>
                                                    {col.toString().substring(0,12)}
                                                </td>
                                            )
                                        }else{
                                             <td key={`td-${key_col}`}>{col.toString().substring(0,12)}</td>
                                        }

                                    })}
                                </React.Fragment>

                        </tr>
                     )
                })}
        </React.Fragment>

    )

}