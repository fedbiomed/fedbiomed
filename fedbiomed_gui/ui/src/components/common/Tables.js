import React from 'react';
import styles from './Tables.module.css'

export const Table = (props) => {
    return(
        <table ref={props.tableRef} className={props.className} style={{...props.style}}>
            {props.children}
        </table>
    )
}


export const TableWrapper = (props) => {
    return(
        <div style={{maxHeight: props.maxHeight, ...props.style}} className={styles.table_wrapper}>
            <div className={styles.wrapper_inner}>
                {props.children}
            </div>
        </div>
    )
}

/**
 *
 * @param props
 * @returns {JSX.Element}
 * @constructor
 */
export const TableBody = (props) => {
    return(
        <tbody className={props.className} style={props.style}>
            {props.children}
        </tbody>
    )
}

/**
 *
 * @param props
 * @returns {JSX.Element}
 * @constructor
 */
export const TableRow = (props) => {
    return(
        <tr className={props.className} style={props.style}>
            {props.children}
        </tr>
    )
}

/**
 *
 * @param props
 * @returns {JSX.Element}
 * @constructor
 */
export const TableHead = (props) => {
    return(
        <thead className={props.className} style={props.style}>
            {props.children}
        </thead>
    )
}


/**
 *
 * @param props
 * @returns {JSX.Element}
 * @constructor
 */
export const TableCol = (props) => {
    return(
        <td onClick={props.onClick} className={props.className}>
            {props.transformation ? props.transformation(props.children) : props.children}
        </td>
    )
}


export const EntryTable = (props) => {
    return(
        <Table className={styles.entryTable}>
            {props.children}
        </Table>
    )
}

export const TableInfo = (props) => {
    const ExtractValues = (data) => {
        try{ return (data.value.toString())}
        catch(e){ return (data)}
    }
    return (
        <div className="table">
            <table className={styles.infoTable}>
                <TableBody>
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

                                    <td>
                                        {ExtractValues(props.info[item])}
                                        
                                    </td>
     
                                    {/* <td>{props.mode? props.info[item].value.toString(): props.info[item]}</td> */}
                                </tr>
                            )
                        }

                    })}
                </TableBody>
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
                row.push( <TableCol key={`td-${key}`} transformation={props.transformation}>{item.toString().substring(0,12)}</TableCol>)
            })           
            result.push(<TableRow key={`tr-${i}`}>{row}</TableRow>)
        }

        return result
    }


    if(props.children){
        return(
            <Table className={styles.dataTable}>
                {props.children}
            </Table>
            )
    }else{
        return (
            <TableWrapper maxHeight={props.maxHeight}>
                <Table className={styles.dataTable}>
                    <TableHead>
                        <TableRow>
                            {props.table.columns.map((item, key) => {
                                return (
                                    <TableCol key={key}>{item}</TableCol>
                                )
                            })}
                        </TableRow>
                    </TableHead>

                    <TableBody>
                        { return_rows() }
                    </TableBody>
                </Table>
            </TableWrapper>
        );
    }
}

export const SelectiveTable = (props) => {
    const tableRef = React.createRef()
    const [hoverColIndex, setHoverColIndex] = React.useState(null)
    const [hoverRowIndex, setHoverRowIndex] = React.useState(null)

    const handleTableHover = React.useCallback( event => {
        let index  = getColIndex(event)
        setHoverColIndex(index)
        index = getRowIndex(event)
        setHoverRowIndex(index)
    }, []);

    const handleTableColumnClick = React.useCallback(event => {
        if(props.hoverColumns) {
            let index  = getColIndex(event)
            if(index !== props.selectedColIndex){
                if(props.onSelect){
                    props.onSelect(index)
                }
            }
        } else {
            let index  = getRowIndex(event)
            if(index !== props.selectedRowIndex){
                if(props.onSelect){
                    props.onSelect(index)
                }
            }
        }
    }, [props]);

    const handleTableUnHover = React.useCallback(event => {
        setHoverColIndex(null)
        setHoverRowIndex(null)
    }, []);


    const setTableRef = React.useCallback( node => {

        if(tableRef.current){
              tableRef.current.removeEventListener('mouseover', handleTableHover);
              tableRef.current.removeEventListener('mouseout', handleTableHover);
              tableRef.current.removeEventListener('mousedown', handleTableColumnClick);
        }

        if(node){
            node.addEventListener('mouseover', handleTableHover);
            node.addEventListener('mouseout', handleTableUnHover);
            node.addEventListener('mousedown', handleTableColumnClick);
        }

        tableRef.current = node

    }, [handleTableUnHover, handleTableHover, handleTableColumnClick, tableRef])


    const getColIndex = (event) => {
        let target = event.target
        let index = [...target.parentElement.children].indexOf(target)
        return index
    }

    const getRowIndex = (event) => {
        let target = event.target
        let index = target.parentElement.rowIndex
        return index - 1  // I think this assumes there is a header row
    }

    return (
        <React.Fragment>
            <TableWrapper maxHeight={props.maxHeight}>
                <Table tableRef={setTableRef}>
                        <DataTableHead
                            table={props.table}
                            hoverColumns={props.hoverColumns}
                            hoverColIndex={hoverColIndex}
                            hoverRowIndex={hoverRowIndex}
                            theadClassName={`${props.theadClassName} ${styles.stickyHead}`}
                            theadStyle={props.theadStyle}
                            activeColIndex={props.selectedColIndex}
                            activeRowIndex={props.selectedRowIndex}
                            selectedLabel={props.selectedLabel}
                        />
                        <DataTableRows
                            table={props.table}
                            tbodyClassName={props.tbodyClassName}
                            tbodyStyle={props.tbodyStyle}
                            hoverColumns={props.hoverColumns}
                            hoverColIndex={hoverColIndex}
                            hoverRowIndex={hoverRowIndex}
                            activeColIndex={props.selectedColIndex}
                            activeRowIndex={props.selectedRowIndex}
                            tranformation={(data) => data.toString().substring(0,30)}
                        />
                </Table>
            </TableWrapper>
            <span className={styles.displayNote}>
                 Displays: {props.table.samples < props.table.displays ?
                                    props.table.samples : props.table.displays} / {props.table.samples}
            </span>
       </React.Fragment>
    );
};

SelectiveTable.defaultProps= {
  hoverColumns: true
}


/**
 * Component to build Table headers
 * @param props
 * @returns {JSX.Element}
 * @constructor
 */
export const DataTableHead = (props) => {
    return (
         <TableHead className={props.theadClassName} style={props.theadStyle}>
            <tr>
                {props.table.index && props.showIndex ? (
                            <th>{props.indexName ? props.indexName : "-"}</th>
                        ) : null
                }
                {props.table.columns.map((item, key) => {
                    if(props.hoverColumns){
                        return <th key={key} className={props.hoverColIndex === key ||
                                                    props.activeColIndex === key ?
                                                        styles.activeCol : null}>
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
         </TableHead>
    )
}


/**
 * Component to build table rows
 * @param props
 * @returns {JSX.Element}
 * @constructor
 */
export const DataTableRows = (props) => {
    return (
        <TableBody className={props.tbodyClassName} style={props.tbodyStyle} >
            {props.table.data.map((row, key) => {
                return(
                    <TableRow key={'row-'+key} className={(!props.hoverColumns) && ((props.hoverRowIndex === key) || (props.activeRowIndex === key)) ? styles.activeCol : props.className} >
                        <React.Fragment>
                            { props.table.index && props.showIndex ? (
                                <TableCol transformation={props.transformation} >
                                    {props.table.index[key]}
                                </TableCol>
                            ): null}
                            {row.map((col, key_col) => {
                                if(props.hoverColumns){
                                    return(
                                        <TableCol
                                            className={props.hoverColIndex === key_col ||
                                                        props.activeColIndex === key_col ?
                                                            styles.activeCol : null}
                                            key={`td-${key_col}`}
                                            transformation={props.transformation}
                                        >
                                            {col}
                                        </TableCol>
                                    )
                                }else{
                                    return(
                                        <TableCol key={`td-${key_col}`} transformation={props.transformation}>
                                            {col}
                                        </TableCol>
                                    )
                                }
                            })}
                        </React.Fragment>
                    </TableRow>
                 )
                })}
        </TableBody>
    )
}
