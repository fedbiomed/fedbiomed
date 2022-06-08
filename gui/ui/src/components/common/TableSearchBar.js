import React from 'react';
import {Select, Text} from "./Inputs";
import styles from "./TableSearchBar.module.css"


const TableSearchBar = (props) => {

    const [search, setSearch] = React.useState(null)
    const [by, setBy] = React.useState(null)
    const selectRef = React.useRef()
    const [timeoutObject, setTimeoutObject] = React.useState({
                                                                        name: '',
                                                                        typing: false,
                                                                        typingTimeout: 0
                                                                    })

    React.useEffect( () => {
        setBy(selectRef.current.value)
    }, [selectRef])

    const onSearchChange = (e) => {
        let search = e.target.value
        if (timeoutObject.typingTimeout) {
           clearTimeout(timeoutObject.typingTimeout);
        }
        setTimeoutObject({
           name: e.target.value,
           typing: false,
           typingTimeout: setTimeout(function () {
               setSearch(search)
               props.onSearch(search, by)
             }, 600)
        });
    }

    const onByChange = (e) => {
        let byS = e.target.value
        setBy(byS)
        props.onSearch(search, byS)
    }

    return (
            <div className={styles.wrapper}>
                <div>
                    <Text
                        className={styles.fields}
                        placeholder={'Search'}
                        onChange={onSearchChange}
                    />
                </div>
                {props.by ? (
                    <React.Fragment>
                        <div className={styles.by}>
                            By
                        </div>
                        <div>
                            <Select
                                selectRef={selectRef}
                                options={props.byOptions}
                                className={styles.fields}
                                onChange={onByChange}
                            />
                        </div>
                    </React.Fragment>
                ) : null}
            </div>
    );
};

export default TableSearchBar;