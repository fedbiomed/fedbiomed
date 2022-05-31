import React from 'react';
import Accordion from "../../components/layout/Accordion";
import dropDownStyle from "../../components/layout/Accordion.module.css"
import {TableData} from "../../components/common/Tables"

const BidsSubjectInformation = (props) => {
    if(props.subjects.available_subjects){
        return (
            <React.Fragment>
                <Accordion
                    color={dropDownStyle.green}
                    label={props.subjects.available_subjects.length.toString() + " subject available for training"}
                >
                    <TableData
                        table={{
                            columns: ["Available Subjects"],
                            data : Array.from(props.subjects.available_subjects, x => [x]),
                            index: Array.from(Array(props.subjects.available_subjects.length).keys())
                        }
                    }/>
                </Accordion>

                <Accordion
                    color={props.subjects.missing_entries.length === 0 ? dropDownStyle.green : dropDownStyle.red }
                    label={props.subjects.missing_entries.length.toString() + " subjects are existing in the BIDS " +
                        "directory but not in the reference tabular data"}
                >
                    <TableData
                        table={{
                            columns: ["Missing Entries"],
                            data : Array.from(props.subjects.missing_entries, x => [x]),
                            index: Array.from(Array(props.subjects.missing_entries.length).keys())
                        }
                        }/>
                </Accordion>
                <Accordion
                    color={props.subjects.missing_folders.length === 0 ? dropDownStyle.green : dropDownStyle.red }
                    label={props.subjects.missing_folders.length.toString() + " subject for that is declared in the " +
                        "CSV does not exist in the BIDS root folder  for training"}
                >
                    <TableData
                        table={{
                            columns: ["Missing Folders"],
                            data : Array.from(props.subjects.missing_folders, x => [x]),
                            index: Array.from(Array(props.subjects.missing_folders.length).keys())
                        }
                        }/>
                </Accordion>
            </React.Fragment>
        );
    }
    return null

};

export default BidsSubjectInformation;