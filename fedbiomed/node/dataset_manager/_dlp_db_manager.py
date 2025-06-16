# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Manage the node's database table for handling DLP / DLB
"""

from fedbiomed.common.dataloadingplan import (
    DataLoadingPlan,  # noqa   # not used yet, sketch
)
from fedbiomed.common.db import DBTable


class DlpDatabaseManager:
    """Manage the node's database table for handling DLP / DLB

    Facility for storing data, retrieving data and getting data info
    for the node. Currently uses TinyDB.
    """

    _dlp_table: DBTable

    # Dispatch functions from DatasetManager:
    #
    # Use DatasetDatabaseManager for dataset DB table handling functions:
    # get_by_id
    # search_by_tags
    # search_conflicting_tags
    # add_database remove_database
    # modify_database_info
    # list_my_data
    # obfuscate_private_information
    #
    # Use DlpDatabaseManager for DLP/DLB DB table handling functions:
    # list_dlp
    # get_dlp_by_id
    # remove_dlp_by_id
    # save_data_loading_plan
    # save_data_loading_block
    #
    # Functions belonging the the dataset specific controllers or readers in new design
    # CsvController : read_csv  load_csv_dataset  get_csv_data_types
    # MnistController  : load_default_database get_torch_dataset_shape
    # MednistController : load_mednist_database  get_torch_dataset_shape
    # ImageFolderController  : load_images_dataset  get_torch_dataset_shape
    #
    # Unused functions:  get_data_loading_blocks_by_id  load_as_dataloader
