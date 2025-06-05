# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Tuple
import warnings
from copy import copy
from collections import defaultdict
from fedbiomed.common.data import DataLoadingPlan, MedicalFolderController, MedicalFolderBase, MapperBlock
from fedbiomed.node.cli_utils._io import validated_path_input


def add_medical_folder_dataset_from_cli(interactive: bool,
                                        dataset_parameters: Optional[dict],
                                        dlp: Optional[DataLoadingPlan]) -> Tuple[str, dict, DataLoadingPlan]:
    print('Please select the root folder of the Medical Folder dataset')
    path = validated_path_input(type='dir')
    controller = MedicalFolderController(path)
    dataset_parameters = {} if dataset_parameters is None else dataset_parameters

    choice = input('\nWould you like to select a demographics csv file? [y/N]\n')
    if choice.lower() == 'y':
        # get tabular file
        print('Please select the demographics file (must be CSV or TSV)')
        tabular_file_path = validated_path_input(type='csv')
        # get index col from user
        column_values = controller.demographics_column_names(tabular_file_path)
        print("\nHere are all the columns contained in demographics file:\n")
        for i, col in enumerate(column_values):
            print(f'{i:3} : {col}')
        if interactive:
            keep_asking_for_input = True
            while keep_asking_for_input:
                try:
                    index_col = input('\nPlease input the (numerical) index of the column containing '
                                      'the subject ids corresponding to image folder names \n')
                    index_col = int(index_col)
                    keep_asking_for_input = False
                except ValueError:
                    warnings.warn('Please input a numeric value (integer)')
        dataset_parameters['tabular_file'] = tabular_file_path
        dataset_parameters['index_col'] = index_col
    modality_folder_names, _ = controller.modalities_candidates_from_subfolders()
    print("\nThe following modalities were detected:\n", "\n".join([m for m in modality_folder_names]))
    # TODO: add CLI support for DLP, temporarily disactivated (not working yet)
    # 
    #if interactive:
    #    choice = input('\nWould you like to associate the detected modalities with other modality names? [y/N]\n')
    #    if choice.lower() == 'y':
    #        dlb = get_map_modalities2folders_from_cli(modality_folder_names)
    #        if dlp is None:
    #            dlp = DataLoadingPlan({MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS: dlb})
    #        else:
    #            dlp.update({MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS: dlb})
    return path, dataset_parameters, dlp


def get_map_modalities2folders_from_cli(modality_folder_names: List[str]) -> MapperBlock:
    modality_names = ['Manually insert new modality name', *copy(MedicalFolderBase.default_modality_names)]
    map_modalities_to_folders = defaultdict(list)
    for modality_folder in modality_folder_names:
        keep_asking_for_this_modality = True
        while keep_asking_for_this_modality:
            for i, m in enumerate(modality_names):
                print(f"{i:3} : {m}")
            try:
                modality_idx = input(f"\nPlease choose the modality corresponding to\n {modality_folder} \nby inserting"
                                     f" the index number from the list above\n")
                modality_idx = int(modality_idx)
            except ValueError:
                warnings.warn('Please input a numeric value (integer)')
            else:
                # Only execute this branch if a numeric value was correctly inserted
                if modality_idx == 0:
                    keep_asking = True
                    while keep_asking:
                        new_modality_name = input('\nPlease input the new modality name\n')
                        confirm = input(f'\nThe new modality {new_modality_name} will be added to the list. Do'
                                        f'you confirm? [Y/n]')
                        if confirm.lower() == 'y' or confirm == '':
                            modality_names.append(new_modality_name)
                            keep_asking = False
                elif modality_idx >= len(modality_names):
                    warnings.warn('Please input a number equal to one of the indices in the list above.')
                else:
                    map_modalities_to_folders[modality_names[modality_idx]].append(modality_folder)
                    keep_asking_for_this_modality = False
    dlb = MapperBlock()
    dlb.map = dict(map_modalities_to_folders)
    return dlb
