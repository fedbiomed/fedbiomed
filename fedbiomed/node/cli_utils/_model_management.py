import os
import shutil
import tkinter.messagebox
import warnings
import uuid
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter
from fedbiomed.node.environ import environ
from fedbiomed.common.logger import logger
from fedbiomed.node.model_manager import ModelManager
from fedbiomed.common.constants  import TrainingPlanApprovalStatus, ModelTypes
from fedbiomed.node.cli_utils._io import validated_path_input


model_manager = ModelManager()


def register_model():
    """Registers an authorized model in the database interactively through the CLI.

    Does not modify model file.
    """

    print('Welcome to the Fed-BioMed CLI data manager')
    name = input('Please enter a model name: ')
    description = input('Please enter a description for the model: ')

    # Allow files saved as txt
    path = validated_path_input(type="txt")

    # Register model
    try:
        model_manager.register_model(name=name,
                                     description=description,
                                     path=path)

    except AssertionError as e:
        try:
            tkinter.messagebox.showwarning(title='Warning', message=str(e))
        except ModuleNotFoundError:
            warnings.warn(f'[ERROR]: {e}')
        exit(1)

    print('\nGreat! Take a look at your data:')
    model_manager.list_models(verbose=True)


def update_model():
    """Updates an authorized model in the database interactively through the CLI.

    Does not modify model file.

    User can either choose different model file (different path)
    to update model or same model file.
    """
    models = model_manager.list_models(verbose=False)

    # Select only registered model to update
    models = [m for m in models if m['model_type'] == ModelTypes.REGISTERED.value]
    if not models:
        logger.warning('No registered models has been found to update')
        return

    options = [m['name'] + '\t Model ID ' + m['model_id'] for m in models]
    msg = "Select the model to update:\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\nSelect: "

    while True:
        try:

            # Get the selection
            opt_idx = int(input(msg)) - 1
            assert opt_idx in range(len(models))
            model_id = models[opt_idx]['model_id']

            if not model_id:
                logger.warning('No matching model to update')
                return

            # Get the new file or same file.  User can provide same model file
            # with updated content or new model file.
            path = validated_path_input(type="txt")

            # Update model through model manager
            model_manager.update_model_hash(model_id, path)

            logger.info('Model has been updated. Here all your models')
            model_manager.list_models(verbose=True)

            return

        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def approve_model(sort_by_date: bool = True):
    """Approves a given model that has either Pending or Rejected status

    Args:
        sort_by_date: whether to sort by last modification date. Defaults to True.
    """
    if sort_by_date:
        sort_by = 'date_modified'
    else:
        sort_by = None
    non_approved_models = model_manager.list_models(sort_by=sort_by,
                                                    select_status=[TrainingPlanApprovalStatus.PENDING,
                                                                   TrainingPlanApprovalStatus.REJECTED],
                                                    verbose=False)
    if not non_approved_models:
        logger.warning("All models have been approved or no model has been registered... aborting")
        return

    options = [m['name'] + '\t Model ID ' + m['model_id'] + '\t model status ' +
               m['model_status'] + '\tdate_last_action ' +
               str(m['date_last_action']) for m in non_approved_models]

    msg = "Select the model to approve:\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\nSelect: "

    while True:
        try:
            opt_idx = int(input(msg)) - 1
            assert opt_idx in range(len(non_approved_models))
            model_id = non_approved_models[opt_idx]['model_id']
            model_manager.approve_model(model_id)
            logger.info(f"Model {model_id} has been approved. Researchers can now train the Training Plan" +
                        f" on Node {environ['NODE_ID']}")
            return

        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def reject_model():
    """Rejects a given model that has either Pending or Approved status
    """
    approved_models = model_manager.list_models(select_status=[TrainingPlanApprovalStatus.APPROVED,
                                                               TrainingPlanApprovalStatus.PENDING],
                                                verbose=False)

    if not approved_models:
        logger.warning("All models have already been rejected or no model has been registered... aborting")
        return

    options = [m['name'] + '\t Model ID ' + m['model_id'] + '\t model status ' +
               m['model_status'] + '\tModel Type ' + m['model_type'] for m in approved_models]

    msg = "Select the model to reject (this will prevent Researcher to run model on Node):\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\nSelect: "

    while True:
        try:
            opt_idx = int(input(msg)) - 1
            assert opt_idx in range(len(approved_models))
            model_id = approved_models[opt_idx]['model_id']
            notes = input("Please give a note to explain why model has been rejected: \n")
            model_manager.reject_model(model_id, notes)
            logger.info(f"Model {model_id} has been rejected. Researchers can not train model" +
                        f" on Node {environ['NODE_ID']} anymore")
            return

        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def delete_model():
    """Deletes an authorized model in the database interactively from the CLI.

    Does not modify or delete model file.

    Deletes only registered and requested models. For default models, files
    should be removed directly from the file system.
    """

    models = model_manager.list_models(verbose=False)
    models = [m for m in models if m['model_type'] in [ModelTypes.REGISTERED.value, ModelTypes.REQUESTED.value]]
    if not models:
        logger.warning('No models to delete')
        return

    options = [m['name'] + '\t Model ID ' + m['model_id'] + '\t Model_type ' +
               m['model_type'] + '\tModel status ' + m['model_status'] for m in models]
    msg = "Select the model to delete:\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\nSelect: "

    while True:
        try:

            opt_idx = int(input(msg)) - 1
            assert opt_idx in range(len(models))
            model_id = models[opt_idx]['model_id']

            if not model_id:
                logger.warning('No matching model to delete')
                return
            # Delete model
            model_manager.delete_model(model_id)
            logger.info('Model has been removed. Here your other models')
            model_manager.list_models(verbose=True)

            return

        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def view_model():
    """Views source code for a model in the database

    If `environ[EDITOR]` is set then use this editor to view a copy of the model source code, so that
    any modification are not saved to the model,

    If `environ[EDITOR]` is unset or cannot be used to view the model, then print the model to the logger.

    If model cannot be displayed to the logger, then abort.
    """
    models = model_manager.list_models(verbose=False)
    if not models:
        logger.warning("No model has been registered... aborting")
        return

    options = [m['name'] + '\t Model ID ' + m['model_id'] + '\t model status ' +
               m['model_status'] for m in models]

    msg = "Select the model to view:\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\n\nDon't try to modify the model with this viewer, modifications will be dropped."
    msg += "\nSelect: "

    while True:
        try:
            opt_idx = int(input(msg)) - 1
            assert opt_idx in range(len(models))
            model_name = models[opt_idx]['name']
        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')
            continue

        # TODO: more robust (when refactor whole CLI)
        # - check `model` though it should never be None, as we just checked for it
        # - check after file copy though it should work
        # - etc.
        model = model_manager.get_model_by_name(model_name)
        model_tmpfile = os.path.join(environ['TMP_DIR'], 'model_tmpfile_' + str(uuid.uuid4()))
        shutil.copyfile(model["model_path"], model_tmpfile)

        # first try to view using system editor
        editor = environ['EDITOR']
        result = os.system(f'{editor} {model_tmpfile} 2>/dev/null')
        if result != 0:
            logger.info(f'Cannot view model with editor "{editor}", display via logger')
            # second try to print via logger (default output)
            try:
                with open(model_tmpfile) as m:
                    model_source = highlight(''.join(m.readlines()), PythonLexer(), Terminal256Formatter())
                    logger.info(f'\n\n{model_source}\n\n')
            except Exception as err:
                logger.critical(f'Cannot display model via logger. Aborting. Error message is: {err}')

        os.remove(model_tmpfile)
        return
