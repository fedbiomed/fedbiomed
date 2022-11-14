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
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager
from fedbiomed.common.constants import TrainingPlanApprovalStatus, TrainingPlanStatus
from fedbiomed.node.cli_utils._io import validated_path_input


tp_security_manager = TrainingPlanSecurityManager()


def register_training_plan():
    """Registers an authorized training plan in the database interactively through the CLI.

    Does not modify training plan file.
    """

    print('Welcome to the Fed-BioMed CLI data manager')
    name = input('Please enter a training plan name: ')
    description = input('Please enter a description for the training plan: ')

    # Allow files saved as txt
    path = validated_path_input(type="txt")

    # Register training plan
    try:
        tp_security_manager.register_training_plan(name=name,
                                     description=description,
                                     path=path)

    except AssertionError as e:
        try:
            tkinter.messagebox.showwarning(title='Warning', message=str(e))
        except ModuleNotFoundError:
            warnings.warn(f'[ERROR]: {e}')
        exit(1)

    print('\nGreat! Take a look at your data:')
    tp_security_manager.list_training_plans(verbose=True)


def update_training_plan():
    """Updates an authorized training plan in the database interactively through the CLI.

    Does not modify training plan file.

    User can either choose different training plan file (different path)
    to update training plan or same training plan file.
    """
    training_plans = tp_security_manager.list_training_plans(verbose=False)

    # Select only registered training plan to update
    training_plans = [m for m in training_plans if m['training_plan_type'] == TrainingPlanStatus.REGISTERED.value]
    if not training_plans:
        logger.warning('No registered training plans has been found to update')
        return

    options = [m['name'] + '\t Training plan ID ' + m['training_plan_id'] for m in training_plans]
    msg = "Select the training plan to update:\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\nSelect: "

    while True:
        try:

            # Get the selection
            opt_idx = int(input(msg)) - 1
            assert opt_idx in range(len(training_plans))
            training_plan_id = training_plans[opt_idx]['training_plan_id']

            if not training_plan_id:
                logger.warning('No matching training plan to update')
                return

            # Get the new file or same file.  User can provide same training plan file
            # with updated content or new training plan file.
            path = validated_path_input(type="txt")

            # Update training plan through training plan manager
            tp_security_manager.update_training_plan_hash(training_plan_id, path)

            logger.info('Training plan has been updated. Here all your training plans')
            tp_security_manager.list_training_plans(verbose=True)

            return

        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def approve_training_plan(sort_by_date: bool = True):
    """Approves a given training plan that has either Pending or Rejected status

    Args:
        sort_by_date: whether to sort by last modification date. Defaults to True.
    """
    if sort_by_date:
        sort_by = 'date_modified'
    else:
        sort_by = None
    non_approved_training_plans = tp_security_manager.list_training_plans(sort_by=sort_by,
                                                    select_status=[TrainingPlanApprovalStatus.PENDING,
                                                                   TrainingPlanApprovalStatus.REJECTED],
                                                    verbose=False)
    if not non_approved_training_plans:
        logger.warning("All training_plans have been approved or no training plan has been registered... aborting")
        return

    options = [m['name'] + '\t Training plan ID ' + m['training_plan_id'] + '\t training plan status ' +
               m['training_plan_status'] + '\tdate_last_action ' +
               str(m['date_last_action']) for m in non_approved_training_plans]

    msg = "Select the training plan to approve:\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\nSelect: "

    while True:
        try:
            opt_idx = int(input(msg)) - 1
            assert opt_idx in range(len(non_approved_training_plans))
            training_plan_id = non_approved_training_plans[opt_idx]['training_plan_id']
            tp_security_manager.approve_training_plan(training_plan_id)
            logger.info(f"Training plan {training_plan_id} has been approved. Researchers can now train the Training Plan" +
                        f" on Node {environ['NODE_ID']}")
            return

        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def reject_training_plan():
    """Rejects a given training plan that has either Pending or Approved status
    """
    approved_training_plans = tp_security_manager.list_training_plans(select_status=[TrainingPlanApprovalStatus.APPROVED,
                                                               TrainingPlanApprovalStatus.PENDING],
                                                verbose=False)

    if not approved_training_plans:
        logger.warning("All training plans have already been rejected or no training plan has been registered... aborting")
        return

    options = [m['name'] + '\t Training plan ID ' + m['training_plan_id'] + '\t training plan status ' +
               m['training_plan_status'] + '\tTraining plan Type ' + m['training_plan_type'] for m in approved_training_plans]

    msg = "Select the training plan to reject (this will prevent Researcher to run training plan on Node):\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\nSelect: "

    while True:
        try:
            opt_idx = int(input(msg)) - 1
            assert opt_idx in range(len(approved_training_plans))
            training_plan_id = approved_training_plans[opt_idx]['training_plan_id']
            notes = input("Please give a note to explain why training plan has been rejected: \n")
            tp_security_manager.reject_training_plan(training_plan_id, notes)
            logger.info(f"Training plan {training_plan_id} has been rejected. Researchers can not train training plan" +
                        f" on Node {environ['NODE_ID']} anymore")
            return

        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def delete_training_plan():
    """Deletes an authorized training plan in the database interactively from the CLI.

    Does not modify or delete training plan file.

    Deletes only registered and requested training_plans. For default training plans, files
    should be removed directly from the file system.
    """

    training_plans = tp_security_manager.list_training_plans(verbose=False)
    training_plans = [m for m in training_plans if m['training_plan_type'] in [TrainingPlanStatus.REGISTERED.value,
                                                       TrainingPlanStatus.REQUESTED.value]]
    if not training_plans:
        logger.warning('No training plans to delete')
        return

    options = [m['name'] + '\t Training plan ID ' + m['training_plan_id'] + '\t Training plan type ' +
               m['training_plan_type'] + '\tTraining plan status ' + m['training_plan_status'] for m in training_plans]
    msg = "Select the training plan to delete:\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\nSelect: "

    while True:
        try:

            opt_idx = int(input(msg)) - 1
            assert opt_idx in range(len(training_plans))
            training_plan_id = training_plans[opt_idx]['training_plan_id']

            if not training_plan_id:
                logger.warning('No matching training plan to delete')
                return
            # Delete training plan
            tp_security_manager.delete_training_plan(training_plan_id)
            logger.info('Training plan has been removed. Here your other training plans')
            tp_security_manager.list_training_plans(verbose=True)

            return

        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')


def view_training_plan():
    """Views source code for a training plan in the database

    If `environ[EDITOR]` is set then use this editor to view a copy of the training plan source code, so that
    any modification are not saved to the training plan,

    If `environ[EDITOR]` is unset or cannot be used to view the training plan, then print the training plan to the logger.

    If training plan cannot be displayed to the logger, then abort.
    """
    training_plans = tp_security_manager.list_training_plans(verbose=False)
    if not training_plans:
        logger.warning("No training plan has been registered... aborting")
        return

    options = [m['name'] + '\t Training plan ID ' + m['training_plan_id'] + '\t training plan status ' +
               m['training_plan_status'] for m in training_plans]

    msg = "Select the training plan to view:\n"
    msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
    msg += "\n\nDon't try to modify the training plan with this viewer, modifications will be dropped."
    msg += "\nSelect: "

    while True:
        try:
            opt_idx = int(input(msg)) - 1
            assert opt_idx in range(len(training_plans))
            training_plan_name = training_plans[opt_idx]['name']
        except (ValueError, IndexError, AssertionError):
            logger.error('Invalid option. Please, try again.')
            continue

        # TODO: more robust (when refactor whole CLI)
        # - check `training_plan` though it should never be None, as we just checked for it
        # - check after file copy though it should work
        # - etc.
        training_plan = tp_security_manager.get_training_plan_by_name(training_plan_name)
        training_plan_tmpfile = os.path.join(environ['TMP_DIR'], 'training_plan_tmpfile_' + str(uuid.uuid4()))
        shutil.copyfile(training_plan["training_plan_path"], training_plan_tmpfile)

        # first try to view using system editor
        editor = environ['EDITOR']
        result = os.system(f'{editor} {training_plan_tmpfile} 2>/dev/null')
        if result != 0:
            logger.info(f'Cannot view training plan with editor "{editor}", display via logger')
            # second try to print via logger (default output)
            try:
                with open(training_plan_tmpfile) as m:
                    training_plan_source = highlight(''.join(m.readlines()), PythonLexer(), Terminal256Formatter())
                    logger.info(f'\n\n{training_plan_source}\n\n')
            except Exception as err:
                logger.critical(f'Cannot display training plan via logger. Aborting. Error message is: {err}')

        os.remove(training_plan_tmpfile)
        return
