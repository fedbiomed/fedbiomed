from typing import List, Optional, Union

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.db import TinyTableConnector
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger


class DatasetDB(TinyTableConnector):
    _table_name = "Datasets"
    _id_name = "dataset_id"

    def search_by_tags(self, tags: Union[tuple, list]) -> list:
        """Searches for data with given tags.

        Args:
            tags:  List of tags

        Returns:
            The list of matching datasets
        """
        return self.get_all_by_condition("tags", lambda x: set(tags).issubset(x))

    def search_conflicting_tags(self, tags: Union[tuple, list]) -> list:
        """Searches for registered data that have conflicting tags with the given tags

        Args:
            tags:  List of tags

        Returns:
            The list of conflicting datasets
        """
        return self.get_all_by_condition(
            "tags",
            lambda x: set(tags).issubset(x) or set(x).issubset(tags),
        )

    def remove_database(self, dataset_id: str):
        """Removes a dataset from database.

        Args:
            dataset_id: Dataset unique ID.

        Raises:
            FedbiomedError: If no dataset with given ID exists in the database.
        """
        if self.get_by_id(dataset_id) is None:
            _msg = ErrorNumbers.FB322.value + f": No dataset found with id {dataset_id}"
            logger.error(_msg)
            raise FedbiomedError(_msg)
        self.delete_by_id(dataset_id)

    def modify_database_info(self, dataset_id: str, modified_dataset: dict):
        """Modifies a dataset in the database.

        Args:
            dataset_id: ID of the dataset to modify.
            modified_dataset: key-value pairs to replace in the existing entry.

        Raises:
            FedbiomedError: conflicting tags with existing dataset
        """
        # Check that there are not existing dataset with conflicting tags
        if "tags" in modified_dataset:
            # the dataset to modify is ignored (can conflict with its previous tags)
            conflicting_ids = [
                _["dataset_id"]
                for _ in self.search_conflicting_tags(modified_dataset["tags"])
                if _["dataset_id"] != dataset_id
            ]
            if len(conflicting_ids) > 0:
                msg = (
                    f"{ErrorNumbers.FB322.value}, one or more registered dataset has conflicting tags: "
                    f" {' '.join([_['name'] for _ in conflicting_ids])}"
                )
                logger.critical(msg)
                raise FedbiomedError(msg)

        _ = self.update_by_id(dataset_id, modified_dataset)


class DlpDB(TinyTableConnector):
    _table_name = "Dlps"
    _id_name = "dlp_id"

    def list_by_target_dataset_type(
        self, target_dataset_type: Optional[str] = None
    ) -> List[dict]:
        """Return all existing DataLoadingPlans.

        Args:
            target_dataset_type: (str or None) if specified, return only dlps matching the requested target type.

        Returns:
            An array of dict, each dict is a DataLoadingPlan
        """
        if target_dataset_type is None:
            return self.all()

        if not isinstance(target_dataset_type, str):
            raise FedbiomedError(
                f"Wrong input type for target_dataset_type. "
                f"Expected str, got {type(target_dataset_type)} instead."
            )
        if target_dataset_type not in [t.value for t in DatasetTypes]:
            raise FedbiomedError(
                "target_dataset_type should be of the values defined in "
                "fedbiomed.common.constants.DatasetTypes"
            )

        return self.get_all_by_value("target_dataset_type", target_dataset_type)


class DlbDB(TinyTableConnector):
    _table_name = "Dlbs"
    _id_name = "dlb_id"
