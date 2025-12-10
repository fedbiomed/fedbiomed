from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.dataset_controller import TabularController
from fedbiomed.common.dataset_reader import CsvReader
from fedbiomed.common.exceptions import FedbiomedError

from .analytics_strategy import AnalyticsStrategy


class TabularAnalytics(AnalyticsStrategy):
    def mean(self, dataset: TabularDataset, params=None):
        # Read data from dataset controller

        # Initialize controller
        controller = dataset._controller
        if not isinstance(controller, TabularController):
            raise FedbiomedError(
                f"Expected TabularController, got {type(controller).__name__} instead."
            )

        # Initialize reader
        reader = controller._reader
        if not isinstance(reader, CsvReader):
            raise FedbiomedError(
                f"Expected CsvReader, got {type(reader).__name__} instead."
            )

        # Read data
        df = reader.data
        if df is None:
            raise FedbiomedError("Dataset is empty or data could not be read.")

        # Determine columns
        if params is None or params.get("cols") is None:
            cols = [c for c, t in df.schema.items() if t.is_numeric()]
        else:
            cols = params["cols"]

        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        # Select only numeric columns
        numeric_cols = [c for c in cols if df.schema[c].is_numeric()]
        if not numeric_cols:
            return {}

        result = df.select(numeric_cols).mean()
        return result.to_dict(as_series=False)
