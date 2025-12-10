from typing import Dict


class AnalyticsStrategy:
    def mean(self, **kwargs) -> Dict:
        """Calculate mean of features across the dataset

        Iterates through dataset items (tuples of data and target tensors/numpy arrays)
        and computes the mean of both data and target features manually.

        Returns:
            Dictionary with 'data' and 'target' keys containing their respective means.
            Preserves the input data type (Tensor or numpy array).
        """
        data_sum = 0
        target_sum = 0
        count = 0

        for idx in range(len(self)):
            data, target = self[idx]
            data_sum += data
            target_sum += target
            count += 1

        if count == 0:
            return {"data": None, "target": None}

        # Calculate means by dividing by count
        data_mean = data_sum / count
        target_mean = target_sum / count

        return {"data": data_mean, "target": target_mean}
