"""
implementation of Round Joye-Libert class of the node component
"""
import pickle
import uuid
from typing import Any, List, Union

import numpy as np
import torch

from fedbiomed.common.constants import VEParameters
from fedbiomed.common.joye_libert_scheme import JLS, EncryptedNumber
from fedbiomed.common.joye_libert_utils import quantize, reverse_quantize
from fedbiomed.common.logger import logger
from fedbiomed.common.vector_encoding import VES
from fedbiomed.node.environ import environ
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.round import Round


class RoundJL(Round):
    def __init__(
        self,
        model_kwargs: dict = None,
        training_kwargs: dict = None,
        training: bool = True,
        dataset: dict = None,
        model_url: str = None,
        model_class: str = None,
        params_url: str = None,
        job_id: str = None,
        researcher_id: str = None,
        history_monitor: HistoryMonitor = None,
        node_args: Union[dict, None] = None,
        current_round: int = 0,
    ):
        super().__init__(
            model_kwargs,
            training_kwargs,
            training,
            dataset,
            model_url,
            model_class,
            params_url,
            job_id,
            researcher_id,
            history_monitor,
            node_args,
        )
        self.vector_encoder = VES(
            ptsize=VEParameters.KEY_SIZE.value // 2,
            addops=VEParameters.NUM_CLIENTS.value,
            valuesize=VEParameters.VALUE_SIZE.value,
            vectorsize=VEParameters.VECTOR_SIZE.value,
        )

        self.jl = JLS(nusers=VEParameters.NUM_CLIENTS.value, VE=self.vector_encoder)
        self.pp, _, self.user_key = self.jl.Setup(VEParameters.KEY_SIZE.value)
        self.current_round = current_round

    def _load_model_obj(self, params_path):
        if params_path.endswith(".pt"):
            try:
                self.model.load(params_path, to_params=False)
            except Exception as e:
                error_message = f"Cannot initialize model parameters: f{str(e)}"
                return self._send_round_reply(success=False, message=error_message)
        if params_path.endswith(".pkl"):
            try:
                with open(params_path, "rb") as handle:
                    params_qt_enc = pickle.load(handle)
                    params_qt_dec = self.vector_encoder.decode(params_qt_enc)
                    logger.info(f"Decoded params done, length: {len(params_qt_dec)}")
                    params_vector = torch.as_tensor(params_qt_dec)
                    torch.nn.utils.vector_to_parameters(
                        params_vector, self.model.parameters()
                    )
            except Exception as e:
                error_message = f"Cannot initialize model parameters: f{str(e)}"
                return self._send_round_reply(success=False, message=error_message)

    def _load_model_file(self):
        status = None
        params_path = ""
        if self.params_url.endswith(".pt"):
            status, params_path = self.repository.download_file(
                self.params_url, "my_model_" + str(uuid.uuid4()) + ".pt"
            )
        if self.params_url.endswith(".pkl"):
            status, params_path = self.repository.download_file(
                self.params_url, "my_model_" + str(uuid.uuid4()) + ".pkl"
            )
        return params_path, status

    def _upload_model(
        self,
        ptime_after,
        ptime_before,
        results,
        rtime_after,
        rtime_before,
        import_module,
    ):
        # Upload results
        results["researcher_id"] = self.researcher_id
        results["job_id"] = self.job_id

        model_ptxt = self.model.after_training_params(vector=True)

        logger.info(f"Quantization of model parameters")
        model_ptxt_qt: np.ndarray = quantize(model_ptxt)
        model_ptxt_qt: List[int] = [x.item() for x in model_ptxt_qt]
        logger.info(
            f"Quantization of model parameters done")
        # TODO For the moment users_key = server_key = 0
        logger.info(f"Protecting model parameters with key {self.user_key}")
        model_ctxt_qt_enc: List[EncryptedNumber] = self.jl.Protect(
            pp=self.pp,
            sk_u=self.user_key,
            tau=self.current_round,
            x_u_tau=model_ptxt_qt,
        )
        logger.info(
            f"Protecting model parameters done")

        results["node_id"] = environ["NODE_ID"]
        results["model_params"] = model_ctxt_qt_enc
        try:
            # TODO : should test status code but not yet returned
            # by upload_file
            filename = environ["TMP_DIR"] + "/node_params_" + str(uuid.uuid4()) + ".pkl"
            with open(filename, "wb") as handle:
                pickle.dump(
                    results["model_params"],
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            res = self.repository.upload_file(filename)
            logger.info("results uploaded successfully ")
        except Exception as e:
            is_failed = True
            error_message = f"Cannot upload results: {str(e)}"
            return self._send_round_reply(success=False, message=error_message)

        # end : clean the namespace
        try:
            del self.model
            del import_module
        except Exception as e:
            logger.debug(f"Exception raise while deleting model {e}")
            pass

        return self._send_round_reply(
            success=True,
            timing={
                "rtime_training": rtime_after - rtime_before,
                "ptime_training": ptime_after - ptime_before,
            },
            params_url=res["file"],
        )
