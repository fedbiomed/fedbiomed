import unittest
import torch

from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from unittest.mock import patch, MagicMock
from fedbiomed.common.privacy import DPController
from fedbiomed.common.exceptions import FedbiomedDPControllerError


class TestDPController(unittest.TestCase):
    # Fake Dataset
    class DS(Dataset):
        def __getitem__(self):
            return 1, 1

        def __len__(self):
            return 1, 1

    def setUp(self) -> None:
        self.dp_args_c = {"type": "central", "sigma": 0.1, "clip": 0.1}
        self.dp_args_l = {"type": "local", "sigma": 0.1, "clip": 0.1}

        self.dpc = DPController(self.dp_args_c)
        self.dpl = DPController(self.dp_args_l)

        self.patcher_privacy_engine = patch("opacus.PrivacyEngine.__init__", MagicMock(return_value=None))
        self.patcher_privacy_engine_make_private = patch("opacus.PrivacyEngine.make_private")
        self.patcher_module_validator = patch("opacus.validators.ModuleValidator.is_valid")
        self.patcher_module_validator_fix = patch("opacus.validators.ModuleValidator.fix")

        self.privacy_engine = self.patcher_privacy_engine.start()
        self.privacy_engine_make_private = self.patcher_privacy_engine_make_private.start()
        self.module_validator = self.patcher_module_validator.start()
        self.module_validator_fix = self.patcher_module_validator_fix.start()

        self.privacy_engine_make_private.return_value = (None, None, None)
        self.module_validator.return_value = None
        self.module_validator_fix.return_value = None

        pass

    def tearDown(self) -> None:
        self.patcher_privacy_engine.stop()
        self.patcher_module_validator.stop()
        self.patcher_module_validator_fix.stop()

        pass

    def test_dep_controller_01_init_1(self):
        dp_controller = DPController()
        self.assertEqual(dp_controller._dp_args, {}, "_dp_args is not none")
        self.assertFalse(dp_controller._is_active, "DPController is active where it should be inactive")

    def test_dep_controller_02_init_2(self):
        """Tests builds DP controller with CENTRAL DP """
        dp_args = {"type": "central", "sigma": 0.1, "clip": 0.1}
        dp_controller = DPController(dp_args)
        self.assertTrue("sigma_CDP" in dp_controller._dp_args, "Sigma CDP is not set when DP type is central")
        self.assertEqual(dp_controller._dp_args["sigma"], 0.0, "Sigma is not set when DP type is central")
        self.assertEqual(dp_controller._dp_args["clip"], dp_args["clip"], "Sigma is not set when DP type is central")
        self.assertEqual(dp_controller._dp_args["type"], dp_args["type"], "Sigma is not set when DP type is central")

    def test_dep_controller_03_init_3(self):
        """Tests builds DP controller with LOCAL DP """
        dp_args = {"type": "local", "sigma": 0.1, "clip": 0.1}
        dp_controller = DPController(dp_args)
        self.assertFalse("sigma_CDP" in dp_controller._dp_args, "Sigma CDP is set when DP type is local")
        self.assertEqual(dp_controller._dp_args["sigma"], dp_args["sigma"], "Sigma is not set when DP type is local")
        self.assertEqual(dp_controller._dp_args["clip"], dp_args["clip"], "Sigma is not set when DP type is local")
        self.assertEqual(dp_controller._dp_args["type"], dp_args["type"], "Sigma is not set when DP type is local")

    def test_dep_controller_04_init_4(self):
        """Tests builds DP controller with invalid arguments """

        # Invalid DP type
        dp_args = {"type": "invalid", "sigma": 0.1, "clip": 0.1}
        with self.assertRaises(FedbiomedDPControllerError):
            DPController(dp_args)

        # Invalid sigma
        dp_args = {"type": "local", "sigma": 0, "clip": 0.1}
        with self.assertRaises(FedbiomedDPControllerError):
            DPController(dp_args)

        # Invalid clip
        dp_args = {"type": "local", "sigma": 0.1, "clip": 0}
        with self.assertRaises(FedbiomedDPControllerError):
            DPController(dp_args)

    def test_dep_controller_05_validate_and_fix_model(self):
        """Tests builds DP controller with invalid arguments """

        model = MagicMock()
        self.module_validator.return_value = True
        self.dpl._is_active = True
        self.dpl.validate_and_fix_model(model)
        self.module_validator.assert_called_once_with(model)

        self.module_validator.return_value = False
        self.dpl.validate_and_fix_model(model)
        self.module_validator_fix.assert_called_once_with(model)

    @patch('fedbiomed.common.privacy.DPController.validate_and_fix_model')
    def test_dep_controller_06_before_training(self, validate_and_fix):
        """Tests before training method with different scenarios"""

        model_false = MagicMock()
        opt_false = MagicMock()
        loader_false = MagicMock()

        model = Module()
        opt = Adam([torch.zeros([2, 4])])
        loader = DataLoader(TestDPController.DS())

        with self.assertRaises(FedbiomedDPControllerError):
            self.dpl.before_training(model_false, opt, loader)

        with self.assertRaises(FedbiomedDPControllerError):
            self.dpl.before_training(model, opt_false, loader)

        with self.assertRaises(FedbiomedDPControllerError):
            self.dpl.before_training(model, opt, loader_false)

        validate_and_fix.return_value = model
        self.privacy_engine_make_private.side_effect = Exception
        with self.assertRaises(FedbiomedDPControllerError):
            self.dpl.before_training(model, opt, loader)

        self.privacy_engine_make_private.side_effect = None
        self.privacy_engine_make_private.reset_mock()
        self.dpl.before_training(model, opt, loader)
        self.privacy_engine_make_private.assert_called_once_with(module=model,
                                                                 optimizer=opt,
                                                                 data_loader=loader,
                                                                 noise_multiplier=self.dp_args_l.get('sigma'),
                                                                 max_grad_norm=self.dp_args_l.get('clip'))

    def test_dep_controller_07_post_process_dp(self):
        """Tests before training method with different scenarios"""

        params = {"a_module.": torch.zeros([2, 4]), "b_module.": torch.zeros([2, 4])}

        # Post processes with DPL
        p = self.dpl._postprocess_dp(params)
        self.assertTrue('module' not in list(p.keys())[0], "`module tag is not properly removed from private "
                                                           "end-model`")

        # Post processes with DPC
        p = self.dpc._postprocess_dp(params)
        self.assertTrue('module' not in list(p.keys())[0], "`module tag is not properly removed from private "
                                                           "end-model`")

    @patch('fedbiomed.common.privacy.DPController._postprocess_dp')
    def test_dep_controller_08_after_training(self, postprocess):
        """Tests before training method with different scenarios"""

        postprocess.return_value = "POSTPROCESS"

        params = {"a_module.": torch.zeros([2, 4]), "b_module.": torch.zeros([2, 4])}

        # Post processes with DPL
        p = self.dpl.after_training(params)
        self.assertEqual(p, "POSTPROCESS")

        # Post processes with DPC
        p = self.dpc.after_training(params)
        self.assertEqual(p, "POSTPROCESS")
