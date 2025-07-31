import unittest
from dataclasses import dataclass

import fedbiomed.common.message as message
from fedbiomed.common.constants import (
    ErrorNumbers,
    TrainingPlanApprovalStatus,
)
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedMessageError

# we also want to test the decorator
from fedbiomed.common.message import (
    AdditiveSSharingReply,
    AdditiveSSharingRequest,
    AdditiveSSSetupReply,
    AdditiveSSSetupRequest,
    catch_dataclass_exception,
)


class TestMessage(unittest.TestCase):
    """
    Test the Message class
    """

    # before the tests
    def setUp(self):
        pass

    # after the tests
    def tearDown(self):
        pass

    #
    # helper function to check failures for all Message classes
    # ---------------------------------------------------------
    def check_class_args(self, cls, expected_result=True, **kwargs):
        result = True

        # list of permitted classes
        all_classes = [
            message.SearchReply,
            message.PingReply,
            message.TrainReply,
            message.Scalar,
            message.Log,
            message.ErrorMessage,
            message.ApprovalReply,
            message.SearchRequest,
            message.PingRequest,
            message.TrainRequest,
            message.ListReply,
            message.ListRequest,
            message.TrainingPlanStatusReply,
            message.TrainingPlanStatusRequest,
            message.ApprovalRequest,
        ]

        # test minimal python (only affectation) to insure
        # that the exception will be trapped only on object affectation
        try:
            valid_class = False
            for c in all_classes:
                if cls == c:
                    # print("DEBUG: detected class:", c)
                    m = c(**kwargs)
                    valid_class = True
                    break

            # the tester passed a bad class name to check_class_args()
            if not valid_class:
                self.fail("check_class_args: bad class name")

        except Exception:
            # print("===== " + str(e.__class__.__name__) + " trapped: " + str(e))
            result = False

        # decode all cases
        if expected_result is True and result is True:
            self.assertTrue(True, "check_class_args: good params detected")

        if expected_result is True and result is False:
            self.fail("check_class_args: good params detected as bad")

        if expected_result is False and result is True:
            self.fail("check_class_args: bad params detected as good")

        if expected_result is False and result is False:
            self.assertTrue(True, "check_class_args: bad params correclty detected")

        pass

    #
    # create a Message class and the decorator to test the raised exception
    # test also the @catch_dataclass_exception dcorator
    #
    @catch_dataclass_exception
    @dataclass
    class DummyMessage(message.Message):
        """
        dummy class to fully test the Message class
        """

        a: int
        b: str

    def test_message_additive_secret_sharing(self):
        AdditiveSSharingRequest(
            **{
                "node_id": "1234",
                "dest_node_id": "dataset_node_1234",
                "secagg_id": "secagg_id_1234",
            }
        )

        AdditiveSSharingReply(
            **{
                "node_id": "1234",
                "dest_node_id": "dataset_node_1234",
                "secagg_id": "secagg_id_1234",
                "share": 12,
            }
        )

        AdditiveSSSetupRequest(
            **{
                "researcher_id": "researcher_1234",
                "secagg_id": "secagg_1234",
                "element": 0,
                "experiment_id": "exp",
                "parties": ["12"],
            }
        )

        AdditiveSSSetupReply(
            **{
                "success": True,
                "researcher_id": "researcher_1234",
                "secagg_id": "secagg_1234",
                "node_id": "node_id_1234",
                "node_name": "node_name_1234",
                "msg": "test",
                "share": 111,
            }
        )

    def test_message_01_dummy(self):
        m0 = self.DummyMessage(1, "test")

        # getter test
        self.assertEqual(m0.get_param("a"), 1)
        self.assertEqual(m0.get_param("b"), "test")

        # test the validate fonction which sends an exception
        # bad parameter type for a
        bad_result = False
        try:
            m1 = self.DummyMessage(a="oh your god!", b="oh my god!")
        except FedbiomedMessageError:
            # we must arrive here, because message is malformed
            bad_result = True
        except Exception as e:
            # we should not arrive here also
            self.assertTrue(
                False,
                "bad exception caught: "
                + e.__class__.__name__
                + " instead of FedbiomedMessageError",
            )

        self.assertTrue(bad_result, "dummyMessage: bad params not detected")

        # bad params number
        bad_result = False
        try:
            m2 = self.DummyMessage(1, "foobar", False)

        except FedbiomedMessageError:
            #
            # we must arrive here, because message is malformed
            bad_result = True

        except Exception as e:
            #
            # @dataclass raises TypeError which is renamed
            # by @catch_dataclass_exception
            #
            # !! we should not reach this part of the code !!
            #
            self.assertTrue(False, "bad exception caught: " + e.__class__.__name__)

        self.assertTrue(bad_result, "dummyMessage: bad param number not detected")

        pass

    def test_message_to_dict_from_dict(self):
        msg = message.PingRequest(researcher_id="r1")

        t_msg = msg.to_dict()
        self.assertTrue("__type_message__" in t_msg)
        self.assertTrue("class" in t_msg["__type_message__"])
        self.assertTrue("module" in t_msg["__type_message__"])

        msg = message.Message.from_dict(t_msg)
        self.assertIsInstance(msg, message.PingRequest)

        with self.assertRaises(FedbiomedError):
            t_msg.pop("__type_message__")
            message.Message.from_dict(t_msg)

        with self.assertRaises(FedbiomedError):
            t = msg.to_dict()
            t["__type_message__"].pop("class")
            message.Message.from_dict(t)

        with self.assertRaises(FedbiomedError):
            t = msg.to_dict()
            t["__type_message__"]["class"] = "logger"
            message.Message.from_dict(t)

        with self.assertRaises(FedbiomedError):
            t = msg.to_dict()
            t["__type_message__"]["class"] = "FieldDescriptor"
            message.Message.from_dict(t)

        with self.assertRaises(FedbiomedError):
            t = msg.to_dict()
            t["__type_message__"]["class"] = "Unkown"
            message.Message.from_dict(t)

    def test_message_02_searchreply(self):
        # verify necessary arguments of all message creation

        # well formatted message
        self.check_class_args(
            message.SearchReply,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
            databases=[1, 2, 3],
            count=666,
            node_id="titi",
            node_name="node_titi",
        )

        # all these test should fail (not enough arguments)
        self.check_class_args(
            message.SearchReply, expected_result=False, researcher_id="toto"
        )

        self.check_class_args(message.SearchReply, expected_result=False, count=666)

        self.check_class_args(
            message.SearchReply, expected_result=False, databases=[1, 2, 3]
        )

        self.check_class_args(
            message.SearchReply, expected_result=False, node_id="toto"
        )

        self.check_class_args(message.SearchReply, expected_result=False)

        # too much arguments
        self.check_class_args(
            message.SearchReply,
            expected_result=False,
            researcher_id="toto",
            databases=[1, 2, 3],
            count=666,
            node_id="titi",
            extra_arg="not_allowed",
        )

        # all the following should be bad (bad argument type)
        self.check_class_args(
            message.SearchReply,
            expected_result=False,
            researcher_id="toto",
            protocol_version="99.99",
            databases=[1, 2, 3],
            count="not_an_integer",
            node_id="titi",
        )

        self.check_class_args(
            message.SearchReply,
            expected_result=False,
            researcher_id=True,
            protocol_version="99.99",
            databases=[1, 2, 3],
            count=666,
            node_id="titi",
        )

        self.check_class_args(
            message.SearchReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            databases=[1, 2, 3],
            count=666,
            node_id=True,
        )

        self.check_class_args(
            message.SearchReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            databases="not a list",
            count=666,
            node_id="titi",
        )

        self.check_class_args(
            message.SearchReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            success="not_a_boolean",
            databases=[],
            count=666,
            node_id="titi",
        )

        pass

    def test_message_03_pingreply(self):
        # verify necessary arguments of all message creation

        # well formatted message
        self.check_class_args(
            message.PingReply,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            node_name="node_titi",
        )

        self.check_class_args(message.PingReply, expected_result=False, node_id="titi")

        self.check_class_args(message.PingReply, expected_result=False, success=False)

        self.check_class_args(
            message.PingReply,
            expected_result=False,
            researcher_id="toto",
            node_id="titi",
            success=True,
            extra_arg="foobar",
        )

        # bad argument type
        self.check_class_args(
            message.PingReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=True,
            node_id="titi",
            success=True,
        )

        self.check_class_args(
            message.PingReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            node_id=True,
            success=True,
        )

        self.check_class_args(
            message.PingReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            success="not_a_bool",
        )

        self.check_class_args(
            message.PingReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            success="not_a_bool",
        )

        self.check_class_args(
            message.PingReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            success=True,
        )

        pass

    def test_message_04_trainreply(self):
        # well formatted message
        self.check_class_args(
            message.TrainReply,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment",
            state_id="state_id_1234",
            success=True,
            node_id="titi",
            node_name="node_titi",
            dataset_id="my_data",
            params={"x": 0},
            timing={"t0": 0.0, "t1": 1.0},
            sample_size=123,
            msg="message_in_a_bottle",
        )

        # bad param number
        self.check_class_args(
            message.TrainReply, expected_result=False, researcher_id="toto"
        )

        self.check_class_args(
            message.TrainReply, expected_result=False, experiment_id="experiment"
        )

        self.check_class_args(message.TrainReply, expected_result=False, success=True)

        self.check_class_args(message.TrainReply, expected_result=False, node_id="titi")

        self.check_class_args(
            message.TrainReply, expected_result=False, dataset_id="my_data"
        )

        self.check_class_args(
            message.TrainReply, expected_result=False, params={"x": 0}
        )

        self.check_class_args(
            message.TrainReply, expected_result=False, params={"x": 0}
        )

        self.check_class_args(
            message.TrainReply, expected_result=False, timing={"t0": 0.0, "t1": 1.0}
        )

        self.check_class_args(
            message.TrainReply, expected_result=False, msg="message_in_a_bottle"
        )

        self.check_class_args(
            message.TrainReply,
            expected_result=False,
            researcher_id="toto",
            experiment_id="experiment",
            success=True,
            node_id="titi",
            dataset_id="my_data",
            params={"x": 0},
            timing={"t0": 0.0, "t1": 1.0},
            msg="message_in_a_bottle",
            sample_size=None,
            extra_param="dont_know_what_to_do_with_you",
        )

        # bad param type
        self.check_class_args(
            message.TrainReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=True,
            experiment_id="experiment",
            success=True,
            node_id="titi",
            dataset_id="my_data",
            params={"x": 0},
            timing={"t0": 0.0, "t1": 1.0},
            msg="message_in_a_bottle",
        )

        self.check_class_args(
            message.TrainReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id=True,
            success=True,
            node_id="titi",
            dataset_id="my_data",
            params={"x": 0},
            timing={"t0": 0.0, "t1": 1.0},
            msg="message_in_a_bottle",
        )

        self.check_class_args(
            message.TrainReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment",
            success="not_a_bool",
            node_id="titi",
            dataset_id="my_data",
            params={"x": 0},
            timing={"t0": 0.0, "t1": 1.0},
            msg="message_in_a_bottle",
        )

        self.check_class_args(
            message.TrainReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment",
            success=True,
            node_id=True,
            dataset_id="my_data",
            params={"x": 0},
            timing={"t0": 0.0, "t1": 1.0},
            msg="message_in_a_bottle",
        )

        self.check_class_args(
            message.TrainReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment",
            success=True,
            node_id="titi",
            dataset_id=True,
            params={"x": 0},
            timing={"t0": 0.0, "t1": 1.0},
            msg="message_in_a_bottle",
        )

        self.check_class_args(
            message.TrainReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment",
            success=True,
            node_id="titi",
            dataset_id="my_data",
            params_url=True,
            timing={"t0": 0.0, "t1": 1.0},
            msg="message_in_a_bottle",
        )

        self.check_class_args(
            message.TrainReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment",
            success=True,
            node_id="titi",
            dataset_id="my_data",
            params={"x": 0},
            timing="not_a_dict",
            msg="message_in_a_bottle",
        )

        self.check_class_args(
            message.TrainReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment",
            success=True,
            node_id="titi",
            dataset_id="my_data",
            params={"x": 0},
            timing={"t0": 0.0, "t1": 1.0},
            msg=True,
        )

    def test_message_05_listreply(self):
        # well formatted message
        self.check_class_args(
            message.ListReply,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
            success=True,
            databases=[1, 2, 3],
            count=666,
            node_id="titi",
            node_name="node_titi",
        )

        # all these test should fail (not enough arguments)
        self.check_class_args(
            message.ListReply, expected_result=False, researcher_id="toto"
        )

        self.check_class_args(message.ListReply, expected_result=False, count=666)

        self.check_class_args(message.ListReply, expected_result=False, success=True)

        self.check_class_args(
            message.ListReply, expected_result=False, databases=[1, 2, 3]
        )

        self.check_class_args(message.ListReply, expected_result=False, node_id="toto")

        # too much arguments
        self.check_class_args(
            message.ListReply,
            expected_result=False,
            researcher_id="toto",
            success=True,
            databases=[1, 2, 3],
            count=666,
            node_id="titi",
            extra_arg="not_allowed",
        )

        # all the following should be bad (bad argument type)
        self.check_class_args(
            message.ListReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            success=True,
            databases=[1, 2, 3],
            count="not_an_integer",
            node_id="titi",
        )

        self.check_class_args(
            message.ListReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=True,
            success=True,
            databases=[1, 2, 3],
            count=666,
            node_id="titi",
        )

        self.check_class_args(
            message.ListReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            success=True,
            databases=[1, 2, 3],
            count=666,
            node_id=True,
        )

        self.check_class_args(
            message.ListReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            success=True,
            databases="not a list",
            count=666,
            node_id="titi",
        )

        self.check_class_args(
            message.ListReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            success="not_a_boolean",
            databases=[],
            count=666,
            node_id="titi",
        )

    def test_message_06_addscalarreply(self):
        # well formatted message

        self.check_class_args(
            message.Scalar,
            expected_result=True,
            node_id="titi",
            node_name="node_titi",
            experiment_id="tutu",
            train=True,
            test=True,
            test_on_local_updates=True,
            test_on_global_updates=True,
            metric={"x": 12},
            iteration=666,
            epoch=12,
            total_samples=12,
            batch_samples=12,
            num_batches=12,
            num_samples_trained=12,
        )

        # bad param number
        self.check_class_args(
            message.Scalar, expected_result=False, num_samples_trained=12
        )

        self.check_class_args(
            message.Scalar,
            expected_result=False,
            node_id="titi",
            experiment_id="tutu",
            iteration=666,
            extra_arg="???",
        )

        # bad param type
        self.check_class_args(
            message.Scalar,
            expected_result=False,
            node_id=12,
            experiment_id="tutu",
            train=True,
            test=True,
            test_on_local_updates=True,
            test_on_global_updates=True,
            metric={"x": 12},
            iteration=666,
            epoch=12,
            total_samples=12,
            batch_samples=12,
            num_batches=12,
        )

        pass

    def test_message_07_modelstatusreply(self):
        self.check_class_args(
            message.TrainingPlanStatusReply,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            node_name="node_titi",
            experiment_id="titi",
            success=True,
            approval_obligation=True,
            status=TrainingPlanApprovalStatus.APPROVED.value,
            msg="sdrt",
            training_plan="TP",
            training_plan_id="id-1234",
        )

        self.check_class_args(
            message.TrainingPlanStatusReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            node_id=12334,
            experiment_id="titi",
            success=True,
            approval_obligation=True,
            status=TrainingPlanApprovalStatus.REJECTED.value,
            msg="sdrt",
            training_plan="TP",
        )

        self.check_class_args(
            message.TrainingPlanStatusReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=12344,
            node_id="12334",
            experiment_id="titi",
            success=True,
            approval_obligation=True,
            status=TrainingPlanApprovalStatus.PENDING.value,
            msg="sdrt",
            training_plan="TP",
        )

        self.check_class_args(
            message.TrainingPlanStatusReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="12344",
            node_id="12334",
            experiment_id="titi",
            success=True,
            approval_obligation=True,
            status=True,
            msg="sdrt",
            training_plan="TP",
        )

        self.check_class_args(
            message.TrainingPlanStatusReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="12344",
            node_id="12334",
            experiment_id="titi",
            success=True,
            approval_obligation="True",
            status="None",
            msg="sdrt",
            training_plan="TP",
        )

        self.check_class_args(
            message.TrainingPlanStatusReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=333,
            node_id=1212,
            experiment_id=False,
            success="not a bool",
            approval_obligation=True,
            status=TrainingPlanApprovalStatus.PENDING.value,
            msg="sdrt",
            training_plan_url=123123,
        )

        self.check_class_args(
            message.TrainingPlanStatusReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=333,
            node_id=1212,
            experiment_id=False,
            success="not a bool",
            approval_obligation=True,
            status=TrainingPlanApprovalStatus.REJECTED.value,
            msg="sdrt",
        )

    def test_message_08_log(self):
        # well formatted message
        self.check_class_args(
            message.Log,
            expected_result=True,
            node_id="titi",
            level="INFO",
            msg="this is an error message",
        )

        # bad param number
        self.check_class_args(message.Log, expected_result=False, researcher_id="toto")

        self.check_class_args(message.Log, expected_result=False, node_id="titi")

        self.check_class_args(message.Log, expected_result=False, level="INFO")

        self.check_class_args(
            message.Log, expected_result=False, msg="this is an error message"
        )

        self.check_class_args(
            message.Log,
            expected_result=False,
            node_id="titi",
            level="INFO",
            msg="this is an error message",
            extra_arg="???",
        )

        self.check_class_args(
            message.Log,
            expected_result=False,
            node_id=False,
            level="INFO",
            msg="this is an error message",
        )

        self.check_class_args(
            message.Log,
            expected_result=False,
            node_id="titi",
            level="INFO",
            msg=[1, 2],
        )

        self.check_class_args(
            message.Log,
            expected_result=False,
            node_id="titi",
            level="INFO",
            msg=[1, 2],
        )

        pass

    def test_message_09_error(self):
        # well formatted message
        self.check_class_args(
            message.ErrorMessage,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            node_name="node_titi",
            errnum=ErrorNumbers.FB100.value,
            extra_msg="this is an error message",
        )

        # bad param number
        self.check_class_args(
            message.ErrorMessage, expected_result=False, researcher_id="toto"
        )

        self.check_class_args(
            message.ErrorMessage, expected_result=False, node_id="titi"
        )

        self.check_class_args(
            message.ErrorMessage, expected_result=False, errnum=ErrorNumbers.FB100.value
        )

        self.check_class_args(
            message.ErrorMessage,
            expected_result=False,
            extra_msg="this is an error message",
        )

        self.check_class_args(
            message.ErrorMessage,
            expected_result=False,
        )

        self.check_class_args(
            message.ErrorMessage,
            expected_result=False,
            researcher_id="toto",
            node_id="titi",
            errnum=ErrorNumbers.FB100.value,
            extra_msg="this is an error message",
            extra_arg="???",
        )

        # bad param type
        self.check_class_args(
            message.ErrorMessage,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=False,
            node_id="titi",
            errnum=ErrorNumbers.FB100.value,
            extra_msg="this is an error message",
        )

        self.check_class_args(
            message.ErrorMessage,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            node_id=False,
            errnum=ErrorNumbers.FB100.value,
            extra_msg="this is an error message",
        )

        self.check_class_args(
            message.ErrorMessage,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            errnum=False,
            extra_msg="this is an error message",
        )

        pass

    def test_message_10_searchrequest(self):
        # well formatted message
        self.check_class_args(
            message.SearchRequest, expected_result=False, researcher_id="toto"
        )

        # bad param number
        self.check_class_args(
            message.SearchRequest,
            expected_result=False,
            protocol_version="99.99",
            tags=["data", "doto"],
        )

        self.check_class_args(
            message.SearchRequest,
            expected_result=False,
            protocol_version="99.99",
        )

        self.check_class_args(
            message.SearchRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            tags=["data", "doto"],
            extra_args="???",
        )

        # bad param type
        self.check_class_args(
            message.SearchRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=False,
            tags=["data", "doto"],
        )

        self.check_class_args(
            message.SearchRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            tags="not_a_list",
        )

    def test_message_11_pingrequest(self):
        # well formatted message
        self.check_class_args(
            message.PingRequest,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
        )

        self.check_class_args(
            message.PingRequest,
            expected_result=False,
        )

        self.check_class_args(
            message.PingRequest,
            expected_result=False,
            researcher_id="toto",
            extra_arg="???",
        )

        # bad param type
        self.check_class_args(
            message.PingRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=False,
        )

    def test_message_12_trainrequest(self):
        # well formatted message
        self.check_class_args(
            message.TrainRequest,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment_number",
            params={"x": 0},
            state_id="state_id_1234",
            training_args={"a": 1, "b": 2},
            dataset_id="MNIST",
            training=True,
            model_args={"c": 3, "d": 4},
            training_plan="tp",
            training_plan_class="my_model",
            secagg_arguments={
                "secagg_servkey_id": None,
                "secagg_random": None,
                "secagg_clipping_range": None,
            },
            round=1,
            aggregator_args={"aggregator_name": "fedavg"},
            optim_aux_var=None,
        )

        # bad param number
        self.check_class_args(
            message.TrainRequest, expected_result=False, researcher_id="toto"
        )

        self.check_class_args(
            message.TrainRequest,
            expected_result=False,
            experiment_id="experiment_number",
        )

        self.check_class_args(
            message.TrainRequest, expected_result=False, params={"x": 0}
        )

        self.check_class_args(
            message.TrainRequest, expected_result=False, training_args={"a": 1, "b": 2}
        )

        self.check_class_args(
            message.TrainRequest, expected_result=False, dataset_id="MNIS"
        )

        self.check_class_args(
            message.TrainRequest, expected_result=False, model_args={"c": 3, "d": 4}
        )

        self.check_class_args(
            message.TrainRequest, expected_result=False, training_plan="xxxx"
        )

        self.check_class_args(
            message.TrainRequest, expected_result=False, training_plan_class="my_model"
        )

        self.check_class_args(
            message.TrainRequest,
            expected_result=False,
        )

        self.check_class_args(
            message.TrainRequest, expected_result=False, secagg_clipping_range="non-int"
        )

        self.check_class_args(
            message.TrainRequest,
            expected_result=False,
            researcher_id="toto",
            experiment_id="experiment_number",
            params={"x": 0},
            training_args={"a": 1, "b": 2},
            dataset_id="MNIS",
            training=False,
            model_args={"c": 3, "d": 4},
            training_plan="TP",
            training_plan_class="my_model",
            extra_arg="???",
        )

        # bad param type
        self.check_class_args(
            message.TrainRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=False,
            experiment_id="experiment_number",
            params={"x": 0},
            training_args={"a": 1, "b": 2},
            dataset_id="MNIS",
            training=False,
            model_args={"c": 3, "d": 4},
            training_plan="TP",
            training_plan_class="my_model",
        )

        self.check_class_args(
            message.TrainRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment_number",
            params_url=False,
            training_args={"a": 1, "b": 2},
            dataset_id="MNIS",
            training=False,
            secagg_random=None,
            model_args={"c": 3, "d": 4},
            training_plan_url="http://dev.null",
            training_plan_class="my_model",
        )

        self.check_class_args(
            message.TrainRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment_number",
            params={"x": 0},
            training_args={"foo": "not_a_str"},
            dataset_id="MNIS",
            training=False,
            model_args={"c": 3, "d": 4},
            training_plan="TP",
            training_plan_class="my_model",
        )

        self.check_class_args(
            message.TrainRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment_number",
            params={"x": 0},
            training_args={"a": 1, "b": 2},
            dataset_id={"foo": "not_a_str"},
            training=False,
            secagg_random=None,
            secagg_clipping_range=None,
            model_args={"c": 3, "d": 4},
            training_plan="TP",
            training_plan_class="my_model",
        )

        self.check_class_args(
            message.TrainRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment_number",
            params={"x": 0},
            training_args={"a": 1, "b": 2},
            dataset_id="MNIS",
            training=False,
            model_args="not_a_dict",
            training_plan="TP",
            training_plan_class="my_model",
        )

        self.check_class_args(
            message.TrainRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment_number",
            params={"x": 0},
            training_args={"a": 1, "b": 2},
            dataset_id="MNIS",
            training=False,
            secagg_clipping_range=None,
            model_args={"c": 3, "d": 4},
            training_plan=False,
            training_plan_class="my_model",
        )

        self.check_class_args(
            message.TrainRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="experiment_number",
            params={"x": 0},
            training_args={"a": 1, "b": 2},
            training_data="MNIS",
            training=False,
            secagg_clipping_range=None,
            model_args={"c": 3, "d": 4},
            training_plan="TP",
            training_plan_class="my_model",
        )

        pass

    def test_message_13_listrequest(self):
        # well formatted message
        self.check_class_args(
            message.ListRequest,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
        )

        # bad param number
        self.check_class_args(
            message.ListRequest, expected_result=False, tags=["data", "doto"]
        )

        self.check_class_args(
            message.ListRequest,
            expected_result=False,
        )

        self.check_class_args(
            message.ListRequest,
            expected_result=False,
            researcher_id="toto",
            tags=["data", "doto"],
            extra_args="???",
        )

        # bad param type
        self.check_class_args(
            message.ListRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=False,
            tags=["data", "doto"],
        )

        # bad param type
        self.check_class_args(
            message.ListRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=False,
        )

        pass

    def test_message_14_modelstatusrequest(self):
        self.check_class_args(
            message.TrainingPlanStatusRequest,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="sdsd",
            training_plan="TP",
        )

        self.check_class_args(
            message.TrainingPlanStatusRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=True,
            experiment_id="sdsd",
            training_plan_url="do_it",
        )

        self.check_class_args(
            message.TrainingPlanStatusRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id=122323,
            training_plan_url="do_it",
        )

        self.check_class_args(
            message.TrainingPlanStatusRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            experiment_id="sdsd",
            training_plan_url=12323,
        )

        self.check_class_args(
            message.TrainingPlanStatusRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="ttot",
            experiment_id="sdsd",
            training_plan_url="do_it",
        )

    def test_message_25_approval_request(self):
        """Test the approval request message fabrication/validation"""

        # well formatted message
        self.check_class_args(
            message.ApprovalRequest,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
            description="this is a description string",
            training_plan="TP",
        )

        # all these test should fail (bad number of args arguments or bad type)
        self.check_class_args(
            message.ApprovalRequest,
            expected_result=False,
            researcher_id="toto",
        )

        self.check_class_args(
            message.ApprovalRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            description="this is a description string",
            training_plan="TP",
            unknown_extra_arg="whatever",
        )

        self.check_class_args(
            message.ApprovalRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=False,
            description="this is a description string",
            training_plan_url="http://dev.null",
        )

        self.check_class_args(
            message.ApprovalRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            description=False,
            training_plan_url="http://dev.null",
        )

        self.check_class_args(
            message.ApprovalRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            description="this is a description string",
            training_plan_url=False,
        )

        self.check_class_args(
            message.ApprovalRequest,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            description="this is a description string",
            training_plan_url="http://dev.null",
        )

    def test_message_26_approval_reply(self):
        """Test the approval reply message fabrication/validation"""
        # well formatted message
        self.check_class_args(
            message.ApprovalReply,
            expected_result=True,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            node_name="node_titi",
            message="xxx",
            training_plan_id="id-xxx",
            status=200,
            success=True,
        )

        # all these test should fail (bad number of args arguments or bad type)
        self.check_class_args(
            message.ApprovalReply,
            expected_result=False,
            researcher_id="toto",
            message="xxx",
        )

        self.check_class_args(
            message.ApprovalReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            status=200,
            success=True,
            message="xxx",
            extra_arg="this will break",
        )

        self.check_class_args(
            message.ApprovalReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id=False,
            message="xxx",
            node_id="titi",
            status=200,
            success=True,
        )

        self.check_class_args(
            message.ApprovalReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            message="xxx",
            node_id=False,
            status=200,
            success=True,
        )

        self.check_class_args(
            message.ApprovalReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            message="xxx",
            status="not an int",
            success=True,
        )

        self.check_class_args(
            message.ApprovalReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            message="xxx",
            node_id="titi",
            status=200,
            success=True,
        )

        self.check_class_args(
            message.ApprovalReply,
            expected_result=False,
            protocol_version="99.99",
            researcher_id="toto",
            node_id="titi",
            message="xxx",
            status=200,
            success="not a bool",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
