import json
import logging
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock

from fedbiomed.common.logger import (
    DEFAULT_LOG_LEVEL,
    SECURITY_CONTEXT,
    _SecurityFormatter,
    _SecurityOnlyFilter,
    logger,
)


class TestLogger(unittest.TestCase):
    """
    Test the Logger class
    """

    def setUp(self):
        """
        before all tests: put the loglevel to a known state
        """
        logger.setLevel(DEFAULT_LOG_LEVEL)
        pass

    def tearDown(self):
        """
        after all test... empty for now
        """
        pass

    def test_logger_00_internal_translator(self):
        """
        as test name says.... string -> logging.* translation
        """

        # string -> logging.* translator
        self.assertEqual(logger._internal_level_translator("DEBUG"), logging.DEBUG)
        self.assertEqual(logger._internal_level_translator("INFO"), logging.INFO)
        self.assertEqual(logger._internal_level_translator("WARNING"), logging.WARNING)
        self.assertEqual(logger._internal_level_translator("ERROR"), logging.ERROR)
        self.assertEqual(
            logger._internal_level_translator("CRITICAL"), logging.CRITICAL
        )

        self.assertEqual(
            logger._internal_level_translator("STUPIDO"), DEFAULT_LOG_LEVEL
        )
        pass

    def test_logger_01_internal_addhandler(self):
        """
        ulgy usage of logger._handlers (this is internal data),
        but no getter for this internal data is/should_be provided

        subject to change if internal design of logger.py is changed
        """

        # handler manager test1
        handler = logging.NullHandler()

        # console handler initialized by the constructor
        self.assertEqual(len(logger._handlers), 1)

        # should also return len() = 1
        logger._internal_add_handler("CONSOLE", handler)
        self.assertEqual(len(logger._handlers), 1)

        # add a new one
        logger._internal_add_handler("TEST1", handler)
        self.assertEqual(len(logger._handlers), 2)

        # handler already exists -> count does not change
        logger._internal_add_handler("TEST1", handler)
        self.assertEqual(len(logger._handlers), 2)

        pass

    def test_logger_02_getattr(self):
        """
        test the __getattr__
        """

        # import the original logger class
        orig_logger = logging.getLogger("root")
        orig_logger.setLevel(logging.DEBUG)

        # method from the original Logger class
        orig_level = orig_logger.getEffectiveLevel()

        # this one uses our logger and the __getattr__ method
        logger.setLevel("DEBUG")
        over_level = logger.getEffectiveLevel()

        self.assertEqual(orig_level, over_level)

        # try that this fails
        try:
            logger.this_method_does_not_exists()
            self.fail("calling this_method_does_not_exists()")
        except AttributeError:
            self.assertTrue(True, "calling this_method_does_not_exists() detected")

        pass

    def test_logger_03_singleton(self):
        """
        test singleton mechanism
        """

        from fedbiomed.common.logger import FedLogger

        second_logger = FedLogger()

        self.assertEqual(logger, second_logger)

        pass

    def test_logger_04_setlevel(self):
        """
        as test name says.... test the setLevel() method
        """

        # initial DEFAULT_LOG_LEVEL
        self.assertEqual(logger.getEffectiveLevel(), DEFAULT_LOG_LEVEL)

        # check setLevel
        for lvl in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]:
            logger.setLevel(lvl)
            self.assertEqual(logger.getEffectiveLevel(), lvl)

        # bounds
        logger.setLevel(10000)
        self.assertEqual(logger.getEffectiveLevel(), DEFAULT_LOG_LEVEL)

        logger.setLevel(-1)
        self.assertEqual(logger.getEffectiveLevel(), DEFAULT_LOG_LEVEL)

        # modify the default level of existing handlers
        logger.setLevel("CRITICAL")
        self.assertEqual(logger.getEffectiveLevel(), logging.CRITICAL)

        self.assertEqual(logger._handlers["CONSOLE"].level, logging.CRITICAL)

        # add a new handler and verify its initial level
        handler = logging.NullHandler()

        logger._internal_add_handler("H_1", handler)
        self.assertEqual(logger._handlers["H_1"].level, DEFAULT_LOG_LEVEL)

        # change again to critical
        logger.setLevel("CRITICAL", "H_1")
        self.assertEqual(logger._handlers["H_1"].level, logging.CRITICAL)

        # change all levels to DEBUG
        logger.setLevel("DEBUG")
        self.assertEqual(logger._handlers["CONSOLE"].level, logging.DEBUG)
        self.assertEqual(logger._handlers["H_1"].level, logging.DEBUG)

        # not initialized handler
        with self.assertLogs("fedbiomed", logging.WARNING) as captured:
            logger.setLevel("DEBUG", "NOT_INITIALIZED_HANDLER")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(
            captured.records[0].getMessage(),
            "NOT_INITIALIZED_HANDLER handler not initialized yet",
        )
        pass

    def test_logger_05_logging(self):
        """
        as test name says.... test that logging.* levels
        """

        # test debug() - string
        with self.assertLogs("fedbiomed", logging.DEBUG) as captured:
            logger.debug("TEST_1")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_1")

        # test info() - string
        with self.assertLogs("fedbiomed", logging.INFO) as captured:
            logger.info("TEST_2")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_2")

        # test warning() - string
        with self.assertLogs("fedbiomed", logging.WARNING) as captured:
            logger.warning("TEST_3")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_3")

        # test error() - string
        with self.assertLogs("fedbiomed", logging.ERROR) as captured:
            logger.error("TEST_4")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_4")

        # test critical() - string
        with self.assertLogs("fedbiomed", logging.CRITICAL) as captured:
            logger.critical("TEST_5")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_5")

        # test log() - string
        with self.assertLogs("fedbiomed", logging.CRITICAL) as captured:
            logger.log(logging.CRITICAL, "TEST_6")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_6")

        with self.assertLogs("fedbiomed", logging.CRITICAL) as captured:
            logger.log("CRITICAL", "TEST_7")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_7")

    # minimal on_* handlers
    def on_message(self, client, userdata, msg):
        """
        empty on_message handler
        """
        pass

    def on_connect(self, client, userdata, flags, rc):
        """
        empty on_connect handler
        """
        pass

    def on_disconnect(self, client, userdata, flags, rc):
        """
        empty on_disconnect handler
        """
        pass

    def test_logger_06_grpc_handler(self):
        """
        test grpc handler
        """

        grpc = MagicMock()

        #
        logger.add_grpc_handler(on_log=grpc.send, node_id="dummy-id")

        logger.debug("console DEBUG message", researcher_id="test-id")
        logger.error("console ERROR message")

        #
        logger.setLevel("DEBUG")
        logger.critical("verify that logger still works properly")

        pass

    def test_logger_07_filehandler(self):
        """
        test file handler
        """

        randomfile = tempfile.NamedTemporaryFile()

        logger.add_file_handler(filename=randomfile.name)
        logger.log("ERROR", "YYY-FIND_THIS_IN_TEMPFILE-XXX")

        # give some time to the logger
        time.sleep(2)

        # verify that the log appeared
        with open(randomfile.name) as f:
            lines = f.readlines()

        # lines[] should contain YYY-FIND_THIS_IN_TEMPFILE-XXX
        if "YYY-FIND_THIS_IN_TEMPFILE-XXX" not in lines[0]:
            self.fail("log message not detected")
        else:
            self.assertTrue(True, "log message detected")

        pass

    def test_logger_08_setprefix(self):
        """
        test setPrefix
        """

        prefix = "[TEST_PREFIX] "
        message = "TEST_MESSAGE"
        logger.setPrefix(prefix)

        with self.assertLogs("fedbiomed", logging.INFO) as captured:
            logger.info(message)

        self.assertEqual(len(captured.records), 1)

        # Handler formatters add the prefix at formatting time; assertLogs gives raw record.
        formatted = logger._handlers["CONSOLE"].formatter.format(captured.records[0])
        self.assertTrue(prefix in formatted, "prefix not found in log message")

        # reset prefix
        logger.setPrefix("")

    def test_logger_09_security_only_filter(self):
        """Test _SecurityOnlyFilter allows only records with is_security=True"""
        filter = _SecurityOnlyFilter()

        # Create mock records
        record_with_flag = MagicMock()
        record_with_flag.is_security = True

        record_without_flag = MagicMock()
        delattr(record_without_flag, "is_security")

        record_false_flag = MagicMock()
        record_false_flag.is_security = False

        # Test filtering
        self.assertTrue(filter.filter(record_with_flag))
        self.assertFalse(filter.filter(record_without_flag))
        self.assertFalse(filter.filter(record_false_flag))

    def test_logger_10_configure_security(self):
        """Test configure_security() sets security defaults"""
        logger.configure_security(
            node_id="test_node_123",
            node_name="test_node_name",
            fedbiomed_version="v1.2.3",
        )

        self.assertEqual(logger._security_defaults["node_id"], "test_node_123")
        self.assertEqual(logger._security_defaults["node_name"], "test_node_name")
        self.assertEqual(logger._security_defaults["fedbiomed_version"], "v1.2.3")

        # Test partial updates
        logger.configure_security(node_id="new_node_456")
        self.assertEqual(logger._security_defaults["node_id"], "new_node_456")
        self.assertEqual(logger._security_defaults["node_name"], "test_node_name")

    def test_logger_11_security_context(self):
        """Test security_context() context manager"""
        # Ensure clean state
        SECURITY_CONTEXT.set(None)

        # Test basic context setting
        with logger.security_context(researcher_id="researcher_1", operation="test_op"):
            ctx = SECURITY_CONTEXT.get()
            self.assertEqual(ctx["researcher_id"], "researcher_1")
            self.assertEqual(ctx["operation"], "test_op")

        # Context should be reset after exiting
        ctx_after = SECURITY_CONTEXT.get()
        self.assertIsNone(ctx_after)

        # Test nested contexts
        with logger.security_context(
            researcher_id="researcher_1", operation="outer_op"
        ):
            ctx_outer = SECURITY_CONTEXT.get()
            self.assertEqual(ctx_outer["researcher_id"], "researcher_1")
            self.assertEqual(ctx_outer["operation"], "outer_op")

            # Override researcher_id in nested context
            with logger.security_context(
                researcher_id="researcher_2", operation="inner_op"
            ):
                ctx_inner = SECURITY_CONTEXT.get()
                self.assertEqual(ctx_inner["researcher_id"], "researcher_2")
                self.assertEqual(ctx_inner["operation"], "inner_op")

            # Should revert to outer context values
            ctx_reverted = SECURITY_CONTEXT.get()
            self.assertEqual(ctx_reverted["researcher_id"], "researcher_1")
            self.assertEqual(ctx_reverted["operation"], "outer_op")

    def test_logger_12_security_formatter_json_passthrough(self):
        """Test _SecurityFormatter passes through existing JSON"""
        security_defaults = {
            "node_id": "node_123",
            "node_name": "node_test",
            "fedbiomed_version": "v1.0",
        }
        formatter = _SecurityFormatter(security_defaults)

        # Create a mock record with JSON message
        record = MagicMock()
        json_msg = json.dumps({"test": "data", "value": 123})
        record.getMessage.return_value = json_msg

        # Should return JSON as-is
        result = formatter.format(record)
        self.assertEqual(result, json_msg)
        self.assertEqual(json.loads(result), {"test": "data", "value": 123})

    def test_logger_13_security_event_with_caller_info(self):
        """Test security_event() captures caller information"""
        logger.configure_security(
            node_id="test_node", node_name="test_name", fedbiomed_version="v1.0"
        )
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log") as f:
            temp_file = f.name

        try:
            logger.add_security_file_handler(filename=temp_file)
            logger.security_event(
                operation="test_operation",
                status="success",
                researcher_id="researcher_1",
                custom_field="custom_value",
            )
            time.sleep(0.1)

            with open(temp_file, "r") as f:
                log_entry = json.loads(f.readline())

            self.assertEqual(log_entry["operation"], "test_operation")
            self.assertEqual(log_entry["status"], "success")
            self.assertEqual(log_entry["researcher_id"], "researcher_1")
            self.assertEqual(log_entry["custom_field"], "custom_value")
            self.assertIn("caller_function", log_entry)
            self.assertIn("caller_module", log_entry)
            self.assertIn("caller_file", log_entry)
            self.assertIn("caller_line", log_entry)

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_logger_14_is_security_flag_behavior(self):
        """Test is_security flag controls writing to security log"""
        logger.configure_security(
            node_id="test_node", node_name="test_name", fedbiomed_version="v1.0"
        )
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log") as f:
            temp_file = f.name

        try:
            logger.add_security_file_handler(filename=temp_file)

            # Without flag - should not write
            logger.info("Regular message")
            logger.error("Regular error")
            time.sleep(0.1)
            with open(temp_file, "r") as f:
                self.assertEqual(len(f.readlines()), 0)

            # With False flag - should not write
            logger.info("False flag", extra={"is_security": False})
            time.sleep(0.1)
            with open(temp_file, "r") as f:
                self.assertEqual(len(f.readlines()), 0)

            # With True flag - should write
            logger.info("Info msg", extra={"is_security": True, "operation": "test_op"})
            logger.error(
                "Error msg",
                extra={
                    "is_security": True,
                    "operation": "error_sent",
                    "error_code": "FB300",
                },
                researcher_id="researcher_1",
            )
            time.sleep(0.1)

            with open(temp_file, "r") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)

            info_entry = json.loads(lines[0])
            self.assertEqual(info_entry["message"], "Info msg")
            self.assertEqual(info_entry["level"], "INFO")
            self.assertEqual(info_entry["operation"], "test_op")

            error_entry = json.loads(lines[1])
            self.assertEqual(error_entry["message"], "Error msg")
            self.assertEqual(error_entry["level"], "ERROR")
            self.assertEqual(error_entry["error_code"], "FB300")
            self.assertEqual(error_entry["researcher_id"], "researcher_1")

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_logger_15_security_context_merges_with_event(self):
        """Test security_context merges with security_event fields"""
        logger.configure_security(
            node_id="test_node", node_name="test_name", fedbiomed_version="v1.0"
        )
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log") as f:
            temp_file = f.name

        try:
            logger.add_security_file_handler(filename=temp_file)
            with logger.security_context(
                researcher_id="researcher_1", experiment_id="exp_123"
            ):
                logger.security_event(
                    operation="training_start", status="initiated", round_number=1
                )

            time.sleep(0.1)
            with open(temp_file, "r") as f:
                log_entry = json.loads(f.readline())

            self.assertEqual(log_entry["researcher_id"], "researcher_1")
            self.assertEqual(log_entry["experiment_id"], "exp_123")
            self.assertEqual(log_entry["operation"], "training_start")
            self.assertEqual(log_entry["round_number"], 1)

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_logger_16_add_security_file_handler_rotation(self):
        """Test daily rotation creates new file when day passes"""
        logger.configure_security(
            node_id="test_node", node_name="test_name", fedbiomed_version="v1.0"
        )

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log") as f:
            temp_file = f.name

        try:
            # Add security file handler with real TimedRotatingFileHandler
            logger.add_security_file_handler(filename=temp_file)

            # Write first log entry
            logger.security_event(
                operation="test_operation_1",
                status="success",
                researcher_id="researcher_1",
            )

            time.sleep(0.1)

            # Verify first entry exists
            with open(temp_file, "r") as f:
                lines_before = f.readlines()

            self.assertEqual(len(lines_before), 1)

            # Get the handler and trigger rollover manually (simulates midnight passing)
            security_handler = logger._handlers.get("SECURITY_FILE")
            self.assertIsNotNone(security_handler, "SECURITY_FILE handler should exist")

            # Trigger rollover (simulates day change)
            security_handler.doRollover()

            # Write second log entry after "day change"
            logger.security_event(
                operation="test_operation_2",
                status="success",
                researcher_id="researcher_2",
            )

            time.sleep(0.1)

            # Verify current file now has only the new entry
            with open(temp_file, "r") as f:
                lines_after = f.readlines()

            self.assertEqual(
                len(lines_after),
                1,
                "Current log file should have only the new entry after rollover",
            )
            log_entry = json.loads(lines_after[0])
            self.assertEqual(log_entry["operation"], "test_operation_2")

            # Verify backup file was created (with timestamp suffix)
            # TimedRotatingFileHandler adds suffix like .2026-02-06
            backup_files = [
                f
                for f in os.listdir(os.path.dirname(temp_file))
                if f.startswith(os.path.basename(temp_file))
                and f != os.path.basename(temp_file)
            ]

            self.assertGreater(
                len(backup_files),
                0,
                "Backup file should be created after rollover",
            )

            # Verify old entry is in backup file
            backup_file = os.path.join(os.path.dirname(temp_file), backup_files[0])
            with open(backup_file, "r") as f:
                backup_lines = f.readlines()

            self.assertEqual(len(backup_lines), 1)
            backup_entry = json.loads(backup_lines[0])
            self.assertEqual(backup_entry["operation"], "test_operation_1")

        finally:
            # Clean up all files
            if os.path.exists(temp_file):
                os.remove(temp_file)
            # Clean up backup files
            if os.path.exists(os.path.dirname(temp_file)):
                for f in os.listdir(os.path.dirname(temp_file)):
                    if f.startswith(os.path.basename(temp_file)):
                        try:
                            os.remove(os.path.join(os.path.dirname(temp_file), f))
                        except OSError:
                            pass


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
