import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fedbiomed.cli import ComponentParser, run


@pytest.mark.parametrize(
    "arguments,expected_code,extra",
    [
        (["--help"], 0, None),
        (["node", "--help"], 0, None),
        (["node", "api", "start", "--help"], 0, None),
        (["node", "gui", "start", "--help"], 0, None),
        (["node", "dataset", "list"], 0, None),
        (["node", "certificate", "list"], 0, None),
        (["node", "api", "start"], 1, "node-api"),
        (["node", "gui", "start"], 1, "gui"),
    ],
)
def test_cli_without_optional_web_packages(tmp_path, arguments, expected_code, extra):
    # Test in a fresh process: already-imported modules in pytest must not hide
    # an eager import. Block web dependencies even if the developer installed them.
    script = """
        import sys
        class NoWebPackages:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {
                    'fedbiomed_node_api', 'fedbiomed_gui', 'flask',
                    'flask_jwt_extended', 'gunicorn', 'cachelib', 'jsonschema',
                }:
                    raise ModuleNotFoundError(f'No module named {fullname!r}', name=fullname)
        sys.meta_path.insert(0, NoWebPackages())
        from fedbiomed.cli import run
        run()
    """
    (tmp_path / "fbm-node").mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    if extra:
        assert f"fedbiomed[{extra}]==" in result.stdout


class TestComponentParser(unittest.TestCase):
    def setUp(self):
        self.main_parser = argparse.ArgumentParser()
        self.parser = self.main_parser.add_subparsers()
        self.conf_parser = ComponentParser(subparser=self.parser)
        self.conf_parser.initialize()

        self.tem = tempfile.TemporaryDirectory()
        self.initial_dir = os.getcwd()
        os.chdir(self.tem.name)

    def tearDown(self):
        os.chdir(self.initial_dir)
        # forces the removal of config files for Nodes and Researchers
        if "fedbiomed.researcher.config" in sys.modules:
            sys.modules.pop("fedbiomed.researcher.config")

        if "fedbiomed.node.config" in sys.modules:
            sys.modules.pop("fedbiomed.node.config")
        self.tem.cleanup()

    def test_01_component_parser_initialize(self):
        """Tests argument initialization"""
        self.assertTrue("component" in self.conf_parser._subparser.choices)
        self.assertTrue(
            "create"
            in self.conf_parser._subparser.choices["component"]
            ._subparsers._group_actions[0]
            .choices
        )
        self.assertEqual(
            self.conf_parser._subparser.choices["component"]
            ._subparsers._group_actions[0]
            .choices["create"]
            ._defaults["func"]
            .__func__.__name__,
            "create",
        )

    def test_02_component_parser_create_success(self):
        with (
            tempfile.TemporaryDirectory() as node_dir,
            tempfile.TemporaryDirectory() as res_dir,
        ):
            args_list_set = [
                [
                    "component",
                    "create",
                    "--path",
                    node_dir,
                    "--component",
                    "NODE",
                    "-eo",
                ],
                [
                    "component",
                    "create",
                    "--path",
                    res_dir,
                    "--component",
                    "RESEARCHER",
                    "-eo",
                ],
                ["component", "create", "--component", "NODE"],
                ["component", "create", "--component", "NODE", "-eo"],
                ["component", "create", "--component", "RESEARCHER"],
                ["component", "create", "--component", "RESEARCHER", "-eo"],
            ]
            for args_list in args_list_set:
                args = self.main_parser.parse_args(args_list)
                self.conf_parser.create(args)
                # If cleanup of researcher config is required between runs:
                sys.modules.pop("fedbiomed.researcher.config", None)

    def test_03_component_parser_create_fail(self):
        """Tests component creation parser fails"""

        args_list_preset = [
            ["component", "create", "--component", "NODE"],
            ["component", "create", "--component", "RESEARCHER"],
        ]
        for args_list in args_list_preset:
            args = self.main_parser.parse_args(args_list)
            self.conf_parser.create(args)

        # 1. fails at parse
        args_list_set = [
            ["component", "create", "-p", "p1", "-p", "p2"],
        ]

        for args_list in args_list_set:
            with self.assertRaises(SystemExit):
                self.main_parser.parse_args(args_list)

        # 2. fails at create
        args_list_set = [
            ["component", "create", "--component"],
            ["component", "create", "--component", "node"],
            ["component", "create", "--component", "researcher"],
        ]

        for args_list in args_list_set:
            args = self.main_parser.parse_args(args_list)
            with self.assertRaises(SystemExit):
                self.conf_parser.create(args)

        # 3. bad component type
        with self.assertRaises(SystemExit):
            self.conf_parser._get_component_instance("any_path", "bad_component_type")

        sys.modules.pop("fedbiomed.researcher.config")

    def test_04_component_parser_create_reports_the_component_it_created(self):
        """Creating a component says where it was created, researcher included.

        A researcher is created by importing its configuration module, which is
        a different path through `create` than the node's.
        """
        for component in ("NODE", "RESEARCHER"):
            for path in ([], ["--path", f"{component.lower()}-at-a-given-path"]):
                with self.subTest(component=component, path=path):
                    args = self.main_parser.parse_args(
                        ["component", "create", "--component", component, *path]
                    )

                    printed = io.StringIO()
                    with contextlib.redirect_stdout(printed):
                        self.conf_parser.create(args)
                    sys.modules.pop("fedbiomed.researcher.config", None)

                    self.assertIn("Operation successful!", printed.getvalue())
                    self.assertIn(
                        "Component has been initialized in", printed.getvalue()
                    )


@pytest.mark.parametrize("unknown", ["--force", "--totally-bogus"])
def test_run_rejects_unknown_arguments(monkeypatch, capsys, unknown):
    """An option the root parser does not know stops the run.

    Ignoring it would carry out a different command than the one typed: a removed
    option such as `--force` would run without it and still report success.
    """
    monkeypatch.setattr(sys, "argv", ["fedbiomed", "certificate-dev-setup", unknown])

    with pytest.raises(SystemExit) as exc_info:
        run()

    assert exc_info.value.code == 2
    assert f"unrecognized arguments: {unknown}" in capsys.readouterr().err


def test_run_forwards_unparsed_arguments_to_the_component_cli(monkeypatch):
    """Node and researcher parse their own arguments, so those are passed on as they
    are rather than judged by the root parser."""
    monkeypatch.setattr(
        sys, "argv", ["fedbiomed", "node", "-p", "x", "certificate", "list"]
    )
    node_cli = MagicMock()
    monkeypatch.setattr("fedbiomed.node.cli.NodeCLI", node_cli)

    run()

    node_cli.return_value.parse_args.assert_called_once_with(
        ["-p", "x", "certificate", "list"]
    )


if __name__ == "__main__":
    unittest.main()
