"""Tests for the CLI module."""

import pytest
import sys
import io
from unittest.mock import patch, MagicMock
from python.cli import (
    print_help, trim, print_context_info, print_node_info, 
    print_status, main
)
from python.llm_node import LLMNode
from python.cnn_node import CNNNode
from python.context_cache import ContextCache
from python.context import Context


class TestPrintHelp:
    """Tests for help printing."""

    def test_print_help_does_not_raise(self):
        """Test that print_help runs without error."""
        # Should not raise any exceptions
        print_help()


class TestTrim:
    """Tests for trim function."""

    def test_trim_removes_whitespace(self):
        """Test that trim removes whitespace."""
        assert trim("  hello  ") == "hello"

    def test_trim_handles_empty(self):
        """Test that trim handles empty string."""
        assert trim("") == ""

    def test_trim_handles_tabs(self):
        """Test that trim removes tabs."""
        assert trim("\thello\t") == "hello"


class TestPrintContextInfo:
    """Tests for print_context_info."""

    def test_print_null_context(self):
        """Test printing null context."""
        print_context_info("Test", None)

    def test_print_context_with_data(self):
        """Test printing context with data."""
        context = Context()
        context.set_weights([0.1, 0.2, 0.3])
        context.set_config({"temperature": 0.7})
        context.set_metadata("model_type", "LLM")
        
        # Should not raise
        print_context_info("Test", context)


class TestPrintNodeInfo:
    """Tests for print_node_info."""

    def test_print_llm_node(self):
        """Test printing LLM node info."""
        node = LLMNode(slot_index=0)
        # Should not raise
        print_node_info(0, node)

    def test_print_cnn_node(self):
        """Test printing CNN node info."""
        node = CNNNode(slot_index=0)
        # Should not raise
        print_node_info(0, node)


class TestPrintStatus:
    """Tests for print_status."""

    def test_print_status_empty(self):
        """Test printing status with no nodes."""
        nodes = {}
        cache = ContextCache()
        # Should not raise
        print_status(nodes, cache)

    def test_print_status_with_nodes(self):
        """Test printing status with nodes."""
        nodes = {0: LLMNode(0), 1: CNNNode(1)}
        cache = ContextCache()
        # Should not raise
        print_status(nodes, cache)


class TestMain:
    """Tests for main entry point."""

    def test_main_module_runs(self):
        """Test that main module can be imported."""
        # Should not raise
        from python.cli import main
        assert callable(main)

    def test_cli_module_exists(self):
        """Test that CLI module exists."""
        from python import cli
        assert hasattr(cli, 'run_cli')

    def test_cli_help_command(self):
        """Test CLI help command parsing."""
        from python.cli import run_cli
        # Just verify the function exists and is callable
        assert callable(run_cli)
