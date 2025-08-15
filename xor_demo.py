#!/usr/bin/env python3
"""Backward-compatible runner that now defers to the xor_demo package CLI."""
from xor_demo.cli import main

if __name__ == "__main__":
    main()

