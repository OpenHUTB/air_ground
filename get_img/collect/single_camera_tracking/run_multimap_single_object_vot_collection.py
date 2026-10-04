#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect the configured SOT motion-pilot sequences from each release map."""

import sys

from run_multimap_tracking_common import main_for_task


if __name__ == "__main__":
    sys.exit(main_for_task("single_object"))
