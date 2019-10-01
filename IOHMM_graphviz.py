#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

__author__ = """Maziar Fooladi"""

import pygraphviz as pgv


B=pgv.AGraph('DBN_Multi_Agent_Unrolled.dot') # create a new graph from file
B.layout() # layout with default (neato)
B.draw('DBN2.png') # draw png