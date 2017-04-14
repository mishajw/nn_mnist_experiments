#!/usr/bin/env python


class Model:
    def __init__(self, _input, _truth_output):
        self.input = _input
        self.truth_output = _truth_output
        self.output = None
        self.cost = None
