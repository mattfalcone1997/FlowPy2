import pandas as pd


class AxisCSV:
    def __init__(self):
        pass

    def add_line(self, **kwargs):
        if len(kwargs) != 2:
            raise ValueError("Too many arguments given. Must be 2")

    def write(self, fname):
        pass
