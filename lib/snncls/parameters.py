#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on June 2018

@author: BG
"""


class ParamSet(dict):
    """
    Parameter set container.
    """
    def overwrite(self, **D):
        """
        Update only matching keys with type safety.
        """
        for k, v in D.items():
            if k in self:
                if isinstance(v, type(self[k])):
                    self[k] = v
                else:
                    raise TypeError
