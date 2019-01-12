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
        Update only matching keys with type safety. None value can be
        overwritten by a float value.
        """
        for k, v in D.items():
            if k in self:
                # If new value has float or None type, then update
                if self[k] is None:
                    if isinstance(v, float) or v is None:
                        self[k] = v
                    else:
                        raise TypeError
                # If new value matches current value type, then update
                elif isinstance(v, type(self[k])):
                    self[k] = v
                else:
                    raise TypeError
