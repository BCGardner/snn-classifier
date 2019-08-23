#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Parameter space definition.

This file is part of snn-classifier.

Copyright (C) 2018  Brian Gardner <brgardner@hotmail.co.uk>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
