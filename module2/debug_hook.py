#!/usr/bin/env python
import code
import sys

def interact():
    frame = sys._getframe(1)
    code.interact(local=frame.f_locals)


