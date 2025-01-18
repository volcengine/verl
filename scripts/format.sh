#!/bin/bash
pip3 install --upgrade yapf
yapf -ir -vv --style ./.style.yapf verl tests single_controller examples