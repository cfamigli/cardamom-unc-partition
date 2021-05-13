#!/bin/bash

ml load viz
ml load py-numpy/1.18.1_py36
ml load py-pandas/1.0.3_py36
ml load py-matplotlib/3.2.1_py36
ml load py-scipy/1.4.1_py36

python3 edit_cbf_files.py
