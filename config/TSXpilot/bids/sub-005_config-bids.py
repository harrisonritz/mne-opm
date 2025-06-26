"""
BIDS configuration file for subject 003 in TSXpilot study.
Harrison Ritz 2025
"""
import os

# Session information (previously under session:)
ids = int(os.environ.get('SUBJECT'))
task = "TSXpilot"
session = "01"

# Trigger mapping and annotation renaming (previously under trigger:)
rename_annot = True

trigger_desc = {
    1: 'feedback',
    2: 'ITI',
    5: 'CSI',
    4: 'trial/read_noresp',
    8: 'trial/listen_noresp',
    16: 'trial/av_noresp',
    32: 'trial/unimodal_read',
    64: 'trial/bimodal_read',
    6: 'trial/unimodal_listen',
    10: 'trial/bimodal_listen',
    9: 'trial/read_read',
    17: 'trial/listen_listen',
    33: 'trial/read_listen',
    65: 'trial/listen_read'
}

response_desc = {
    'BNC 1 Z': 'response/left',
    'BNC 5 Z': 'response/right'
}

# Recording information (previously under recording_info:)
line_freq = 60.0
bads = [
    'P8 24 Z', 'P8 24 Y', 'P8 24 X',
    'Pz 25 Z', 'Pz 25 Y', 'Pz 25 X', 
    'C4 2C Z', 'C4 2C Y', 'C4 2C X', 
    'AFz 2G Z', 'AFz 2G Y', 'AFz 2G X', 
    'T4 2J Z', 'T4 2J Y', 'T4 2J X', 
    'F10 2O Z', 'F10 2O Y', 'F10 2O X', 
    'O5 31 Z', 'O5 31 Y', 'O5 31 X', 
    'C3 37 Z', 'C3 37 Y', 'C3 37 X', 
    'Fz 3C Z', 'Fz 3C Y', 'Fz 3C X', 
    'F9 3G Z', 'F9 3G Y', 'F9 3G X', 
    'F3 3I Z', 'F3 3I Y', 'F3 3I X', 
    'Fpz 3K X', 'Fpz 3K Y', 'Fpz 3K Z',
    # 'F4 2N X', 'F4 2N Y', 'F4 2N Z',
    # 'T6 1Y X', 'T6 1Y Y', 'T6 1Y Z',
]
