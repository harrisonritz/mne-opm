"""
BIDS configuration file for subject 003 in TSXpilot study.
Harrison Ritz 2025
"""

# Session information (previously under session:)
ids = 4
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
    'T11 32 Y', 'T11 32 X', 'T11 32 Z', 
    'O8 21 Z', 'O8 21 Y', 'O8 21 X', 
    'Pz 25 Z', 'Pz 25 Y', 'Pz 25 X', 
    'C4 2C Z', 'C4 2C Y', 'C4 2C X', 
    'AFz 2G Z', 'AFz 2G Y', 'AFz 2G X', 
    'F10 2O Z', 'F10 2O Y', 'F10 2O X', 
    'C3 37 Z', 'C3 37 Y', 'C3 37 X', 
    'F9 3G Z', 'F9 3G Y', 'F9 3G X', 
    'Fpz 3K Z', 'Fpz 3K Y', 'Fpz 3K X',
]
