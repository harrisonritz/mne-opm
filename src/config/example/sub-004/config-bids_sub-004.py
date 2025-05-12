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
bads = ''
