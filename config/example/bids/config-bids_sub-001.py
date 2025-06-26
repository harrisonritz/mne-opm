"""
BIDS configuration file for subject 003 in TSXpilot study.
Harrison Ritz 2025
"""
import os

# Session information
ids = int(os.environ.get("SUBJECT"))
task = "AV"
session = "01"

# Trigger mapping and annotation renaming
rename_annot = True

trigger_desc = {
    4: "read",
    8: "listen",
    16: "av"
}

response_desc = {
}

# Recording information
line_freq = 60.0
bads = ['P10 1V Z', 'P10 1V Y', 'P10 1V X',
        'C6 2B Z','C6 2B Y', 'C6 2B X', 
        'T4 2J Y', 
        'T10 2I Z', 'T10 2I Y', 'T10 2I X', 
        'F2 2M Z', 'F2 2M Y', 'F2 2M X']
