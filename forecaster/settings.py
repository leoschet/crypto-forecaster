"""
Common settings and globals.
If you plan to run this project create and configure 'development.py'.
For production, you must create and configure the 'production.py' file
Both should be under 'settings' folder.
"""


import os


########## PATH CONFIGURATION
# Project useful directories
ROOT = os.path.join(os.path.dirname(__file__), '..')

# Resources
RESOURCES_DIR = os.path.join(ROOT, 'res')
########## END PATH CONFIGURATION

########## DEBUG CONFIGURATION
# If True, return stacktrace for unhandled exceptions, error 500 otherwise.
DEBUG = False
########## END DEBUG CONFIGURATION