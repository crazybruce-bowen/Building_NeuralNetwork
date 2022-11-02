"""
It is recommended to use a browser to download the data by
url: https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

"""

#%% Environment
import os
from os.path import dirname
import wget

# root path
script_path = os.path.abspath(__file__)
root_path = dirname(dirname(dirname(script_path)))
os.chdir(root_path)

#%% Download data
url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
download_path = os.path.join(root_path, 'assets', 'source')
wget.download(url, download_path)  # TODO test this
