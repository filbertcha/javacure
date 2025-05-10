# __init__.py (Perbaikan dengan relasi user yang lebih jelas)
from flask import Flask

app = Flask(__name__)

# Import routes di akhir untuk menghindari circular import
from routes import *