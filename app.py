from routes import app
import os
import logging
from utils.logs import LogConfig

LogConfig(os.path.split(os.path.realpath(__file__))[0])
app.config['JSON_AS_ASCII'] = False
# app.run(host='0.0.0.0', port=80)
app.run(host='0.0.0.0')
