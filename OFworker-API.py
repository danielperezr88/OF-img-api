# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:26:33 2016

@author: Dani
"""

import tornado.ioloop
from tornado_json.routes import get_routes
from tornado_json.application import Application

from google.protobuf import timestamp_pb2
from gcloud import storage

import OFWorker

PICKLE_BUCKET = 'pickles-python'

if __name__ == '__main__':
    
    client = storage.Client()
    for filename in ['age_classifier.pkl','gender_classifier.pkl']:
        cblob = client.get_bucket(PICKLE_BUCKET).get_blob(filename)
        fp = open(filename,'wb')
        cblob.download_to_file(fp)
        fp.close()
    
    routes = get_routes(OFWorker)
    
    application = Application(routes=routes, settings={})
    
    application.listen(8889)
    
    tornado.ioloop.IOLoop.instance().start()