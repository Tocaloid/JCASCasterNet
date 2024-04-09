# -*- coding: utf-8 -*-

import logging

def logger(s):
    tmp=logging.getLogger('log');
    if not tmp.handlers:
        tmp.setLevel(level=logging.INFO);
        handler=logging.FileHandler("./log.txt");
        handler.setLevel(logging.INFO);
        formatter=logging.Formatter('[%(asctime)s]: %(message)s');
        handler.setFormatter(formatter);
         
        console=logging.StreamHandler();
        console.setLevel(logging.INFO);
        console.setFormatter(formatter);
         
        tmp.addHandler(handler);
        tmp.addHandler(console);
    
    tmp.info(s);

def rmLogger():
    logging.shutdown();