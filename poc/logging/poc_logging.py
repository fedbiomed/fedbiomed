#!/usr/bin/env python

'''
Validate logging mechanism in:
- stdout/sdterr in script / cell in notebook
- file
(or both)

We cannot use logger.basicConfig() because of notebook usage.
'''

from mylogger import logger
import logging

#
jmesg = {
    "key_1" : 12.345,
    "key_2" : "this is a string",
    "key_3" : ( "this", "is", 1 , "list")
}

# check levels

logger.setLevel("DEBUG")
logger.debug( "this is a string")
logger.debug(jmesg)


# check not overridden methods

try:
    x = 0.0 / 0.0
except:
    logger.exception("Got the trace ? ")



# adding a console handler (not a json this time)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# should log on console and file
logger.debug('console and file ???')
