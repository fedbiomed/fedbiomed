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



# should be on console only

logger.setLevel("DEBUG")
logger.debug( "this is a string on console")
logger.debug(jmesg)

# add file handler with ERROR
logger.addJsonFileHandler( level = logging.ERROR )

logger.debug( "only on console because level is low")
logger.error( "should be on console and file" )
logger.error( jmesg )

# test the hack around not overrided methods()
try:
    x = 0.0 / 0.0
except:
    logger.exception("Got the trace ? ")


# change globally the log levelS
logger.setLevel("WARNING")

logger.debug("nobody can see me")
logger.warning("every body should see me")
