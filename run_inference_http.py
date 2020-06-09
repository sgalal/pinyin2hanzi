# -*- coding: utf-8 -*-

import cherrypy
from inference_sd import convert

class HelloWorld(object):
	@cherrypy.expose
	@cherrypy.tools.json_out()
	def jyutping(self, s):
		return \
			{ "result": convert(s)
			, "status": "T"
			}

conf = \
	{ 'server.socket_host': '127.0.0.1'
	, 'server.socket_port': 4030
	}

cherrypy.config.update(conf)
cherrypy.quickstart(HelloWorld())
