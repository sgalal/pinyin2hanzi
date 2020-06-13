# -*- coding: utf-8 -*-

import cherrypy
from inference_sd import convert

def parse_segment(ss, res):
	a = []
	for s, r in zip(ss, res):
		a.append(s)
		if r != 'x':
			a.append("'")
	return ''.join(a)[:-1]

class HelloWorld(object):
	@cherrypy.expose
	@cherrypy.tools.json_out()
	def jyutping(self, s):
		res = convert(s)
		if not res or '<' in res:
			return { "status": "F" }
		return \
			{ "result": res.replace('x', '')
			, "segment": parse_segment(s, res)
			, "status": "T"
			}

conf = \
	{ 'environment': 'production'
	, 'server.socket_host': '127.0.0.1'
	, 'server.socket_port': 4030
	}

cherrypy.config.update(conf)
cherrypy.quickstart(HelloWorld())
