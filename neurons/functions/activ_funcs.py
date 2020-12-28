from math import isfinite, exp, tanh, sin, cos, pi

f_types = ["unipolstep", "heaviside",
			"bipolstep", 
			"unipollin",
			"bipollin",
			"sigmoid", 
			"tanh", 
			"bipolsin", 
			"unipolcos",
			"linear",
			"gaussian"]


def get_activ_fun(f_type, param):
	if not isinstance(f_type, str):
		raise Exception("f_type must be str object from list above:\n"+str(f_types))
	f_type = f_type.lower()
	if f_type not in f_types:
		raise Exception("f_type must be one from list above:\n"+str(f_types))
	if not isinstance(param, int) and not isinstance(param, float):
		raise Exception("param must be int or float object")
	if not isfinite(param):
		raise Exception("param must have finit value")
	if param <= 0 and f_type in ["sigmoid", "tanh", "bipolsin", "unipolcos"]:
		raise Exception("param must be positive when %s activation function used" % f_type)

	if f_type in ["unipolstep", "heaviside"]:
		return lambda x: 1 if x >= param else 0
	elif f_type == "bipolstep":
		return lambda x: 1 if x >= param else -1
	elif f_type == "unipollin":
		return lambda x: 1 if x >= param else 0 if x <= param else 0.5*(1+x/param)
	elif f_type == "bipollin":
		return lambda x: 1 if x >= param else -1 if x <= param else x/param
	elif f_type == "sigmoid":
		return lambda x: 1/(1+exp(-x*param))
	elif f_type == "tanh":
		return lambda x: tanh(param*x)
	elif f_type == "bipolsin":
		return lambda x: -1 if x <= -param else 1 if x >= param else sin(0.5*pi*x/param)
	elif f_type == "unipolcos":
		return lambda x: 0 if x <= -param else 1 if x >= param else 0.5*(1+cos(0.5*pi*(x-param)/param))
	elif f_type == "linear":
		return lambda x: x*param
	else:
		raise Exception("Activation function of type '"+f_type+"' not found.")



