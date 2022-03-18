from amplitf import interface as atfi

class TreeFunction:
    def __init__(self):
        pass

class TreeCaller:
    def __init__(self,functions:dict,kwargs:dict):
        """
        kwargs = dict[list] dict of list of all the functions and their ordered parameters(names only)
        functions _ dict[callable] key = name of function value = function
        """
        self.arg_f_mapping = {}
        self.state = {}
        self.f_args_mapping = kwargs
        self.current_kwargs = {}
        self.functions = functions
        for func_name,f in functions.items():
            kw = kwargs[func_name]
            for k in kw:
                self.arg_f_mapping[k] = (kw,f,func_name)
                self.current_kwargs[k] = 0
        
    def check(self,k,v):
        if (v2 := self.current_kwargs.get(k,None)) is not None:
            return v != v2

    def update_state(self, kwargs:dict):
        diff = {k for k,v in kwargs.items() if self.check(k,v)}
        names = set()
        for k in diff:
            f,kw,func_name = self.arg_f_mapping[k]
            names.add(func_name)
        
        for func_name in names:
            self.state[func_name] = f(*[kwargs[key] for key in self.f_args_mapping[func_name]])
        
    def get(self,kw):
        return self.state[kw]
