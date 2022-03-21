
from amplitf.amplitudes.resonances import BaseResonance

class TreeAmplitude:
    def __init__(self,root=False,name=""):
        self.root = root
        self.parents = []
        self.branches = {}
        self.dependencies = {}
        self.state = None
        self.func = None
        self.last_params = {}
        self.name = name
    
    def __repr__(self):
        return "TreeAmplitude %s"%self.name

    def update(self,parameter,*args,**kwargs):
        if not self.update_needed(parameter) and self.state is not None:
            return
        print("Updated %s"%self)
        for name, branch in self.branches.items():
            branch.update(parameter,*args,**kwargs)
        self.last_params = parameter.copy()
        self.state = self.func(self,parameter)

    def add_parent(self,parent):
        if not isinstance(parent,TreeAmplitude):
            raise(ValueError("Only TreeAmplitudes can be set as parents!"))
        self.parents.append(parent)

    def add_branch(self, name, branch):
        self.branches[name] = branch
        branch.add_parent(self)
        self.dependencies.update(branch.dependencies)
    
    def __call__(self,parameter,*args,**kword_args):            
        self.update(parameter,*args,**kword_args)
        return self.state

    def set_func(self,fncn):
        """functions have to fit into the given schema
        f(t:TreeAmplitude,parameter) parameter is a dict
        """
        self.func = fncn

    def set_dependency(self,param):
        self.dependencies[param] = True
        for p in self.parents:
            p.set_dependency(param)
    
    def update_needed(self,parameter):
        for k,v in parameter.items():
            if self.dependencies.get(k,False):
                last_param = self.last_params.get(k,None)
                if last_param is not None and last_param != v:
                    return True
        return False

        

class TreeResonance(TreeAmplitude):
    def __init__(self,resonance:BaseResonance,name=""):
        super().__init__(False,name)
        self.resonance = resonance



    def __iter__(self):
        return self.resonance.__iter__()