


class TreeAmplitude:
    def __init__(self,root=False):
        self.root = root
        self.parents = []
        self.branches = {}
        self.dependencies = {}
        self.state = None
        self.func = None
        self.last_params = None
    
    def update(self,kwargs):
        for branch in self.branches:
            branch.update(kwargs)
        self.state = self.func(self,kwargs)

    def add_parent(self,parent):
        if not isinstance(parent,TreeAmplitude):
            raise(ValueError("Only TreeAmplitudes can be set as parents!"))
        self.parents.append(parent)

    def add_branch(self, name, branch):
        self.branches[name] = branch
        branch.add_parent(self)
        self.dependencies.update(branch.dependencies)
    
    def __call__(self,kwargs):
        self.last_params = kwargs.copy()
        if self.update_needed(kwargs) or self.state is None:
            self.update(kwargs)
        return self.state

    def set_func(self,fncn):
        """functions have to fit into the given schema
        f(t:TreeAmplitude,kwargs) kwargs is a dict
        """
        self.func = fncn

    def set_dependency(self,kwarg):
        self.dependencies[kwarg] = True
        for p in self.parents:
            p.set_dependency(kwarg)
    
    def update_needed(self,kwargs):
        for k,v in kwargs.items():
            if self.dependencies.get(k,False):
                last_param = self.last_params.get(k,None)
                if last_param is not None and last_param != v:
                    return True
        return False

