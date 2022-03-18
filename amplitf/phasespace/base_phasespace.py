from multiprocessing.sharedctypes import Value
import re
import amplitf.interface as atfi


class BasePhaseSpace:
    def __init__(self,ndim):
        self.ndim = ndim

    def inside(self) -> bool: 
        """check if point is inside"""
        raise NotImplementedError("This is just an abstract Base class! Youse specific pahse spaces instead!")

    def VaribaleMapping(self):
        """returns a dictionary of names and indices for the different variables"""
        return {}
    
    def dimensionality(self):
        return self.ndim

    def filter(self):
        raise NotImplementedError()
    
    def unfiltered_sample(self):
        raise NotImplementedError()
    
    def uniform_sample(self):
        raise NotImplementedError()

class PhaseSpaceSample:
    """An amount of points inside a specified phase space.
    Point of having a special class for this is so, that one can easily assure, that the variables work and are not confused with one another.
    """
    def __init__(self,phasespace:BasePhaseSpace,sample = None):
        self.__phasespace = phasespace
        self.sample = sample

        self.variable_map = phasespace.VaribaleMapping()

        for var_name,index in self.variable_map.items():
            """create getter functions for the variables"""
            @atfi.function
            def f(index=index):
                return self.GetVariable(index=index)
            setattr(self,"get_%s"%var_name,f)

    def __getitem__(self,slice):
        if self.sample is None:
            raise ValueError("No data in sample yet!")
        return self.sample[slice]

    def setSample(self,sample):
        if atfi.shape(sample)[-1] != self.__phasespace.dimensionality():
            raise ValueError("Sample does not fit the Phase Space!")
        self.sample = sample

    @atfi.function
    def GetVariable(self,name=None,index=None):
        if index is not None and isinstance(index,int):
            return self.sample[:,index]
        if name is not None:
            index = self.variable_map[name]
            return self.sample[...,index]
        raise ValueError("Either variable Name or Index must be specified!")

    def ChangePhaseSpace(self,phasespace:BasePhaseSpace) -> None:
        if not isinstance(phasespace,type(self.__phasespace)):
            raise ValueError("You can not switch the type of the phase space! Old: %s to new: %s"%(type(self.__phasespace), type(phasespace)))

        if self.__phasespace.ndim != phasespace.ndim:
            raise ValueError("Dimensions of old and new phasespace need to be the same! Old %s, new: %s"%(self.__phasespace.ndim, phasespace.ndim))
        
        self.__phasespace = phasespace

    def filter(self):
        if self.sample is None:
            raise ValueError("Empty sample can not be filtered!")
        return PhaseSpaceSample(self.__phasespace,sample = self.__phasespace.filter(self.sample))
    
    def __len__(self):
        if self.sample is None:
            return 0
        return len(self.sample)

    @property
    def data(self):
        return self.sample