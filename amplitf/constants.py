import amplitf.interface as atfi
import tensorflow as tf
class spin:
    # doubled
    SPIN_0 = 0
    SPIN_HALF = 1
    SPIN_1 = 2
    SPIN_3HALF = 3
    SPIN_2 = 4 
    SPIN_5HALF = 5
    SPIN_3 = 6 

    @staticmethod
    def is_half(s):
        return s%2 != 0 
    
    @staticmethod
    def direction_options(s):
        return [s_z for s_z in range(-s,s+1,2)]
    @staticmethod
    def direction_options_tf(s):
        return tf.data.Dataset.range(-s,s+1,2)
             

class angular:
    # not doubled 
    L_0 = 0
    L_1 = 1
    L_2 = 2
    L_3 = 3
    L_4 = 4 

