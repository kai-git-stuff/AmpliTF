from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.base_phasespace import PhaseSpaceSample
from amplitf.dynamics import breit_wigner_decay_lineshape
from amplitf.kinematics import *
from amplitf.dalitz_decomposition import *
import amplitf.interface as atfi
from amplitf.constants import spin as sp


def helicity_options(J,s1,s2,s3):
    """gives all possible helicity comibnations for 3 given spins 
        J is ignored for now!
    """
    options = []
    for m1 in sp.direction_options(s1):
        for m2 in sp.direction_options(s2):
            for m3 in sp.direction_options(s3):
                options.append((m1,m2,m3))
    return options

def possible_LS_states(J,s1,s2,P,p1,p2,parity_conservation=True) -> dict:
    bls = {}
    if (sp.is_half(J) and sp.is_half(s1+s2) ) or not(sp.is_half(J) or sp.is_half(s1+s2)) or not parity_conservation:
        s_max,s_min = s1+s2, abs(s1-s2)
        for s in range(s_min,s_max+1,2):
            for l in range(0,J+s+1,2):
                if J <= l+s and J >= abs(l-s)  and P == p1*p2*(-1)**(l/2.):
                    bls[(l,s)] = ((l + 1)/(J+1))**0.5
    return bls

def phasespace_factor(md,ma,mb):
    # phasespace factor for the dalitz functions
    return atfi.cast_complex(4 * atfi.pi()* atfi.sqrt(md/two_body_momentum(md,ma,mb)))

def helicity_coupling_times_d(theta,J,s1,s2,l1,l2,nu,bls):
    return (atfi.cast_complex(helicity_couplings_from_ls(J,s1,s2,l1,l2,bls)) * # helicity based
            atfi.cast_complex(wigner_small_d(theta,J,nu,l1-l2)) ) * (-1)**((s2-l2)/2) # spin orientation based -> -l2 = m2 (z-achsis is along l1)

class dalitz_decay:
    """
    class modeled after dalitz function from dalitz plot decomposition from https://arxiv.org/pdf/1910.04566.pdf
    Can take any decay with defined resonances
    Background terms need to be given as a resonance aswell
    For the resonances the functions will expect a list of objects inheriting from BaseResonance
    The needed functions from BaseResonance, to be implemented are (for the rest do not change the main impelmentation): 
    X(s,L) -> the lineshape function depending on the CMS energy and the angular momentum
    
    use d=None to disable the use of Blatt-Weisskopf form factors and orbital barriers
    if d is None the orbital barriers need to be implemented in the resonances
    This is best done by overwritng the bls method from BaseResonance in your resonance classes,
    because all angular momentum based funtions should live in the bls couplings
    """
    def __init__(self,md,ma,mb,mc,sd,sa,sb,sc,pd,pa,pb,pc,d=1.5/1000.,phsp = None):
        self.pd = pd 
        self.pa = pa 
        self.pb = pb 
        self.pc = pc 
        self.sd = sd
        self.sa = sa
        self.sb = sb
        self.sc = sc

        self.ma = ma
        self.mb = mb 
        self.mc = mc 
        self.md = md 

        self.d = d
        # d is the radius of the decaying particle for the Blatt-Weisskopf FF
        # set d = None to disable use of Blatt-Weisskopf FF

        if phsp is None:
            self.phsp = DalitzPhaseSpace(ma,mb,mc,md)
        else:
            self.phsp = phsp

    def chain3(self,smp:PhaseSpaceSample,ld,la,lb,lc,resonances):
        """
        ld: helicity of mother particle in CMS sytem of mother (see https://arxiv.org/pdf/1910.04566.pdf)
        la,lb,lc helicites of decay products
        resonances: list[BaseResonance] of the resonances (background term needs to be resonance aswell)
        """
        # channel 3   
        # d -> A c : A -> a b
        sgma3 = self.phsp.m2ab(smp)
        sgma2 = self.phsp.m2ac(smp)
        sgma1 = self.phsp.m2bc(smp)
        ampl = atfi.zeros_tensor(sgma1.shape,atfi.ctype())
        
        # Rotation in the isobar system
        # angle between momentum of L_b and spectator 
        theta_hat = atfi.acos(cos_theta_hat_3_canonical_1(self.md, self.ma, self.mb, self.mc, sgma1, sgma2, sgma3))
        theta = atfi.acos(cos_theta_12(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        zeta_1 = atfi.acos(cos_zeta_1_aligned_3_in_frame_1(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        zeta_2 = atfi.acos(cos_zeta_2_aligned_2_in_frame_3(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        # remember to apply (-1)**((lb_ - lb)/2) in front of the d matrix (switch last 2 indices)
        zeta_3 = 0
        # aligned system

        for sA,pA,helicities_A,bls_in,bls_out,X in resonances:
            ns = atfi.cast_complex(atfi.sqrt(atfi.const(sA+1)))
            nj = atfi.cast_complex(atfi.sqrt(atfi.const(self.sd+1)))
            bls_in = bls_in(s = sgma3,d = self.d,md = self.md,mbachelor=self.mc)
            bls_out = bls_out(sgma3)
            for lA in helicities_A:           
                helicities_abc = helicity_options(sA,self.sa,self.sb,self.sc)
                for la_,lb_,lc_ in helicities_abc:
                    H_A_c = ( atfi.sqrt(phasespace_factor(self.md,sgma3,self.mc)* phasespace_factor(sgma3,self.ma,self.mb)) * 
                        helicity_coupling_times_d(theta_hat,self.sd,sA,self.sc,lA,lc_,ld,bls_in) 
                        )

                    # Rotation in the isobar system
                    # angle between A momentum (isobar) and lmbda_c in rest frame of Isobar 
                    H_a_b =  helicity_coupling_times_d(theta,sA,self.sa,self.sb,la_,lb_,lA,bls_out)
                    H_a_b *= (-1)**((lb_-lb)/2) * (       # prefactors for index switches
                        atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la_,la)) * 
                        atfi.cast_complex(wigner_small_d(zeta_2,self.sb,lb_,lb)) * 
                        atfi.cast_complex(wigner_small_d(zeta_3,self.sc,lc_,lc))  )
                    ampl +=nj * ns* H_A_c * H_a_b 
        return ampl

    def chain2(self,smp:PhaseSpaceSample,ld,la,lb,lc,resonances):
        """For explanation see chain3"""
        # channel 2
        # L_b -> B b : B -> (a,c)
        sgma3 = self.phsp.m2ab(smp)
        sgma2 = self.phsp.m2ac(smp)
        sgma1 = self.phsp.m2bc(smp)
        ampl = atfi.zeros_tensor(self.phsp.m2ab(smp).shape,atfi.ctype())

        zeta_1 = atfi.acos(cos_zeta_1_aligned_1_in_frame_2(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        # remember to apply (-1)**((la - la_)/2) in front of the d matrix (switch last 2 indices)
        zeta_2 = 0 # allways one is 0
        zeta_3 = atfi.acos(cos_zeta_3_aligned_2_in_frame_3(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        theta_hat =  atfi.acos(cos_theta_hat_1_canonical_2(self.md, self.ma, self.mb, self.mc, sgma1, sgma2, sgma3))
        # remember factor of (-1)**((ld - lB + lb_)/2) because we switched indices 1 and 2
        theta = atfi.acos(cos_theta_31(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        phsp_factor = atfi.sqrt(phasespace_factor(sgma2,self.ma,self.mc)* phasespace_factor(self.md,sgma2,self.mb))
        for sB,pB,helicities_B,bls_in,bls_out,X in resonances:
            ns = atfi.cast_complex(atfi.sqrt(atfi.const(sB+1)))
            nj = atfi.cast_complex(atfi.sqrt(atfi.const(self.sd+1)))
            bls_in = bls_in(s = sgma2,d = self.d,md = self.md,mbachelor=self.mb)
            bls_out = bls_out(sgma2)
            for lB in helicities_B:
                # channel 2
                # L_b -> B b : B -> (a,c)
                helicities_abc = helicity_options(sB,self.sa,self.sb,self.sc)
                for la_,lb_,lc_ in helicities_abc:
                    # Rotation in the isobar system
                    H_B_b = phsp_factor * helicity_coupling_times_d(theta_hat,self.sd,sB,self.sb,lB,lb_,ld,bls_in) # ToDo why the other way arround then in the paper? 
                    H_a_c =   helicity_coupling_times_d(theta,sB,self.sc,self.sa,lc_,la_,lB,bls_out)
                    H_a_c *= (-1)**((ld - lB + lb_)/2)  * (-1)**((la_ - la)/2) * ( # prefactors for index switches
                        atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la_,la)) *  
                        atfi.cast_complex(wigner_small_d(zeta_2,self.sb,lb_,lb)) *
                        atfi.cast_complex(wigner_small_d(zeta_3,self.sc,lc_,lc)) )
                    ampl += nj * ns * H_B_b * H_a_c 
        return ampl

    def chain1(self,smp:PhaseSpaceSample,ld,la,lb,lc,resonances):
        """For explanation see chain3"""
        # channel 1
        # L_b -> C a : C -> (b,c)  
        sgma3 = self.phsp.m2ab(smp)
        sgma2 = self.phsp.m2ac(smp)
        sgma1 = self.phsp.m2bc(smp)
        ampl = atfi.zeros_tensor(self.phsp.m2ab(smp).shape,atfi.ctype())
        # Rotation in the isobar system
        theta_hat =  atfi.acos(cos_theta_hat_1_canonical_1(self.md, self.ma, self.mb, self.mc, sgma1, sgma2, sgma3))
        # will just return one, as we are in the alligned system anyways
        theta = atfi.acos(cos_theta_23(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        zeta_1 = 0
        # own system
        zeta_2 = atfi.acos(cos_zeta_2_aligned_1_in_frame_2(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        zeta_3 = atfi.acos(cos_zeta_3_aligned_3_in_frame_1(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        # remember to apply (-1)**((lc - lc_)/2) in front of the d matrix (switch last 2 indices)
        for sC,pC,helicities_C,bls_in,bls_out,X in resonances:
            
            ns = atfi.cast_complex(atfi.sqrt(atfi.const(sC+1)))
            nj = atfi.cast_complex(atfi.sqrt(atfi.const(self.sd+1)))

            # getting the Blatt-Weisskopf form factors into our bls
            bls_in = bls_in(s = sgma1,d = self.d,md = self.md,mbachelor=self.ma)
            bls_out = bls_out(sgma1)
            for lC in helicities_C:
                helicities_abc = helicity_options(sC,self.sa,self.sb,self.sc)
                for la_,lb_,lc_ in helicities_abc:
                    # C -> b c
                    # Rotation in the isobar system
                    # angle between A momentum (isobar) and lmbda_c in rest frame of Isobar #
                    H_A_c = ( atfi.sqrt(phasespace_factor(sgma1,self.mb,self.mc) * phasespace_factor(self.md,sgma1,self.ma)) * 
                            helicity_coupling_times_d(theta_hat,self.sd,sC,self.sa,lC,la_,ld,bls_in)   )
                    H_b_c =  helicity_coupling_times_d(theta,sC,self.sb,self.sc,lb_,lc_,lC,bls_out)
                    # symmetry of the d matrices
                    H_b_c *=  (-1)**((lc_ - lc)/2) * (   # prefactors for index switches  
                         atfi.cast_complex(wigner_small_d(zeta_3,self.sc,lc_,lc)) * 
                         atfi.cast_complex(wigner_small_d(zeta_2,self.sb,lb_,lb)) * 
                         atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la_,la)) )
                    ampl += ns * nj * H_A_c * H_b_c 
        return ampl

