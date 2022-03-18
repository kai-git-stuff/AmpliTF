   def chain1_asTree(self,smp:PhaseSpaceSample,ld,la,lb,lc,resonances):
        """For explanation see chain3"""
        # channel 1
        # L_b -> C a : C -> (b,c)  
        sgma1 = self.phsp.m2bc(smp)
        sgma2 = self.phsp.m2ac(smp)
        sgma3 = self.phsp.m2ab(smp)
        def do_Resonance(self,res):
            sC,pC,helicities_C,bls_in,bls_out,X = res
            ns = atfi.cast_complex(atfi.sqrt(atfi.const(sC+1)))
            nj = atfi.cast_complex(atfi.sqrt(atfi.const(self.sd+1)))

            # getting the Blatt-Weisskopf form factors into our bls
            bls_in = bls_in(s = sgma1,d = self.d,md = self.md,mbachelor=self.ma)
            bls_out = bls_out(sgma1)
            for lC in helicities_C:
                helicities_abc = helicity_options(sC,self.sa,self.sb,self.sc)
                for la_,lb_,lc_ in helicities_abc:
                    if la_ != la : continue # atfi.cast_complex(wigner_small_d(zeta_1,self.sa,la_,la)) yields delta, bc. zeta_1 = 0
                    if lC != la_-ld: continue # bc. theta_hat = 0, the d_matrix for this is also a delta
                    # C -> b c
                    # Rotation in the isobar system
                    # angle between A momentum (isobar) and lmbda_c in rest frame of Isobar #
                    H_A_c = ( atfi.sqrt(phasespace_factor(sgma1,self.mb,self.mc) * phasespace_factor(self.md,sgma1,self.ma)) * 
                            atfi.cast_complex(helicity_couplings_from_ls(self.sd,sC,self.sa,lC,la_-ld,bls_in))  )
                    H_b_c =  helicity_coupling_times_d(theta,sC,self.sb,self.sc,lb_,lc_,lC,bls_out)
                    # symmetry of the d matrices
                    H_b_c *=  (-1)**((lc - lc_)/2) * (   # prefactors for index switches  
                            atfi.cast_complex(wigner_small_d(zeta_3,self.sc,lc_,lc)) * 
                            atfi.cast_complex(wigner_small_d(zeta_2,self.sb,lb_,lb)) )
                    return ns * nj * H_A_c * H_b_c 


        ampl = atfi.zeros_tensor(self.phsp.m2ab(smp).shape,atfi.ctype())
        # Rotation in the isobar system
        theta_hat =  atfi.acos(cos_theta_hat_1_canonical_1(self.md, self.ma, self.mb, self.mc, sgma1, sgma2, sgma3))
        theta_hat = 0
        # will just return one, as we are in the alligned system anyways
        theta = atfi.acos(cos_theta_23(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        zeta_1 = 0
        # own system
        zeta_2 = atfi.acos(cos_zeta_2_aligned_1_in_frame_2(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        zeta_3 = atfi.acos(cos_zeta_3_aligned_3_in_frame_1(self.md,self.ma,self.mb,self.mc,sgma1,sgma2,sgma3))
        # remember to apply (-1)**((lc - lc_)/2) in front of the d matrix (switch last 2 indices)
        for res in resonances:

        return ampl