import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from CGE import CGE


SAM      = pd.read_excel("SHM.xlsx", index_col = "index")
SAMCFrun = pd.read_excel("SHMCFrun.xlsx", index_col = "index")

class CGEResults():

    def __init__(self, sec_code):
    

        ########### KARŞI OLGUSAL DENGE [BAZ SENARYO - ENERJİ FİYAT ARTIŞI] ##############
        self.m1 = CGE(SAM)
        self.result1 = self.m1.SolveModel()

        self.m2 = CGE(SAM)
        self.m2.Pwmco*=1.25
        self.m2.Pwmng*=1.25

        self.result2 = self.m2.SolveModel()
        print(self.result2.message)
        ###################################################################################

        ########### KARŞI OLGUSAL DENGE [KARŞI OLGUSAL DENGE - ENERJİ FİYAT ARTIŞI] #######
        self.m1 = CGE(SAMCFrun)
        self.result1 = self.m1.SolveModel()

        self.m2 = CGE(SAMCFrun)
        self.m2.Pwmco*=1.25
        self.m2.Pwmng*=1.25
        
        self.result2 = self.m2.SolveModel()
        print(self.result2.message)
        ####################################################################################

        self.EndoVarandParameters()
        self.MacroVariables()

        result_df = pd.DataFrame(columns = ["base", "cfrun", "%Change"], index   = self.m1.init_values_str)
        result_df.base = self.result1.x
        result_df.cfrun = self.result2.x
        result_df["%Change"] = (result_df.cfrun - result_df.base) / result_df.base * 100
        result_df.to_excel("Results.xlsx")

       
        self.U_base, self.U_cfrun = -1*self.result1.fun, -1*self.result2.fun

    def EndoVarandParameters(self):

        [   X1_base, L1_base, K1_base, I11_base, I21_base, I31_base, Ir1_base, Ib1_base, Z1_base, E1_base, D1_base, Q1_base, M1_base, 
            X2_base, L2_base, K2_base, I12_base, I22_base, I32_base, Ir2_base, Ib2_base, Z2_base, E2_base, D2_base, Q2_base, M2_base, 
            X3_base, L3_base, K3_base, I13_base, I23_base, I33_base, Ir3_base, Ib3_base, Z3_base, E3_base, D3_base, Q3_base, M3_base, 
            Xr_base, Lr_base, Kr_base, COr_base, MCOr_base, DCOr_base, XCOr_base, I1r_base, I2r_base, I3r_base,  Irr_base, Ibr_base, Zr_base, Er_base, Dr_base, Qr_base, Mr_base, 
            Xb_base, Lb_base, Kb_base,  NGb_base, MNGb_base, DNGb_base, XNGb_base, I1b_base, I2b_base, I3b_base,  Irb_base,  Ibb_base, Zb_base, Eb_base, Db_base, 
            C1_base, C2_base, C3_base,  Cr_base, Cb_base, Y_base, self.Yd_base, 
            TPAO1_base, TPAO2_base, TPAO3_base,  OIL_INCOME_base, 
            G1_base, G2_base, G3_base,  T_base, Td_base, Tz_base, Tva_base, Tz1_base, Tz2_base, Tz3_base, Tzr_base, Tzb_base, Tva1_base, Tva2_base, Tva3_base,  Tvar_base, Tvab_base, 
            INV1_base, INV2_base, INV3_base, S_base, Sp_base, Sg_base,
            px1_base, px2_base, px3_base, pxr_base, pxb_base, 
            pz1_base, pz2_base, pz3_base, pzr_base, pzb_base, 
            pe1_base, pe2_base, pe3_base, per_base, peb_base, 
            pd1_base, pd2_base, pd3_base, pdr_base, pdb_base, 
            pq1_base, pq2_base, pq3_base, pqr_base, 
            pm1_base, pm2_base, pm3_base, pmr_base, 
            pmco_base, pdco_base, pco_base, pxco_base, 
            pmng_base, pdng_base, png_base, pxng_base, 
            Sf_base, self.r_base ] = self.result1.x

        [   X1_cfrun, L1_cfrun, K1_cfrun, I11_cfrun, I21_cfrun, I31_cfrun, Ir1_cfrun, Ib1_cfrun, Z1_cfrun, E1_cfrun, D1_cfrun, Q1_cfrun, M1_cfrun, 
            X2_cfrun, L2_cfrun, K2_cfrun, I12_cfrun, I22_cfrun, I32_cfrun, Ir2_cfrun, Ib2_cfrun, Z2_cfrun, E2_cfrun, D2_cfrun, Q2_cfrun, M2_cfrun, 
            X3_cfrun, L3_cfrun, K3_cfrun, I13_cfrun, I23_cfrun, I33_cfrun, Ir3_cfrun, Ib3_cfrun, Z3_cfrun, E3_cfrun, D3_cfrun, Q3_cfrun, M3_cfrun, 
            Xr_cfrun, Lr_cfrun, Kr_cfrun, COr_cfrun, MCOr_cfrun, DCOr_cfrun, XCOr_cfrun, I1r_cfrun, I2r_cfrun, I3r_cfrun,  Irr_cfrun, Ibr_cfrun, Zr_cfrun, Er_cfrun, Dr_cfrun, Qr_cfrun, Mr_cfrun, 
            Xb_cfrun, Lb_cfrun, Kb_cfrun,  NGb_cfrun, MNGb_cfrun, DNGb_cfrun, XNGb_cfrun, I1b_cfrun, I2b_cfrun, I3b_cfrun,  Irb_cfrun,  Ibb_cfrun, Zb_cfrun, Eb_cfrun, Db_cfrun, 
            C1_cfrun, C2_cfrun, C3_cfrun,  Cr_cfrun, Cb_cfrun, Y_cfrun, self.Yd_cfrun, 
            TPAO1_cfrun, TPAO2_cfrun, TPAO3_cfrun,  OIL_INCOME_cfrun, 
            G1_cfrun, G2_cfrun, G3_cfrun,  T_cfrun, Td_cfrun, Tz_cfrun, Tva_cfrun, Tz1_cfrun, Tz2_cfrun, Tz3_cfrun, Tzr_cfrun, Tzb_cfrun, Tva1_cfrun, Tva2_cfrun, Tva3_cfrun,  Tvar_cfrun, Tvab_cfrun, 
            INV1_cfrun, INV2_cfrun, INV3_cfrun, S_cfrun, Sp_cfrun, Sg_cfrun,
            px1_cfrun, px2_cfrun, px3_cfrun, pxr_cfrun, pxb_cfrun, 
            pz1_cfrun, pz2_cfrun, pz3_cfrun, pzr_cfrun, pzb_cfrun, 
            pe1_cfrun, pe2_cfrun, pe3_cfrun, per_cfrun, peb_cfrun, 
            pd1_cfrun, pd2_cfrun, pd3_cfrun, pdr_cfrun, pdb_cfrun, 
            pq1_cfrun, pq2_cfrun, pq3_cfrun, pqr_cfrun, 
            pm1_cfrun, pm2_cfrun, pm3_cfrun, pmr_cfrun, 
            pmco_cfrun, pdco_cfrun, pco_cfrun, pxco_cfrun, 
            pmng_cfrun, pdng_cfrun, png_cfrun, pxng_cfrun, 
            Sf_cfrun, self.r_cfrun ] = self.result2.x
        
        
        # Macro Variables
        # ---------------
        self.GDP_base_total = Y_base + OIL_INCOME_base + Tva_base + Tz_base
        self.CES_base_total  = C1_base*pq1_base + C2_base*pq2_base + C3_base*pq3_base + Cr_base*pqr_base + Cb_base*pdb_base
        self.TPAO_base_total = TPAO1_base*pq1_base + TPAO2_base*pq2_base + TPAO3_base*pq3_base
        self.G_base_total   = G1_base*pq1_base + G2_base*pq2_base + G3_base*pq3_base
        self.INV_base_total = INV1_base*pq1_base + INV2_base*pq2_base + INV3_base*pq3_base
        self.E_base_total   = E1_base * pe1_base + E2_base*pe2_base + E3_base*pe3_base + Er_base*per_base + Eb_base*peb_base + self.m1.epsilon * self.m1.E_Energy
        self.M_base_total   = M1_base*pm1_base + M2_base*pm2_base + M3_base*pm3_base + MCOr_base*pmco_base + Mr_base*pmr_base + MNGb_base*pmng_base
        self.Total_base_Consumption =  self.CES_base_total +  self.TPAO_base_total + self.G_base_total  + self.INV_base_total + self.E_base_total - self.M_base_total 
        self.NE_base_total  = self.E_base_total - self.M_base_total
        self.Tva_base_total = Tva1_base + Tva2_base + Tva3_base + Tvar_base + Tvab_base
        self.Tz_base_total  = Tz1_base + Tz2_base + Tz3_base + Tzr_base + Tzb_base
        
        self.GDP_cfrun_total = Y_cfrun + OIL_INCOME_cfrun + Tva_cfrun + Tz_cfrun
        self.CES_cfrun_total = C1_cfrun*pq1_cfrun + C2_cfrun*pq2_cfrun + C3_cfrun*pq3_cfrun + Cr_base*pqr_base + Cb_base*pdb_base
        self.TPAO_cfrun_total = TPAO1_cfrun*pq1_cfrun + TPAO2_cfrun*pq2_cfrun + TPAO3_cfrun*pq3_cfrun
        self.G_cfrun_total   = G1_cfrun*pq1_cfrun + G2_cfrun*pq2_cfrun + G3_cfrun*pq3_cfrun
        self.INV_cfrun_total = INV1_cfrun*pq1_cfrun + INV2_cfrun*pq2_cfrun + INV3_cfrun*pq3_cfrun
        self.E_cfrun_total   = E1_cfrun * pe1_cfrun + E2_cfrun*pe2_cfrun + E3_cfrun*pe3_cfrun + Er_cfrun*per_cfrun + Eb_cfrun*peb_cfrun + self.m2.epsilon * self.m2.E_Energy
        self.M_cfrun_total   = M1_cfrun*pm1_cfrun + M2_cfrun*pm2_cfrun + M3_cfrun*pm3_cfrun + MCOr_cfrun*pmco_cfrun + Mr_cfrun*pmr_cfrun + MNGb_cfrun*pmng_cfrun
        self.Total_cfrun_Consumption =  self.CES_cfrun_total +  self.TPAO_cfrun_total + self.G_cfrun_total  + self.INV_cfrun_total +  self.E_cfrun_total - self.M_cfrun_total 
        self.NE_cfrun_total  = self.E_cfrun_total - self.M_cfrun_total
        self.Tva_cfrun_total = Tva1_cfrun + Tva2_cfrun + Tva3_cfrun + Tvar_cfrun + Tvab_cfrun
        self.Tz_cfrun_total  = Tz1_cfrun + Tz2_cfrun + Tz3_cfrun + Tzr_cfrun + Tzb_cfrun


        self.index = ["GDP Gelir", "Özel Tüketim", "TPAO", "Kamu", "Yatırım", "İhracat", "İthalat", "Toplam Harcama", "Net İhracat", "Ürün Vergi", "Üretim Vergi"]
       
        self.base_values = [self.GDP_base_total,self.CES_base_total, self.TPAO_base_total, self.G_base_total, self.INV_base_total, self.E_base_total, self.M_base_total,  
                            self.Total_base_Consumption, self.NE_base_total, self.Tva_base_total, self.Tz_base_total ]
        
        self.cfrun_values = [self.GDP_cfrun_total,self.CES_cfrun_total, self.TPAO_cfrun_total, self.G_cfrun_total, self.INV_cfrun_total, self.E_cfrun_total, self.M_cfrun_total,  
                            self.Total_cfrun_Consumption, self.NE_cfrun_total, self.Tva_cfrun_total, self.Tz_cfrun_total ]
              
    
        self.HouseholdBaseIncome = Y_base
        self.HouseholdCfrunIncome = Y_cfrun
        self.GovBaseIncome = T_base
        self.GovCfrunIncome = T_cfrun
        self.SavingsBase = S_base
        self.SavingsCfrun = S_cfrun
        self.EIncomeBase =  OIL_INCOME_base
        self.EIncomeCfrun =  OIL_INCOME_cfrun

        self.PriceBase = [px1_base, px2_base, px3_base, pxr_base, pxb_base, pz1_base, pz2_base, pz3_base, pzr_base, pzb_base, pe1_base, pe2_base, pe3_base, per_base, peb_base, 
                          pd1_base, pd2_base, pd3_base, pdr_base, pdb_base, pq1_base, pq2_base, pq3_base, pqr_base, pm1_base, pm2_base, pm3_base, pmr_base, pmco_base,
                          pco_base, pxco_base, pmng_base,png_base, pxng_base, self.r_base ]
        
        self.PriceCfrun = [px1_cfrun, px2_cfrun, px3_cfrun, pxr_cfrun, pxb_cfrun, pz1_cfrun, pz2_cfrun, pz3_cfrun, pzr_cfrun, pzb_cfrun, pe1_cfrun, pe2_cfrun, pe3_cfrun, per_cfrun, peb_cfrun, 
                          pd1_cfrun, pd2_cfrun, pd3_cfrun, pdr_cfrun, pdb_cfrun, pq1_cfrun, pq2_cfrun, pq3_cfrun, pqr_cfrun, pm1_cfrun, pm2_cfrun, pm3_cfrun, pmr_cfrun, pmco_cfrun,
                          pco_cfrun, pxco_cfrun, pmng_cfrun,png_cfrun, pxng_cfrun, self.r_cfrun ]
        

        cfrunQuantityValues = {
        "agr"  : [I11_cfrun, I12_cfrun, I13_cfrun, I1r_cfrun, I1b_cfrun, 0,0,0,0,0,0,0,0,0, C1_cfrun, TPAO1_cfrun, G1_cfrun, INV1_cfrun, 0, E1_cfrun],
        "ser"  : [I21_cfrun, I22_cfrun, I23_cfrun, I2r_cfrun, I2b_cfrun, 0,0,0,0,0,0,0,0,0, C2_cfrun, TPAO2_cfrun, G2_cfrun, INV2_cfrun, 0, E2_cfrun],
        "ind"  : [I31_cfrun, I32_cfrun, I33_cfrun, I3r_cfrun, I3b_cfrun, 0,0,0,0,0,0,0,0,0, C3_cfrun, TPAO3_cfrun, G3_cfrun, INV3_cfrun, 0, E3_cfrun],
        "raf"  : [Ir1_cfrun, Ir2_cfrun, Ir3_cfrun, Irr_cfrun, Irb_cfrun, 0,0,0,0,0,0,0,0,0, Cr_cfrun, 0,0,0, 0, Er_cfrun],
        "bts"  : [Ib1_cfrun, Ib2_cfrun, Ib3_cfrun, Ibr_cfrun, Ibb_cfrun, 0,0,0,0,0,0,0,0,0, Cb_cfrun, 0,0,0, 0, Eb_cfrun],
        "lab"  : [L1_cfrun, L2_cfrun, L3_cfrun, Lr_cfrun, Lb_cfrun,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "cap"  : [K1_cfrun, K2_cfrun, K3_cfrun, Kr_cfrun, Kb_cfrun,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "mco"  : [0,0,0, MCOr_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "dco"  : [0,0,0, DCOr_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "mng"  : [0,0,0,0, MNGb_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "dng"  : [0,0,0,0, DNGb_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "dtax" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,Td_cfrun,0,0,0, 0,0],
        "gtax" : [Tva1_cfrun, Tva2_cfrun, Tva3_cfrun, Tvar_cfrun, Tvab_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "ptax" : [Tz1_cfrun, Tz2_cfrun, Tz3_cfrun, Tzr_cfrun, Tzb_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "hh"   : [0,0,0,0,0, self.m2.Lbar, self.m2.Kbar,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "TPAO" : [0,0,0,0,0,0,0,0,DCOr_cfrun, 0, DNGb_cfrun, 0,0,0,0,0,0,0, self.m2.E_Energy,0],
        "gov"  : [0,0,0,0,0,0,0,0,0,0,0,Td_cfrun, Tva_cfrun, Tz_cfrun, 0,0,0,0, 0,0],
        "sav"  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,Sp_cfrun, 0, Sg_cfrun, 0, 0, Sf_cfrun],
        "eexp"  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0,self.m2.E_Energy],
        "imp"  : [M1_cfrun, M2_cfrun, M3_cfrun, Mr_cfrun,0,0,0,MCOr_cfrun, 0, MNGb_cfrun, 0,0,0,0,0,0,0,0, 0,0]}

        self.CFRunQuantitiySAM = pd.DataFrame(index = SAM.index[0:-1], columns = SAM.columns[0:-1])
        for key in cfrunQuantityValues.keys():
            self.CFRunQuantitiySAM.loc[key] = cfrunQuantityValues[key]

        self.CFRunQuantitiySAM.loc["total"] = self.CFRunQuantitiySAM.sum()
        self.CFRunQuantitiySAM["total"] = self.CFRunQuantitiySAM.sum(axis = 1)
        



        cfrunPriceValues = {
        "agr"  : [pq1_cfrun, pq1_cfrun, pq1_cfrun, pq1_cfrun, pq1_cfrun, 0,0,0,0,0,0,0,0,0, pq1_cfrun, pq1_cfrun, pq1_cfrun, pq1_cfrun,0, pe1_cfrun],
        "ser"  : [pq2_cfrun, pq2_cfrun, pq2_cfrun, pq2_cfrun, pq2_cfrun, 0,0,0,0,0,0,0,0,0, pq2_cfrun, pq2_cfrun, pq2_cfrun, pq2_cfrun,0, pe2_cfrun],
        "ind"  : [pq3_cfrun, pq3_cfrun, pq3_cfrun, pq3_cfrun, pq3_cfrun, 0,0,0,0,0,0,0,0,0, pq3_cfrun, pq3_cfrun, pq3_cfrun, pq3_cfrun,0, pe3_cfrun],
        "raf"  : [pqr_cfrun, pqr_cfrun, pqr_cfrun, pqr_cfrun, pqr_cfrun, 0,0,0,0,0,0,0,0,0, pqr_cfrun, 0,0,0,0, per_cfrun],
        "bts"  : [pdb_cfrun, pdb_cfrun, pdb_cfrun, pdb_cfrun, pdb_cfrun, 0,0,0,0,0,0,0,0,0, pdb_cfrun, 0,0,0,0, peb_cfrun],
        "lab"  : [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "cap"  : [self.r_cfrun, self.r_cfrun, self.r_cfrun, self.r_cfrun, self.r_cfrun,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "mco"  : [0,0,0, pmco_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "dco"  : [0,0,0, pdco_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "mng"  : [0,0,0,0, pmng_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "dng"  : [0,0,0,0, pdng_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "dtax" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
        "gtax" : [1, 1, 1, 1, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "ptax" : [1, 1, 1, 1, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "hh"   : [0,0,0,0,0, 1,self.r_cfrun,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "TPAO" : [0,0,0,0,0,0,0,0,pdco_cfrun, 0, pdng_cfrun, 0,0,0,0,0,0,0,self.m2.epsilon, 0],
        "gov"  : [0,0,0,0,0,0,0,0,0,0,0,1, 1, 1, 0,0,0,0,0,0],
        "sav"  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1, 0, 1, 0,0, self.m2.epsilon],
        "eexp"  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, self.m2.epsilon],
        "imp"  : [pm1_cfrun, pm2_cfrun, pm3_cfrun, pmr_cfrun,0,0,0,pmco_cfrun, 0, pmng_cfrun, 0,0,0,0,0,0,0,0,0,0]}


        cfrunDatabaseValues = {
        "agr"  : [],
        "ser"  : [],
        "ind"  : [],
        "raf"  : [],
        "bts"  : [],
        "lab"  : [],
        "cap"  : [],
        "mco"  : [],
        "dco"  : [],
        "mng"  : [],
        "dng"  : [],
        "dtax" : [],
        "gtax" : [],
        "ptax" : [],
        "hh"   : [],
        "TPAO" : [],
        "gov"  : [],
        "sav"  : [],
        "eexp"  : [],
        "imp"  : []}


        for i in cfrunQuantityValues.keys():
            for j in range(len(cfrunQuantityValues[i])):

                quantity = cfrunQuantityValues[i][j]
                price    = cfrunPriceValues[i][j]
                value = quantity*price

                cfrunDatabaseValues[i].append(value)


        cfrunDatabase = pd.DataFrame(index = SAM.index, columns = SAM.columns[0:-1])

        for i in cfrunDatabaseValues.keys():
            cfrunDatabase.loc[i] = cfrunDatabaseValues[i]

        cfrunDatabase.loc["total"] = cfrunDatabase.sum()
        cfrunDatabase["total"] = cfrunDatabase.sum(axis = 1)

        self.CFSHM = cfrunDatabase
            
        cfrunDatabase.to_excel("SHMCFrun_.xlsx")

    def EV_CV_Calculation(self):

        EV = (self.U_cfrun - self.U_base) / self.U_base * self.Yd_base
        CV = (self.U_cfrun - self.U_base) / self.U_cfrun * self.Yd_cfrun
        

        print("U1: {}\nU2 {}".format(round(self.U_base,2), round(self.U_cfrun,2)))
        print("EV: {}\nCV {}".format(round(EV,2), round(CV,2)))

    def MacroVariables(self):

        df = pd.DataFrame(index = self.index)
        df["Base Year"] = self.base_values
        df["CF Year"] = self.cfrun_values
        df["%Change"] = (np.array(self.cfrun_values) - np.array(self.base_values))/ np.array(self.base_values) * 100
        # print(df)
        # df.to_excel("MacroVariables.xlsx")

        return df
 
    def SAMPlot(self):

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        BaseSHM = SAM
        CFSHM1 = self.CFSHM
        # CFSHM1 = self.CFRunQuantitiySAM

        ChangeSHM1 = pd.DataFrame(index = BaseSHM.index, columns = BaseSHM.columns)

        for row in ChangeSHM1.index:
            for col in ChangeSHM1.columns:
                baseValue = BaseSHM.loc[row, col]
                cfValue = CFSHM1.loc[row,col]

                if isinstance(baseValue, float) and isinstance(cfValue, float) and baseValue != 0:
                    difference = round((cfValue - baseValue )/ baseValue *100, 2)
                    ChangeSHM1.loc[row,col] = difference

                else:
                    if baseValue == 0 and cfValue == 0:
                        ChangeSHM1.loc[row,col] = 0

                    elif baseValue == 0 and cfValue > 1:
                        ChangeSHM1.loc[row,col] = 100
                    
                    else: 
                        ChangeSHM1.loc[row,col] = 0


        color = "teal"
        
        xpos1, ypos1 = np.meshgrid(np.arange(ChangeSHM1.shape[1]), np.arange(ChangeSHM1.shape[0]))
        xpos1 = xpos1.flatten()
        ypos1 = ypos1.flatten()
        zpos1 = np.zeros_like(xpos1)

        dx1= dy1 = 1
        dz1 = ChangeSHM1.values.flatten()
        dz1_ = np.copy(dz1)
        dz1_[dz1_ < 0] = 0

                
        colors = np.where(dz1_ > 0, color, 'none')
        ax1.bar3d(xpos1, ypos1, zpos1, dx1, dy1, dz1_, color=colors, edgecolor ='black')
        ax1.set_xticks(np.arange(ChangeSHM1.shape[1]))
        ax1.set_xticklabels(ChangeSHM1.columns, fontsize = 6, fontweight = "bold", rotation = 120)
        ax1.set_yticks(np.arange(ChangeSHM1.shape[1]))
        ax1.set_yticklabels(ChangeSHM1.columns, fontsize = 6, fontweight = "bold", rotation = 120)
        ax1.set_xlabel('Sütun')
        ax1.set_ylabel('Satır')
        ax1.set_zlabel('% Değişim')
        ax1.set_xlim(xmin = 0)
        ax1.set_ylim(ymin = 0)
        ax1.set_zlim(zmin = 0)
        ax1.set_title("Panel A", fontweight = "bold")
        ax1.grid(False)

        dz2 = np.copy(dz1)
        dz2[dz2 > 0] = 0
        dz2 = dz2*-1

        colors = np.where(dz2 > 0, color, 'none')
        ax2.bar3d(xpos1, ypos1, zpos1, dx1, dy1, dz2, color=colors , edgecolor ='black')
        ax2.set_xticks(np.arange(ChangeSHM1.shape[1]))
        ax2.set_xticklabels(ChangeSHM1.columns, fontsize = 6, fontweight = "bold", rotation = 120)
        ax2.set_yticks(np.arange(ChangeSHM1.shape[1]))
        ax2.set_yticklabels(ChangeSHM1.columns, fontsize = 6, fontweight = "bold", rotation = 120)

        ax2.set_xlabel('Sütun')
        ax2.set_ylabel('Satır', rotation = -45)
        ax2.set_zlabel('% Değişim')
        ax2.set_xlim(xmin = 0)
        ax2.set_ylim(ymin = 0)
        ax2.set_zlim(zmin = 0)
        ax2.set_title("Panel B", fontweight = "bold")
        ax2.grid(False)

        plt.tight_layout()

        plt.show()





result = CGEResults(2)     # 0: Tarım, 1: Ticaret ve Hizmet, 2: Sanayi
result.SAMPlot()
# result.ArmingtonCETPlot()
result.EV_CV_Calculation()
print(result.MacroVariables())

