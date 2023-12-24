from scipy.optimize import minimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import networkx as nx


SAM = pd.read_excel("SHM.xlsx", index_col = "index")



class CGE():
    
    def __init__(self, SAM):  
        self.SAM = SAM
        shm = self.SAM.loc

        # DIŞŞSAl DEĞİŞKENLER

        ## Üretim Faktörleri
        self.DCOBar = shm["dco", "raf"] 
        self.DNGBar = shm["dng", "bts"]
        self.Lbar   = shm["hh", "lab"]
        self.Kbar   = shm["hh", "cap"]
        self.E_Energy = 0

        ## Yabancı Tasarruf # SF Yabancı Dışsal Değişken Değil

        ## Dünya Fiyatları
        self.Pwe1, self.Pwe2, self.Pwe3, self.Pwer, self.Pweb, \
        self.Pwm1, self.Pwm2, self.Pwm3, self.Pwmr, \
        self.Pwmco, self.Pwmng, self.epsilon = np.ones(12)

        ## Dönüşüm esneklikleri
        self.psi1 = 1.12
        self.psi2 = 0.5
        self.psi3 = 0.601
        self.psir = 10
        self.psib = 10 
        
        ## Armington İkame Esneklikleri
        self.sigma1   = 1.81
        self.sigma2   = 0.8
        self.sigma3   = 0.4
        self.sigmar   = 10 # Rafineriler D / M arasındaki ikame petrol ürünleri
        self.sigmaco  = 10 # İthal hampetrol / yurtiçi hampetrol arasındaki ikame
        self.sigmang  = 10 # İthal doğalgaz / yurtiçi doğalgaz arasındaki ikame



        ## CES İkame Esnekliği
        self.omega1 = 0.678
        self.omega2 = 0.4
        self.omega3 = 0.678
        self.omegar = 0.678 # Sanayi ile aynı olduğu düşünülmüştür.
        self.omegab = 0.678 # Sanayi ile aynı olduğu düşünülmüştür.

        self.omegaxco = 0.01 # Kompozit faktör / ham petrol arasındaki ikame
        self.omegaxng = 0.01 # Kompozit faktör / ham petrol arasındaki ikame




        # BAŞLANGIÇ DEĞERLERİ (BAZ TIL DEĞERLERİ)
        ## Fiyatlar
        px1, px2, px3, pxr, pxb, \
        pz1, pz2, pz3, pzr, pzb, \
        pe1, pe2, pe3, per, peb, \
        pd1, pd2, pd3, pdr, pdb, \
        pq1, pq2, pq3, pqr, \
        pm1, pm2, pm3, pmr, pmco, pdco, \
        pco, pxco, pmng, pdng, png, pxng, r, w = np.ones(38)

        ## Diğer Başlangıç (Baz Yıl) Değerleri
        Y = w*self.Lbar + r*self.Kbar
        OIL_INCOME = pdco*self.DCOBar + pdng*self.DNGBar
        L1 = shm["lab", "agr"]
        L2 = shm["lab", "ser"]
        L3 = shm["lab", "ind"]
        Lr = shm["lab", "raf"]
        Lb = shm["lab", "bts"]
        K1 = shm["cap", "agr"]
        K2 = shm["cap", "ser"]
        K3 = shm["cap", "ind"]
        Kr = shm["cap", "raf"]
        Kb = shm["cap", "bts"]
        X1 = L1 + K1
        X2 = L2 + K2
        X3 = L3 + K3
        Xr = Lr + Kr
        Xb = Lb + Kb
        MCOr = shm["mco","raf"]
        DCOr = shm["dco", "raf"]
        COr = MCOr + DCOr
        XCOr = Xr + COr
        MNGb = shm["mng", "bts"]
        DNGb = shm["dng", "bts"]
        NGb = MNGb + DNGb
        XNGb = Xb + NGb

        I11 = shm["agr", "agr"]
        I21 = shm["ser", "agr"]
        I31 = shm["ind", "agr"]
        Ir1 = shm["raf", "agr"]
        Ib1 = shm["bts", "agr"]

        I12 = shm["agr", "ser"]
        I22 = shm["ser", "ser"]
        I32 = shm["ind", "ser"]
        Ir2 = shm["raf", "ser"]
        Ib2 = shm["bts", "ser"]

        I13 = shm["agr", "ind"]
        I23 = shm["ser", "ind"]
        I33 = shm["ind", "ind"]
        Ir3 = shm["raf", "ind"]
        Ib3 = shm["bts", "ind"]

        I1r = shm["agr", "raf"]
        I2r = shm["ser", "raf"]
        I3r = shm["ind", "raf"]
        Irr = shm["raf", "raf"]
        Ibr = shm["bts", "raf"]

        I1b = shm["agr", "bts"]
        I2b = shm["ser", "bts"]
        I3b = shm["ind", "bts"]
        Irb = shm["raf", "bts"]
        Ibb = shm["bts", "bts"]

        Z1 = X1 + I11 + I21 + I31 + Ir1 + Ib1
        Z2 = X2 + I12 + I22 + I32 + Ir2 + Ib2
        Z3 = X3 + I13 + I23 + I33 + Ir3 + Ib3
        Zr = XCOr + I1r + I2r + I3r + Irr + Ibr
        Zb = XNGb + I1b + I2b + I3b + Irb + Ibb

        E1 = shm["agr", "exp"]
        E2 = shm["ser", "exp"]
        E3 = shm["ind", "exp"]
        Er = shm["raf", "exp"]
        Eb = shm["bts", "exp"]
        
        M1 = shm["imp", "agr"]
        M2 = shm["imp", "ser"]
        M3 = shm["imp", "ind"]
        Mr = shm["imp", "raf"]

        Td = shm["dtax", "hh"]

        Tva1 = shm["gtax", "agr"]
        Tva2 = shm["gtax", "ser"]
        Tva3 = shm["gtax", "ind"]
        Tvar = shm["gtax", "raf"]
        Tvab = shm["gtax", "bts"]
        Tva = Tva1 + Tva2 + Tva3 + Tvar + Tvab

        Tz1 = shm["ptax", "agr"]
        Tz2 = shm["ptax", "ser"]
        Tz3 = shm["ptax", "ind"]
        Tzr = shm["ptax", "raf"]
        Tzb = shm["ptax", "bts"]
        Tz = Tz1 + Tz2 + Tz3 +  Tzr + Tzb


        T = Td + Tz + Tva

        D1 = Z1 + (Tva1 + Tz1) - E1
        D2 = Z2 + (Tva2 + Tz2) - E2
        D3 = Z3 + (Tva3 + Tz3) - E3
        Dr = Zr + (Tvar + Tzr) - Er
        Db = Zb + (Tvab + Tzb) - Eb

        C1 = shm["agr", "hh"]
        C2 = shm["ser", "hh"]
        C3 = shm["ind", "hh"]
        Cr = shm["raf", "hh"]
        Cb = shm["bts", "hh"]

        TPAO1 = shm["agr", "TPAO"]
        TPAO2 = shm["ser", "TPAO"]
        TPAO3 = shm["ind", "TPAO"]
      
        G1 = shm["agr", "gov"]
        G2 = shm["ser", "gov"]
        G3 = shm["ind", "gov"]
     
        INV1 = shm["agr", "inv"]
        INV2 = shm["ser", "inv"]
        INV3 = shm["ind", "inv"]
      
        Q1 = C1 + TPAO1 + G1 + INV1 + I11 + I12 + I13 + I1r + I1b
        Q2 = C2 + TPAO2 + G2 + INV2 + I21 + I22 + I23 + I2r + I2b
        Q3 = C3 + TPAO3 + G3 + INV3 + I31 + I32 + I33 + I3r + I3b
        Qr = Cr + Ir1 + Ir2 + Ir3 +  Irr + Irb
        Db = Cb + Ib1 + Ib2 + Ib3 +  Ibr + Ibb

        Sp = shm["sav", "hh"]
        Sg = shm["sav", "gov"]
        Sf = shm["sav", "exp"]  # Sf içsel değişken.....
        S = Sp + Sg + Sf
        Yd = Y - Sp - Td
        
        # PARAMETRELERİN KALİBRASYONU
      
        self.td = Td / Y 
        self.tva1 = Tva1 / Z1
        self.tva2 = Tva2 / Z2
        self.tva3 = Tva3 / Z3
        self.tvar = Tvar / Zr
        self.tvab = Tvab / Zb

        self.tz1 = Tz1 / Z1
        self.tz2 = Tz2 / Z2
        self.tz3 = Tz3 / Z3
        self.tzr = Tzr / Zr
        self.tzb = Tzb / Zb

    
        self.alpha1 = (self.omega1 - 1 ) / self.omega1
        self.alpha2 = (self.omega2 - 1 ) / self.omega2
        self.alpha3 = (self.omega3 - 1 ) / self.omega3
        self.alphar = (self.omegar - 1 ) / self.omegar
        self.alphab = (self.omegab - 1 ) / self.omegab

        self.delta1 = L1**(1-self.alpha1) / (L1**(1-self.alpha1) + K1**(1-self.alpha1))
        self.delta2 = L2**(1-self.alpha2) / (L2**(1-self.alpha2) + K2**(1-self.alpha2))
        self.delta3 = L3**(1-self.alpha3) / (L3**(1-self.alpha3) + K3**(1-self.alpha3))
        self.deltar = Lr**(1-self.alphar) / (Lr**(1-self.alphar) + Kr**(1-self.alphar))
        self.deltab = Lb**(1-self.alphab) / (Lb**(1-self.alphab) + Kb**(1-self.alphab))

        self.beta1 = K1**(1-self.alpha1) / (L1**(1-self.alpha1) + K1**(1-self.alpha1))
        self.beta2 = K2**(1-self.alpha2) / (L2**(1-self.alpha2) + K2**(1-self.alpha2))
        self.beta3 = K3**(1-self.alpha3) / (L3**(1-self.alpha3) + K3**(1-self.alpha3))
        self.betar = Kr**(1-self.alphar) / (Lr**(1-self.alphar) + Kr**(1-self.alphar))
        self.betab = Kb**(1-self.alphab) / (Lb**(1-self.alphab) + Kb**(1-self.alphab))

        self.A1 = X1 / (self.delta1*L1**self.alpha1 + self.beta1*K1**self.alpha1)**(1/self.alpha1)
        self.A2 = X2 / (self.delta2*L2**self.alpha2 + self.beta2*K2**self.alpha2)**(1/self.alpha2)
        self.A3 = X3 / (self.delta3*L3**self.alpha3 + self.beta3*K3**self.alpha3)**(1/self.alpha3)
        self.Ar = Xr / (self.deltar*Lr**self.alphar + self.betar*Kr**self.alphar)**(1/self.alphar)
        self.Ab = Xb / (self.deltab*Lb**self.alphab + self.betab*Kb**self.alphab)**(1/self.alphab)

        self.alphaxco = (self.omegaxco - 1) / self.omegaxco
        self.xr   = Xr**(1-self.alphaxco) / (Xr**(1-self.alphaxco) + COr**(1-self.alphaxco))
        self.co   = COr**(1-self.alphaxco) / (Xr**(1-self.alphaxco) + COr**(1-self.alphaxco))
        self.Axco = XCOr / (self.xr*Xr**self.alphaxco + self.co*COr**self.alphaxco)**(1/self.alphaxco)

        self.alphaxng = (self.omegaxng - 1) / self.omegaxng
        self.xb   = Xb**(1-self.alphaxng) / (Xb**(1-self.alphaxng) + NGb**(1-self.alphaxng))
        self.ng   = NGb**(1-self.alphaxng) / (Xb**(1-self.alphaxng) + NGb**(1-self.alphaxng))
        self.Axng = XNGb / (self.xb*Xb**self.alphaxng + self.ng*NGb**self.alphaxng)**(1/self.alphaxng)

        self.a11 = I11 / Z1
        self.a21 = I21 / Z1
        self.a31 = I31 / Z1
        self.ar1 = Ir1 / Z1
        self.ab1 = Ib1 / Z1

        self.a12 = I12 / Z2
        self.a22 = I22 / Z2
        self.a32 = I32 / Z2
        self.ar2 = Ir2 / Z2
        self.ab2 = Ib2 / Z2

        self.a13 = I13 / Z3
        self.a23 = I23 / Z3
        self.a33 = I33 / Z3
        self.ar3 = Ir3 / Z3
        self.ab3 = Ib3 / Z3
      
        self.a1r = I1r / Zr
        self.a2r = I2r / Zr
        self.a3r = I3r / Zr
        self.arr = Irr / Zr
        self.abr = Ibr / Zr

        self.a1b = I1b / Zb
        self.a2b = I2b / Zb
        self.a3b = I3b / Zb
        self.arb = Irb / Zb
        self.abb = Ibb / Zb

        self.x1 = X1 / Z1
        self.x2 = X2 / Z2
        self.x3 = X3 / Z3
       
        self.xcor = XCOr / Zr
        self.xngb = XNGb / Zb

        self.rho1 = (self.psi1 + 1) / self.psi1
        self.rho2 = (self.psi2 + 1) / self.psi2
        self.rho3 = (self.psi3 + 1) / self.psi3
        self.rhor = (self.psir + 1) / self.psir
        self.rhob = (self.psib + 1) / self.psib

        self.eta1 = (self.sigma1 - 1) / self.sigma1
        self.eta2 = (self.sigma2 - 1) / self.sigma2
        self.eta3 = (self.sigma3 - 1) / self.sigma3
        self.etar = (self.sigmar - 1) / self.sigmar
        self.etaco = (self.sigmaco - 1) / self.sigmaco
        self.etang = (self.sigmang - 1) / self.sigmang

        self.e1 = E1**(1-self.rho1) / (E1**(1-self.rho1) + D1**(1-self.rho1))
        self.e2 = E2**(1-self.rho2) / (E2**(1-self.rho2) + D2**(1-self.rho2))
        self.e3 = E3**(1-self.rho3) / (E3**(1-self.rho3) + D3**(1-self.rho3))
        self.er = Er**(1-self.rhor) / (Er**(1-self.rhor) + Dr**(1-self.rhor))
        self.eb = Eb**(1-self.rhob) / (Eb**(1-self.rhob) + Db**(1-self.rhob))

        self.dt1 = D1**(1-self.rho1) / (E1**(1-self.rho1) + D1**(1-self.rho1))
        self.dt2 = D2**(1-self.rho2) / (E2**(1-self.rho2) + D2**(1-self.rho2))
        self.dt3 = D3**(1-self.rho3) / (E3**(1-self.rho3) + D3**(1-self.rho3))
        self.dtr = Dr**(1-self.rhor) / (Er**(1-self.rhor) + Dr**(1-self.rhor))
        self.dtb = Db**(1-self.rhob) / (Eb**(1-self.rhob) + Db**(1-self.rhob))

        self.theta1 = Z1 / (self.e1*E1**self.rho1 + self.dt1*D1**self.rho1)**(1/self.rho1)
        self.theta2 = Z2 / (self.e2*E2**self.rho2 + self.dt2*D2**self.rho2)**(1/self.rho2)
        self.theta3 = Z3 / (self.e3*E3**self.rho3 + self.dt3*D3**self.rho3)**(1/self.rho3)
        self.thetar = Zr / (self.er*Er**self.rhor + self.dtr*Dr**self.rhor)**(1/self.rhor)
        self.thetab = Zb / (self.eb*Eb**self.rhob + self.dtb*Db**self.rhob)**(1/self.rhob)

        self.m1   = M1**(1-self.eta1) / (M1**(1-self.eta1) + D1**(1-self.eta1))
        self.m2   = M2**(1-self.eta2) / (M2**(1-self.eta2) + D2**(1-self.eta2))
        self.m3   = M3**(1-self.eta3) / (M3**(1-self.eta3) + D3**(1-self.eta3))
        self.mr   = Mr**(1-self.etar) / (Mr**(1-self.etar) + Dr**(1-self.etar))
        self.mcor = MCOr**(1-self.etaco) / (MCOr**(1-self.etaco) + DCOr**(1-self.etaco))
        self.mngb = MNGb**(1-self.etang) / (MNGb**(1-self.etang) + DNGb**(1-self.etang))

        self.m1   = M1**(1-self.eta1) / (M1**(1-self.eta1) + D1**(1-self.eta1))
        self.m2   = M2**(1-self.eta2) / (M2**(1-self.eta2) + D2**(1-self.eta2))
        self.m3   = M3**(1-self.eta3) / (M3**(1-self.eta3) + D3**(1-self.eta3))
        self.mr   = Mr**(1-self.etar) / (Mr**(1-self.etar) + Dr**(1-self.etar))
        self.mcor = MCOr**(1-self.etaco) / (MCOr**(1-self.etaco) + DCOr**(1-self.etaco))
        self.mngb = MNGb**(1-self.etang) / (MNGb**(1-self.etang) + DNGb**(1-self.etang))

        self.da1   = D1**(1-self.eta1) / (M1**(1-self.eta1) + D1**(1-self.eta1))
        self.da2   = D2**(1-self.eta2) / (M2**(1-self.eta2) + D2**(1-self.eta2))
        self.da3   = D3**(1-self.eta3) / (M3**(1-self.eta3) + D3**(1-self.eta3))
        self.dar   = Dr**(1-self.etar) / (Mr**(1-self.etar) + Dr**(1-self.etar))
        self.dcor  = DCOr**(1-self.etaco) / (MCOr**(1-self.etaco) + DCOr**(1-self.etaco))
        self.dngb  = DNGb**(1-self.etang) / (MNGb**(1-self.etang) + DNGb**(1-self.etang))

        self.lambda1 = Q1 / (self.m1*M1**self.eta1 + self.da1*D1**self.eta1)**(1/self.eta1)
        self.lambda2 = Q2 / (self.m2*M2**self.eta2 + self.da2*D2**self.eta2)**(1/self.eta2)
        self.lambda3 = Q3 / (self.m3*M3**self.eta3 + self.da3*D3**self.eta3)**(1/self.eta3)
        self.lambdar = Qr / (self.mr*Mr**self.etar + self.dar*Dr**self.etar)**(1/self.etar)
        self.lambdaco = COr / (self.mcor*MCOr**self.etaco + self.dcor*DCOr**self.etaco)**(1/self.etaco)
        self.lambdang = NGb / (self.mngb*MNGb**self.etang + self.dngb*DNGb**self.etang)**(1/self.etang)

        self.c1 = C1 / Yd
        self.c2 = C2 / Yd
        self.c3 = C3 / Yd
        self.cr = Cr / Yd
        self.cb = Cb / Yd

        self.mu1 = TPAO1 / OIL_INCOME 
        self.mu2 = TPAO2 / OIL_INCOME 
        self.mu3 = TPAO3 / OIL_INCOME 
       
        self.g1 = G1 / (T - Sg)
        self.g2 = G2 / (T - Sg)
        self.g3 = G3 / (T - Sg)
       
        self.inv1 = INV1 / S
        self.inv2 = INV2 / S
        self.inv3 = INV3 / S
        
        self.sp = Sp / Y
        self.sg = Sg / T
        
        self.init_values =  [
            X1, L1, K1, I11, I21, I31, Ir1, Ib1, Z1,
            E1, D1, Q1, M1, X2, L2, K2, I12, I22, I32, 
            Ir2, Ib2, Z2, E2, D2, Q2, M2, X3, L3, K3, 
            I13, I23, I33, Ir3, Ib3, Z3, E3, D3, Q3, 
            M3, Xr, Lr, Kr, COr, MCOr, DCOr, XCOr, I1r, I2r, 
            I3r,  Irr, Ibr, Zr, Er, Dr, Qr, Mr, Xb, Lb, Kb, 
            NGb, MNGb, DNGb, XNGb, I1b, I2b, I3b,  Irb, 
            Ibb, Zb, Eb, Db, C1, C2, C3,  Cr, Cb, Y,
            Yd, TPAO1, TPAO2, TPAO3,  OIL_INCOME, G1, G2, 
            G3,  T, Td, Tz, Tva, Tz1, Tz2, Tz3, 
            Tzr, Tzb, Tva1, Tva2, Tva3,  Tvar, 
            Tvab, INV1, INV2, INV3,  
             S, Sp, Sg, px1, px2, px3,  pxr, pxb, 
            pz1, pz2, pz3, pzr, pzb, pe1, pe2, pe3, 
             per, peb, pd1, pd2, pd3, pdr, pdb, 
            pq1, pq2, pq3,  pqr, pm1, pm2, pm3, 
            pmr, pmco, pdco, pco, pxco, pmng, pdng, png, pxng, 
            Sf, r 
        ]
        
        self.init_values_str = [
            "X1", "L1", "K1", "I11", "I21", "I31", "Ir1", "Ib1", "Z1",
            "E1", "D1", "Q1", "M1", "X2", "L2", "K2", "I12", "I22", "I32", 
            "Ir2", "Ib2", "Z2", "E2", "D2", "Q2", "M2", "X3", "L3", "K3", 
            "I13", "I23", "I33", "Ir3", "Ib3", "Z3", "E3", "D3", "Q3", 
            "M3", "Xr", "Lr", "Kr", "COr", "MCOr", "DCOr", "XCOr", "I1r", "I2r", 
            "I3r",  "Irr", "Ibr", "Zr", "Er", "Dr", "Qr", "Mr", "Xb", "Lb", "Kb", 
            "NGb", "MNGb", "DNGb", "XNGb", "I1b", "I2b", "I3b",  "Irb", 
            "Ibb", "Zb", "Eb", "Db", "C1", "C2", "C3",  "Cr", "Cb", "Y",
            "Yd", "TPAO1", "TPAO2", "TPAO3",  "OIL_INCOME", "G1", "G2", 
            "G3",  "T", "Td", "Tz", "Tva",  "Tz1", "Tz2", "Tz3", 
            "Tzr", "Tzb", "Tva1", "Tva2", "Tva3",  "Tvar", 
            "Tvab", "INV1", "INV2", "INV3",  
             "S", "Sp", "Sg", "px1", "px2", "px3",  "pxr", "pxb", 
            "pz1", "pz2", "pz3", "pzr", "pzb", "pe1", "pe2", "pe3", 
            "per", "peb", "pd1", "pd2", "pd3", "pdr", "pdb", 
            "pq1", "pq2", "pq3",  "pqr", "pm1", "pm2", "pm3", 
            "pmr", "pmco", "pdco", "pco", "pxco", "pmng", "pdng", "png", "pxng", 
            "Sf", "r"    
        ]
        
        self.parameters_str=[
                "td", "tva1", "tva2", "tva3", "tvar", "tvab", "tz1", "tz2", "tz3", "tzr", "tzb",
                "alpha1", "alpha2", "alpha3", "alphar","alphab",
                "delta1", "delta2", "delta3", "deltar", "deltab", 
                "beta1", "beta2", "beta3", "betar", "betab",
                "A1", "A2", "A3", "Ar", "Ab",
                "Axco", "alphaxco", "xr", "co",
                "Axng", "alphaxng", "xb", "ng", 
                "a11", "a21", "a31", "ar1", "ab1", "a12", "a22", "a32", "ar2", "ab2", "a13", "a23",
                "a33", "ar3", "ab3", "a1r", "a2r", "a3r", "arr", "abr", "a1b", "a2b", "a3b", "arb", "abb", "x1", "x2",
                "x3", "xcor", "xngb", "rho1", "rho2", "rho3", "rhor", "rhob", "eta1", "eta2", "eta3", "etar", "etaco",
                "etang", "e1", "e2", "e3", "er", "eb", "dt1", "dt2", "dt3", "dtr", "dtb", "theta1", "theta2", "theta3",
                "thetar", "thetab", "m1", "m2", "m3", "mr", "mcor", "mngb", "m1", "m2", "m3", "mr", "mcor", "mngb", "da1",
                "da2", "da3", "dar", "dcor", "dngb", "lambda1", "lambda2", "lambda3", "lambdar", "lambdaco", "lambdang",
                "c1", "c2", "c3", "cr", "cb", "mu1", "mu2", "mu3", "g1", "g2", "g3", "inv1", "inv2", "inv3", "sp", "sg"
        ]
        
    def model_parameters(self):
        
        self.parameters=[
                self.td, self.tva1, self.tva2, self.tva3, self.tvar, self.tvab, self.tz1, self.tz2, self.tz3, self.tzr, self.tzb,
                self.alpha1, self.alpha2, self.alpha3, self.alphar,self.alphab, self.delta1, self.delta2,
                self.delta3, self.deltar, self.deltab, self.beta1, self.beta2, self.beta3,self.betar,self.betab, self.A1, self.A2, self.A3, 
                self.Ar, self.Ab, self.Axco, self.alphaxco, self.xr, self.co, self.Axng, self.alphaxng, self.xb, self.ng, 
                self.a11, self.a21, self.a31, self.ar1, self.ab1, self.a12, self.a22, self.a32, self.ar2, self.ab2, self.a13, self.a23,
                self.a33, self.ar3, self.ab3, self.a1r, self.a2r, self.a3r, self.arr, self.abr, self.a1b, self.a2b, self.a3b, self.arb, 
                self.abb, self.x1, self.x2, self.x3, self.xcor, self.xngb, self.rho1, self.rho2, self.rho3, self.rhor, self.rhob, self.eta1, 
                self.eta2, self.eta3, self.etar, self.etaco, self.etang, self.e1, self.e2, self.e3, self.er, self.eb, self.dt1, self.dt2, self.dt3, 
                self.dtr, self.dtb, self.theta1, self.theta2, self.theta3, self.thetar, self.thetab, self.m1, self.m2, self.m3, self.mr, self.mcor, 
                self.mngb, self.m1, self.m2, self.m3, self.mr, self.mcor, self.mngb, self.da1, self.da2, self.da3, self.dar, self.dcor, self.dngb, 
                self.lambda1, self.lambda2, self.lambda3, self.lambdar, self.lambdaco, self.lambdang, self.c1, self.c2, self.c3, self.cr, self.cb, 
                self.mu1, self.mu2, self.mu3, self.g1, self.g2, self.g3, self.inv1, self.inv2, self.inv3, self.sp, self.sg
            ]

        return self.parameters
        
    def objValue(self, x):
        
        C1 = x[71]
        C2 = x[72]
        C3 = x[73]
        Cr = x[74]
        Cb = x[75]

        return -C1**self.c1 * C2**self.c2 * C3**self.c3 * Cr**self.cr * Cb**self.cb
        
    def constraints(self, x):
        X1  = x[0]
        L1  = x[1]
        K1  = x[2]
        I11 = x[3]
        I21 = x[4]
        I31 = x[5]
        Ir1 = x[6]
        Ib1 = x[7]
        Z1  = x[8]
        E1  = x[9]
        D1  = x[10]
        Q1  = x[11]
        M1  = x[12]
        X2  = x[13]
        L2  = x[14]
        K2  = x[15]
        I12 = x[16]
        I22 = x[17]
        I32 = x[18]
        Ir2 = x[19]
        Ib2 = x[20]
        Z2  = x[21]
        E2  = x[22]
        D2  = x[23]
        Q2  = x[24]
        M2  = x[25]
        X3  = x[26]
        L3  = x[27]
        K3  = x[28]
        I13 = x[29]
        I23 = x[30]
        I33 = x[31]
        Ir3 = x[32]
        Ib3 = x[33]
        Z3  = x[34]
        E3  = x[35]
        D3  = x[36]
        Q3  = x[37]
        M3  = x[38]
        Xr   = x[39]
        Lr   = x[40]
        Kr   = x[41]
        COr  = x[42]
        MCOr = x[43]
        DCOr = x[44]
        XCOr = x[45]
        I1r  = x[46]
        I2r  = x[47]
        I3r  = x[48]
        Irr  = x[49]
        Ibr  = x[50]
        Zr   = x[51]
        Er   = x[52]
        Dr   = x[53]
        Qr   = x[54]
        Mr   = x[55]
        Xb   = x[56]
        Lb   = x[57]
        Kb   = x[58]
        NGb  = x[59]
        MNGb = x[60]
        DNGb = x[61]
        XNGb = x[62]
        I1b = x[63]
        I2b = x[64]
        I3b = x[65]
        Irb = x[66]
        Ibb = x[67]
        Zb  = x[68]
        Eb  = x[69]
        Db  = x[70]
        C1 = x[71]
        C2 = x[72]
        C3 = x[73]
        Cr = x[74]
        Cb = x[75]
        Y = x[76]
        Yd  = x[77]
        TPAO1 = x[78]
        TPAO2 = x[79]
        TPAO3 = x[80]
        OIL_INCOME = x[81]
        G1   = x[82]
        G2   = x[83]
        G3   = x[84]
        T    = x[85]
        Td   = x[86]
        Tz   = x[87]
        Tva  = x[88]
        Tz1  = x[89]
        Tz2  = x[90]
        Tz3  = x[91]
        Tzr  = x[92]
        Tzb  = x[93]
        Tva1 = x[94]
        Tva2 = x[95]
        Tva3 = x[96]
        Tvar = x[97]
        Tvab = x[98]
        INV1 = x[99]
        INV2 = x[100]
        INV3 = x[101]
        S    = x[102]
        Sp   = x[103]
        Sg   = x[104]
        px1  = x[105]
        px2  = x[106]
        px3  = x[107]
        pxr  = x[108]
        pxb  = x[109]
        pz1  = x[110]
        pz2  = x[111]
        pz3  = x[112]
        pzr  = x[113]
        pzb  = x[114]
        pe1  = x[115]
        pe2  = x[116]
        pe3  = x[117]
        per  = x[118]
        peb  = x[119]
        pd1  = x[120]
        pd2  = x[121]
        pd3  = x[122]
        pdr  = x[123]
        pdb  = x[124]
        pq1  = x[125]
        pq2  = x[126]
        pq3  = x[127]
        pqr  = x[128]
        pm1  = x[129]
        pm2  = x[130]
        pm3  = x[131]
        pmr  = x[132]
        pmco = x[133]
        pdco = x[134]
        pco  = x[135]
        pxco = x[136]
        pmng = x[137]
        pdng = x[138]
        png  = x[139]
        pxng = x[140]
        Sf = x[141]
        r    = x[142]
        w    = 1
        
        return [
        X1 - self.A1*(self.delta1*L1**self.alpha1 + self.beta1*K1**self.alpha1)**(1/self.alpha1),
        L1 - px1 * X1 / (w + r*(self.delta1*r/(self.beta1*w))**(1/(self.alpha1-1))),
        K1 - px1 * X1 / (r + w*(self.beta1*w /(self.delta1*r))**(1/(self.alpha1-1))),
        I11 - self.a11*Z1,
        I21 - self.a21*Z1,
        I31 - self.a31*Z1,
        Ir1 - self.ar1*Z1,
        Ib1 - self.ab1*Z1,
        X1  - self.x1*Z1,
        pz1 - (px1*self.x1 + self.a11*pq1 + self.a21*pq2 + self.a31*pq3 +  self.ar1*pqr + self.ab1*pdb),
        Z1  - self.theta1 * (self.e1*E1**self.rho1 + self.dt1*D1**self.rho1)**(1/self.rho1),
        E1  - (self.theta1 ** self.rho1 * self.e1 * (1+self.tz1 + self.tva1)*pz1 / pe1 )**(1/(1-self.rho1))*Z1,
        D1  - (self.theta1 ** self.rho1 * self.dt1 * (1+self.tz1 + self.tva1)*pz1 / pd1 )**(1/(1-self.rho1))*Z1,
        Q1  - self.lambda1*(self.m1*M1**self.eta1 + self.da1*D1**self.eta1)**(1/self.eta1),
        M1  - (self.lambda1**self.eta1 * self.m1  *pq1 / (pm1))**(1/(1-self.eta1))*Q1,
        D1  - (self.lambda1**self.eta1 * self.da1 *pq1 / pd1)**(1/(1-self.eta1))*Q1,
        X2 - self.A2*(self.delta2*L2**self.alpha2 + self.beta2*K2**self.alpha2)**(1/self.alpha2),
        L2 - px2 * X2 / (w + r*(self.delta2*r/(self.beta2*w))**(1/(self.alpha2-1))),
        K2 - px2 * X2 / (r + w*(self.beta2*w /(self.delta2*r))**(1/(self.alpha2-1))),
        I12 - self.a12*Z2,
        I22 - self.a22*Z2,
        I32 - self.a32*Z2,
        Ir2 - self.ar2*Z2,
        Ib2 - self.ab2*Z2,
        X2  - self.x2*Z2,
        pz2 - (px2*self.x2 + self.a12*pq1 + self.a22*pq2 + self.a32*pq3 + self.ar2*pqr + self.ab2*pdb),
        Z2  - self.theta2 * (self.e2*E2**self.rho2 + self.dt2*D2**self.rho2)**(1/self.rho2),
        E2  - (self.theta2 ** self.rho2 * self.e2 * (1+self.tz2 + self.tva2)*pz2 / pe2 )**(1/(1-self.rho2))*Z2,
        D2  - (self.theta2 ** self.rho2 * self.dt2 * (1+self.tz2 + self.tva2)*pz2 / pd2 )**(1/(1-self.rho2))*Z2,
        Q2  - self.lambda2*(self.m2*M2**self.eta2 + self.da2*D2**self.eta2)**(1/self.eta2),
        M2  - (self.lambda2**self.eta2 * self.m2  *pq2 / (pm2))**(1/(1-self.eta2))*Q2,
        D2  - (self.lambda2**self.eta2 * self.da2 *pq2 / pd2)**(1/(1-self.eta2))*Q2,
        X3 - self.A3*(self.delta3*L3**self.alpha3 + self.beta3*K3**self.alpha3)**(1/self.alpha3),
        L3 - px3 * X3 / (w + r*(self.delta3*r/(self.beta3*w))**(1/(self.alpha3-1))),
        K3 - px3 * X3 / (r + w*(self.beta3*w /(self.delta3*r))**(1/(self.alpha3-1))),
        I13 - self.a13*Z3,
        I23 - self.a23*Z3,
        I33 - self.a33*Z3,
        Ir3 - self.ar3*Z3,
        Ib3 - self.ab3*Z3,
        X3  - self.x3*Z3,
        pz3 - (px3*self.x3 + self.a13*pq1 + self.a23*pq2 + self.a33*pq3 + self.ar3*pqr + self.ab3*pdb),
        Z3  - self.theta3 * (self.e3*E3**self.rho3 + self.dt3*D3**self.rho3)**(1/self.rho3),
        E3  - (self.theta3 ** self.rho3 * self.e3 * (1+self.tz3 + self.tva3)*pz3 / pe3 )**(1/(1-self.rho3))*Z3,
        D3  - (self.theta3 ** self.rho3 * self.dt3 * (1+self.tz3 + self.tva3)*pz3 / pd3 )**(1/(1-self.rho3))*Z3,
        Q3  - self.lambda3*(self.m3*M3**self.eta3 + self.da3*D3**self.eta3)**(1/self.eta3),
        M3  - (self.lambda3**self.eta3 * self.m3  *pq3 / (pm3))**(1/(1-self.eta3))*Q3,
        D3  - (self.lambda3**self.eta3 * self.da3 *pq3 / pd3)**(1/(1-self.eta3))*Q3,
        Xr - self.Ar*(self.deltar*Lr**self.alphar + self.betar*Kr**self.alphar)**(1/self.alphar),
        Lr - pxr * Xr / (w + r*(self.deltar*r/(self.betar*w))**(1/(self.alphar-1))),
        Kr - pxr * Xr / (r + w*(self.betar*w /(self.deltar*r))**(1/(self.alphar-1))),
        COr  - self.lambdaco*(self.mcor*MCOr**self.etaco + self.dcor*DCOr**self.etaco)**(1/self.etaco),
        MCOr - (self.lambdaco**self.etaco*self.mcor*pco/pmco)**(1 / (1-self.etaco)) * COr,
        DCOr - (self.lambdaco**self.etaco*self.dcor*pco/pdco)**(1 / (1-self.etaco)) * COr,
        XCOr - self.Axco * (self.xr*Xr**self.alphaxco + self.co*COr**self.alphaxco)**(1/self.alphaxco),
        Xr - (self.Axco**self.alphaxco * self.xr * pxco  / pxr) ** (1 / (1-self.alphaxco)) * XCOr,
        COr - (self.Axco**self.alphaxco * self.co * pxco  / pco) ** (1 / (1-self.alphaxco)) * XCOr,
        I1r  - self.a1r*Zr,
        I2r  - self.a2r*Zr,
        I3r  - self.a3r*Zr,
        Irr  - self.arr*Zr,
        Ibr  - self.abr*Zr,
        XCOr - self.xcor*Zr,
        pzr  - (pxco*self.xcor + self.a1r*pq1 + self.a2r*pq2 + self.a3r*pq3 + self.arr*pqr + self.abr*pdb),
        Zr   - self.thetar * (self.er*Er**self.rhor + self.dtr*Dr**self.rhor)**(1/self.rhor),
        Er   - (self.thetar ** self.rhor * self.er * (1+self.tzr + self.tvar)*pzr / per )**(1/(1-self.rhor))*Zr,
        Dr   - (self.thetar ** self.rhor * self.dtr * (1+self.tzr + self.tvar)*pzr / pdr )**(1/(1-self.rhor))*Zr,
        Qr   - self.lambdar*(self.mr*Mr**self.etar + self.dar*Dr**self.etar)**(1/self.etar),
        Mr   - (self.lambdar**self.etar * self.mr  *pqr / pmr)**(1/(1-self.etar))*Qr,
        Dr   - (self.lambdar**self.etar * self.dar *pqr / pdr)**(1/(1-self.etar))*Qr,
        Xb - self.Ab*(self.deltab*Lb**self.alphab + self.betab*Kb**self.alphab)**(1/self.alphab),
        Lb - pxb * Xb / (w + r*(self.deltab*r/(self.betab*w))**(1/(self.alphab-1))),
        Kb - pxb * Xb / (r + w*(self.betab*w /(self.deltab*r))**(1/(self.alphab-1))),
        NGb  - self.lambdang * (self.mngb*MNGb**self.etang + self.dngb*DNGb**self.etang)**(1/self.etang),
        MNGb - (self.lambdang**self.etang * self.mngb*png / pmng)**(1/(1-self.etang)) * NGb,
        DNGb - (self.lambdang**self.etang * self.dngb * png / pdng)**(1/(1-self.etang)) * NGb,
        XNGb - self.Axng * (self.xb*Xb**self.alphaxng + self.ng*NGb**self.alphaxng)**(1/self.alphaxng),
        Xb - (self.Axng**self.alphaxng * self.xb * pxng  / pxb) ** (1 / (1-self.alphaxng)) * XNGb,
        NGb - (self.Axng**self.alphaxng * self.ng * pxng  / png) ** (1 / (1-self.alphaxng)) * XNGb,
        I1b  - self.a1b*Zb,
        I2b  - self.a2b*Zb,
        I3b  - self.a3b*Zb,
        Irb  - self.arb*Zb,
        Ibb  - self.abb*Zb,
        XNGb - self.xngb*Zb,
        pzb  - (pxng*self.xngb + self.a1b*pq1 + self.a2b*pq2 + self.a3b*pq3 + self.arb*pqr + self.abb*pdb),
        Zb   - self.thetab * (self.eb*Eb**self.rhob + self.dtb*Db**self.rhob)**(1/self.rhob),
        Eb   - (self.thetab ** self.rhob * self.eb  * (1+self.tzb + self.tvab) *pzb / peb )**(1/(1-self.rhob))*Zb,
        Db   - (self.thetab ** self.rhob * self.dtb * (1+self.tzb + self.tvab) *pzb / pdb )**(1/(1-self.rhob))*Zb,
        C1 - self.c1 / pq1 * Yd,
        C2 - self.c2 / pq2 * Yd,
        C3 - self.c3 / pq3 * Yd,
        Cr - self.cr / pqr * Yd,
        Cb - self.cb / pdb * Yd,
        Yd - (Y - Sp - Td),
        Y  - (w*self.Lbar + r*self.Kbar),
        TPAO1 - self.mu1 / pq1 * OIL_INCOME,
        TPAO2 - self.mu2 / pq2 * OIL_INCOME,
        TPAO3 - self.mu3 / pq3 * OIL_INCOME,
        OIL_INCOME - (pdco * self.DCOBar + pdng*self.DNGBar + self.epsilon*self.E_Energy),
        G1   - self.g1 / pq1 * (T-Sg),
        G2   - self.g2 / pq2 * (T-Sg),
        G3   - self.g3 / pq3 * (T-Sg),
        T    - (Td + Tz + Tva),
        Td   - self.td*Y,
        Tz   - (Tz1 + Tz2 + Tz3 + Tzr + Tzb),
        Tz1  - self.tz1 * pz1 * Z1,
        Tz2  - self.tz2 * pz2 * Z2,
        Tz3  - self.tz3 * pz3 * Z3,
        Tzr  - self.tzr * pzr * Zr,
        Tzb  - self.tzb * pzb * Zb,
        Tva  - (Tva1 + Tva2 + Tva3 +  Tvar + Tvab), 
        Tva1 - self.tva1 * pz1 * Z1,
        Tva2 - self.tva2 * pz2 * Z2,
        Tva3 - self.tva3 * pz3 * Z3,
        Tvar - self.tvar * pzr * Zr,
        Tvab - self.tvab * pzb * Zb,
        INV1 - self.inv1 / pq1 * S,
        INV2 - self.inv2 / pq2 * S,
        INV3 - self.inv3 / pq3 * S,
        S    - (Sp + Sg + Sf*self.epsilon),
        Sp   - self.sp*Y,
        Sg   - self.sg*T,
        pe1 - self.epsilon * self.Pwe1,
        pe2 - self.epsilon * self.Pwe2,
        pe3 - self.epsilon * self.Pwe3,
        per - self.epsilon * self.Pwer,
        peb - self.epsilon * self.Pweb,
        pm1 - self.epsilon * self.Pwm1,
        pm2 - self.epsilon * self.Pwm2,
        pm3 - self.epsilon * self.Pwm3,
        pmr - self.epsilon * self.Pwmr,
        pmco - self.epsilon * self.Pwmco,
        pmng - self.epsilon * self.Pwmng,
        self.Pwe1*E1 + self.Pwe2*E2 + self.Pwe3*E3 +  self.Pwer*Er + self.Pweb*Eb + Sf + self.E_Energy - (self.Pwm1*M1 + self.Pwm2*M2 + self.Pwm3*M3 +  self.Pwmr*Mr + self.Pwmco * MCOr + self.Pwmng * MNGb),
        Q1 - (C1 + TPAO1 + G1 + INV1 + I11 + I12 + I13 +  I1r + I1b),
        Q2 - (C2 + TPAO2 + G2 + INV2 + I21 + I22 + I23 +  I2r + I2b),
        # Q3 - (C3 + TPAO3 + G3 + INV3 + I31 + I32 + I33 +  I3r + I3b),
        Qr - (Cr + Ir1 + Ir2 + Ir3 + Irr + Irb),
        Db - (Cb + Ib1 + Ib2 + Ib3 + Ibr + Ibb),
        self.Lbar - (L1 + L2 + L3 + Lr + Lb),
        self.Kbar - (K1 + K2 + K3 + Kr + Kb),
        self.DCOBar - DCOr,
        self.DNGBar - DNGb,

        ]
    
    def SolveModel(self):
        
        cons = {"type":"eq", "fun":self.constraints}
        x0 = self.init_values
        bnds = []

        for val in self.init_values_str:
            if val == "Tz1" or val == "Tva1":
                bnds.append((None, None))
            else:
                bnds.append((0.000000000001, None))
        
        result = minimize(self.objValue,
                          x0, 
                          constraints = cons,
                         bounds = bnds) 

        return result
        


model = CGE(SAM)
result = model.SolveModel()

print(result)



class CGEResults():

    def __init__(self, sec_code):


        # Baz Yıl
        self.m1 = CGE(SAM)
        self.result1 = self.m1.SolveModel()
        

        # Karşı Olgusal Denge
        self.m2 = CGE(SAM)
        self.m2.DCOBar = 27
        self.m2.DNGBar = 8
        self.m2.epsilon /= 1.2
        self.m2.E_Energy = 40
        self.m2.Ar *= 1.2
        self.m2.Ab *= 1.2
        self.result2 = self.m2.SolveModel()

        self.sectors = {0: "Tarım", 1: "Hizmet", 2:"Sanayi"}
        self.sec_code = sec_code

        self.EndoVarandParameters()
        self.MacroVariables()
        self.parametersValues()
        self.EndoVarValues()


        df = pd.DataFrame(columns = ["base", "cfrun", "%Change"], 
                          index   = self.basestr)

        df.base = self.baselist
        df.cfrun = self.Cflist
        df["%Change"] = (df.cfrun - df.base) / df.base * 100
        df.to_excel("Results.xlsx")


        fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1,3, figsize = (15, 4))

        plt.tight_layout()
        self.AxisLimits()
        self.axes1()
        self.axes2()
        self.axes3()
        plt.show()

    def parametersValues(self):

        df = pd.DataFrame(index = self.m1.parameters_str, columns = ["Values"])
        df.Values = self.m1.parameters
        df.to_excel("ParamterValues.xlsx")

    def EndoVarValues(self):
        
        df = pd.DataFrame(index = self.m1.init_values_str, columns = ["Values"])
        df.Values = self.m1.init_values
        df.to_excel("InıtValues.xlsx")
        
    def AxisLimits(self):

        #-------------------------------Axes 1------------------------------------
        self.axis1xmin = [self.D_base[self.sec_code] if self.D_base[self.sec_code] < self.D_cfrun[self.sec_code] else self.D_cfrun[self.sec_code]][0]
        self.axis1xmax = [self.D_base[self.sec_code] if self.D_base[self.sec_code] > self.D_cfrun[self.sec_code] else self.D_cfrun[self.sec_code]][0]

        self.axis1ymin = [self.M_base[self.sec_code] if self.M_base[self.sec_code] < self.M_cfrun[self.sec_code] else self.M_cfrun[self.sec_code]][0]
        self.axis1ymax = [self.M_base[self.sec_code] if self.M_base[self.sec_code] > self.M_cfrun[self.sec_code] else self.M_cfrun[self.sec_code]][0]

        if self.sec_code == 0 or self.sec_code == 1:

            self.ax1.set_xlim(xmin = self.axis1xmin - self.axis1xmin/5, xmax = self.axis1xmax + self.axis1xmax/7)
            self.ax1.set_ylim(ymin = 0, ymax = self.axis1ymax + self.axis1ymax*4 )

        else:

            self.ax1.set_xlim(xmin = self.axis1xmin - self.axis1xmin/2, xmax = self.axis1xmax + self.axis1xmax/2)
            self.ax1.set_ylim(ymin = self.axis1ymin - self.axis1ymin/2, ymax = self.axis1ymax + self.axis1ymax/2 )


        #-------------------------------Axes 2------------------------------------
        self.axis2xmin = [self.D_base[self.sec_code] if self.D_base[self.sec_code] < self.D_cfrun[self.sec_code] else self.D_cfrun[self.sec_code]][0]
        self.axis2xmax = [self.D_base[self.sec_code] if self.D_base[self.sec_code] > self.D_cfrun[self.sec_code] else self.D_cfrun[self.sec_code]][0]

        self.axis2ymin = [self.E_base[self.sec_code] if self.E_base[self.sec_code] < self.E_cfrun[self.sec_code] else self.E_cfrun[self.sec_code]][0]
        self.axis2ymax = [self.E_base[self.sec_code] if self.E_base[self.sec_code] > self.E_cfrun[self.sec_code] else self.E_cfrun[self.sec_code]][0]

        if self.sec_code == 0 or self.sec_code == 1:

            self.ax2.set_xlim(xmin = self.axis2xmin - self.axis2xmin/5, xmax = self.axis2xmax + self.axis2xmax/7)
            self.ax2.set_ylim(ymin = 0, ymax = self.axis2ymax + self.axis2ymax*4 )

        else:

            self.ax2.set_xlim(xmin = self.axis2xmin - self.axis2xmin/2, xmax = self.axis2xmax + self.axis2xmax/2)
            self.ax2.set_ylim(ymin = self.axis2ymin - self.axis2ymin/2, ymax = self.axis2ymax + self.axis2ymax/2 )


        #-------------------------------Axes 3------------------------------------
        self.ForeignTradeXLineMaxValue = [self.E_base[self.sec_code] if self.E_base[self.sec_code] > self.E_cfrun[self.sec_code] else self.E_cfrun[self.sec_code]][0]
        self.ForeignTradeYLineMaxValue = [self.M_base[self.sec_code] if self.M_base[self.sec_code] > self.M_cfrun[self.sec_code] else self.M_cfrun[self.sec_code]][0]

        ax3ymin = [self.Sf_base[self.sec_code] if self.Sf_base[self.sec_code] <self.Sf_cfrun[self.sec_code] else self.Sf_cfrun[self.sec_code]][0]

        self.ax3.set_xlim(xmin = 0, xmax = self.ForeignTradeXLineMaxValue*2.2 )
        self.ax3.set_ylim(ymin = ax3ymin, ymax = self.ForeignTradeYLineMaxValue*1.5)
        
        if ax3ymin < 0:
            self.ax3.spines['bottom'].set_position("zero")
        else:
            self.ax3.spines['bottom'].set_position(("data", ax3ymin))

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
        

        self.baselist = [X1_base, L1_base, K1_base, I11_base, I21_base, I31_base, Ir1_base, Ib1_base, Z1_base, E1_base, D1_base, Q1_base, M1_base, 
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
            Sf_base, self.r_base ]
        
        self.Cflist = [X1_cfrun, L1_cfrun, K1_cfrun, I11_cfrun, I21_cfrun, I31_cfrun, Ir1_cfrun, Ib1_cfrun, Z1_cfrun, E1_cfrun, D1_cfrun, Q1_cfrun, M1_cfrun, 
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
            Sf_cfrun, self.r_cfrun  ]

        self.basestr = self.m1.init_values_str



        self.U_base, self.U_cfrun = -1*self.result1.fun, -1*self.result2.fun
        
        [td_base, tva1_base, tva2_base, tva3_base, tvar_base, tvab_base, tz1_base, tz2_base, tz3_base, tzr_base, tzb_base,
        alpha1_base, alpha2_base, alpha3_base, alphar_base,alphab_base, delta1_base, delta2_base,
        delta3_base, deltar_base, deltab_base, beta1_base, beta2_base, beta3_base,betar_base, betab_base, A1_base, A2_base, A3_base, 
        Ar_base, Ab_base, Axco_base, alphaxco_base, xr_base, co_base, Axng_base, alphaxng_base, xb_base, ng_base, 
        a11_base, a21_base, a31_base, ar1_base, ab1_base, a12_base, a22_base, a32_base, ar2_base, ab2_base, a13_base, a23_base,
        a33_base, ar3_base, ab3_base, a1r_base, a2r_base, a3r_base, arr_base, abr_base, a1b_base, a2b_base, a3b_base, arb_base, 
        abb_base, x1_base, x2_base, x3_base, xcor_base, xngb_base, rho1_base, rho2_base, rho3_base, rhor_base, rhob_base, eta1_base, 
        eta2_base, eta3_base, etar_base, etaco_base, etang_base, e1_base, e2_base, e3_base, er_base, eb_base, dt1_base, dt2_base, dt3_base, 
        dtr_base, dtb_base, theta1_base, theta2_base, theta3_base, thetar_base, thetab_base, m1_base, m2_base, m3_base, mr_base, mcor_base, 
        mngb_base, m1_base, m2_base, m3_base, mr_base, mcor_base, mngb_base, da1_base, da2_base, da3_base, dar_base, dcor_base, dngb_base, 
        lambda1_base, lambda2_base, lambda3_base, lambdar_base, lambdaco_base, lambdang_base, c1_base, c2_base, c3_base, cr_base, cb_base, 
        mu1_base, mu2_base, mu3_base, g1_base, g2_base, g3_base, inv1_base, inv2_base, inv3_base, sp_base, sg_base] = self.m1.model_parameters()

        [td_cfrun, tva1_cfrun, tva2_cfrun, tva3_cfrun, tvar_cfrun, tvab_cfrun, tz1_cfrun, tz2_cfrun, tz3_cfrun, tzr_cfrun, tzb_cfrun,
        alpha1_cfrun, alpha2_cfrun, alpha3_cfrun, alphar_cfrun,alphab_cfrun, delta1_cfrun, delta2_cfrun,
        delta3_cfrun, deltar_cfrun, deltab_cfrun, beta1_cfrun, beta2_cfrun, beta3_cfrun,betar_cfrun, betab_cfrun, A1_cfrun, A2_cfrun, A3_cfrun, 
        Ar_cfrun, Ab_cfrun, Axco_cfrun, alphaxco_cfrun, xr_cfrun, co_cfrun, Axng_cfrun, alphaxng_cfrun, xb_cfrun, ng_cfrun, 
        a11_cfrun, a21_cfrun, a31_cfrun, ar1_cfrun, ab1_cfrun, a12_cfrun, a22_cfrun, a32_cfrun, ar2_cfrun, ab2_cfrun, a13_cfrun, a23_cfrun,
        a33_cfrun, ar3_cfrun, ab3_cfrun, a1r_cfrun, a2r_cfrun, a3r_cfrun, arr_cfrun, abr_cfrun, a1b_cfrun, a2b_cfrun, a3b_cfrun, arb_cfrun, 
        abb_cfrun, x1_cfrun, x2_cfrun, x3_cfrun, xcor_cfrun, xngb_cfrun, rho1_cfrun, rho2_cfrun, rho3_cfrun, rhor_cfrun, rhob_cfrun, eta1_cfrun, 
        eta2_cfrun, eta3_cfrun, etar_cfrun, etaco_cfrun, etang_cfrun, e1_cfrun, e2_cfrun, e3_cfrun, er_cfrun, eb_cfrun, dt1_cfrun, dt2_cfrun, dt3_cfrun, 
        dtr_cfrun, dtb_cfrun, theta1_cfrun, theta2_cfrun, theta3_cfrun, thetar_cfrun, thetab_cfrun, m1_cfrun, m2_cfrun, m3_cfrun, mr_cfrun, mcor_cfrun, 
        mngb_cfrun, m1_cfrun, m2_cfrun, m3_cfrun, mr_cfrun, mcor_cfrun, mngb_cfrun, da1_cfrun, da2_cfrun, da3_cfrun, dar_cfrun, dcor_cfrun, dngb_cfrun, 
        lambda1_cfrun, lambda2_cfrun, lambda3_cfrun, lambdar_cfrun, lambdaco_cfrun, lambdang_cfrun, c1_cfrun, c2_cfrun, c3_cfrun, cr_cfrun, cb_cfrun, 
        mu1_cfrun, mu2_cfrun, mu3_cfrun, g1_cfrun, g2_cfrun, g3_cfrun, inv1_cfrun, inv2_cfrun, inv3_cfrun, sp_cfrun, sg_cfrun] = self.m2.model_parameters()


        self.D_base      = [D1_base, D2_base, D3_base]
        self.Q_base      = [Q1_base, Q2_base, Q3_base]
        self.M_base      = [M1_base, M2_base, M3_base]
        self.lambda_base = [lambda1_base, lambda2_base, lambda3_base]
        self.eta_base    = [eta1_base, eta2_base, eta3_base]
        self.m_base      = [m1_base, m2_base, m3_base]
        self.da_base     = [da1_base, da2_base, da3_base]
        self.pq_base     = [pq1_base, pq2_base, pq3_base]
        self.pd_base     = [pd1_base, pd2_base, pd3_base]
        self.pm_base     = [pm1_base, pm2_base, pm3_base]
        self.Tva_base     = [Tva1_base, Tva2_base, Tva3_base]
        self.Tz_base     = [Tz1_base, Tz2_base, Tz3_base]
        
        self.Z_base      = [Z1_base, Z2_base, Z3_base]
        self.E_base      = [E1_base, E2_base, E3_base]
        self.theta_base  = [theta1_base, theta2_base, theta3_base]
        self.rho_base    = [rho1_base, rho2_base, rho3_base]
        self.e_base      = [e1_base, e2_base, e3_base]
        self.dt_base     = [dt1_base, dt2_base, dt3_base]
        self.tz_base     = [tz1_base, tz2_base, tz3_base]
        self.tva_base    = [tva1_base, tva2_base, tva3_base]
        self.pz_base     = [pz1_base, pz2_base, pz3_base]
        self.pe_base     = [pe1_base, pe2_base, pe3_base]
        
        self.D_cfrun      = [D1_cfrun, D2_cfrun, D3_cfrun]
        self.Q_cfrun      = [Q1_cfrun, Q2_cfrun, Q3_cfrun]
        self.M_cfrun      = [M1_cfrun, M2_cfrun, M3_cfrun]
        self.lambda_cfrun = [lambda1_cfrun, lambda2_cfrun, lambda3_cfrun]
        self.eta_cfrun    = [eta1_cfrun, eta2_cfrun, eta3_cfrun]
        self.m_cfrun      = [m1_cfrun, m2_cfrun, m3_cfrun]
        self.da_cfrun     = [da1_cfrun, da2_cfrun, da3_cfrun]
        self.pq_cfrun     = [pq1_cfrun, pq2_cfrun, pq3_cfrun]
        self.pd_cfrun     = [pd1_cfrun, pd2_cfrun, pd3_cfrun]
        self.pm_cfrun     = [pm1_cfrun, pm2_cfrun, pm3_cfrun]
       
        self.Tva_cfrun     = [Tva1_cfrun, Tva2_cfrun, Tva3_cfrun]
        self.Tz_cfrun     = [Tz1_cfrun, Tz2_cfrun, Tz3_cfrun]
        
        self.Z_cfrun      = [Z1_cfrun, Z2_cfrun, Z3_cfrun]
        self.E_cfrun      = [E1_cfrun, E2_cfrun, E3_cfrun]
        self.theta_cfrun  = [theta1_cfrun, theta2_cfrun, theta3_cfrun]
        self.rho_cfrun    = [rho1_cfrun, rho2_cfrun, rho3_cfrun]
        self.e_cfrun      = [e1_cfrun, e2_cfrun, e3_cfrun]
        self.dt_cfrun     = [dt1_cfrun, dt2_cfrun, dt3_cfrun]
        self.tz_cfrun     = [tz1_cfrun, tz2_cfrun, tz3_cfrun]
        self.tva_cfrun    = [tva1_cfrun, tva2_cfrun, tva3_cfrun]
        self.pz_cfrun     = [pz1_cfrun, pz2_cfrun, pz3_cfrun]
        self.pe_cfrun     = [pe1_cfrun, pe2_cfrun, pe3_cfrun]

        self.Sf_base  = [M1_base*pm1_base - E1_base*pe1_base, M2_base*pm2_base - E2_base*pe2_base,M3_base*pm3_base - E3_base*pe3_base]
        self.Sf_cfrun = [M1_cfrun*pm1_cfrun - E1_cfrun*pe1_cfrun, M2_cfrun*pm2_cfrun - E2_cfrun*pe2_cfrun,M3_cfrun*pm3_cfrun - E3_cfrun*pe3_cfrun]

        self.L_base     = [L1_base, L2_base, L3_base]
        self.K_base     = [K1_base, K2_base, K3_base]
        self.X_base     = [X1_base, X2_base, X3_base]
        self.A_base     = [A1_base, A2_base, A3_base]
        self.alpha_base = [alpha1_base, alpha2_base, alpha3_base]
        self.delta_base = [delta1_base, delta2_base, delta3_base]
        self.beta_base  = [beta1_base, beta2_base, beta3_base]
        self.px_base    = [px1_base, px2_base, px3_base]

        self.L_cfrun     = [L1_cfrun, L2_cfrun, L3_cfrun]
        self.K_cfrun     = [K1_cfrun, K2_cfrun, K3_cfrun]
        self.X_cfrun     = [X1_cfrun, X2_cfrun, X3_cfrun]
        self.A_cfrun     = [A1_cfrun, A2_cfrun, A3_cfrun]
        self.alpha_cfrun = [alpha1_cfrun, alpha2_cfrun, alpha3_cfrun]
        self.delta_cfrun = [delta1_cfrun, delta2_cfrun, delta3_cfrun]
        self.beta_cfrun  = [beta1_cfrun, beta2_cfrun, beta3_cfrun]
        self.px_cfrun    = [px1_cfrun, px2_cfrun, px3_cfrun]

        self.C_base = [C1_base, C2_base, C3_base]
        self.G_base = [G1_base, G2_base, G3_base]
        self.INV_base = [INV1_base, INV2_base, INV3_base]
        self.pq_base = [pq1_base, pq2_base, pq3_base]

        self.C_cfrun = [C1_cfrun, C2_cfrun, C3_cfrun]
        self.G_cfrun = [G1_cfrun, G2_cfrun, G3_cfrun]
        self.INV_cfrun = [INV1_cfrun, INV2_cfrun, INV3_cfrun]
        self.pq_cfrun = [pq1_cfrun, pq2_cfrun, pq3_cfrun]

        self.I1_base = [I11_base, I12_base, I13_base]
        self.I2_base = [I21_base, I22_base, I23_base]
        self.I3_base = [I31_base, I32_base, I33_base]

        self.C_cfrun = [C1_cfrun, C2_cfrun, C3_cfrun]
        self.G_cfrun = [G1_cfrun, G2_cfrun, G3_cfrun]
        self.INV_cfrun = [INV1_cfrun, INV2_cfrun, INV3_cfrun]
        self.pq_cfrun = [pq1_cfrun, pq2_cfrun, pq3_cfrun]

        self.C_cfrun = [C1_cfrun, C2_cfrun, C3_cfrun]
        self.G_cfrun = [G1_cfrun, G2_cfrun, G3_cfrun]
        self.INV_cfrun = [INV1_cfrun, INV2_cfrun, INV3_cfrun]
        self.pq_cfrun = [pq1_cfrun, pq2_cfrun, pq3_cfrun]

        self.I1_cfrun = [I11_cfrun, I21_cfrun, I31_cfrun]
        self.I2_cfrun = [I12_cfrun, I22_cfrun, I32_cfrun]
        self.I3_cfrun = [I13_cfrun, I23_cfrun, I33_cfrun]

        # Macro Variables
        # ---------------
        self.GDP_base_total = Y_base + Tva_base + Tz_base
        self.M_base_total   = M1_base*pm1_base + M2_base*pm2_base + M3_base*pm3_base
        self.E_base_total   = E1_base * pe1_base + E2_base*pe2_base + E3_base*pe3_base
        self.NE_base_total  = self.E_base_total - self.M_base_total
        self.Tva_base_total = Tva1_base + Tva2_base + Tva3_base
        self.Tz_base_total  = Tz1_base + Tz2_base + Tz3_base
        self.INV_base_total = INV1_base*pq1_base + INV2_base*pq2_base + INV3_base*pq3_base
        self.CES_base_total = C1_base*pq1_base + C2_base*pq2_base + C3_base*pq3_base
        self.G_base_total   = G1_base*pq1_base + G2_base*pq2_base + G3_base*pq3_base

        self.GDP_cfrun_total = Y_cfrun + Tva_cfrun + Tz_cfrun
        self.M_cfrun_total   = M1_cfrun*pm1_cfrun + M2_cfrun*pm2_cfrun + M3_cfrun*pm3_cfrun
        self.E_cfrun_total   = E1_cfrun * pe1_cfrun + E2_cfrun*pe2_cfrun + E3_cfrun*pe3_cfrun
        self.NE_cfrun_total  = self.E_cfrun_total - self.M_cfrun_total
        self.Tva_cfrun_total = Tva1_cfrun + Tva2_cfrun + Tva3_cfrun
        self.Tz_cfrun_total  = Tz1_cfrun + Tz2_cfrun + Tz3_cfrun
        self.INV_cfrun_total = INV1_cfrun*pq1_cfrun + INV2_cfrun*pq2_cfrun + INV3_cfrun*pq3_cfrun
        self.CES_cfrun_total = C1_cfrun*pq1_cfrun + C2_cfrun*pq2_cfrun + C3_cfrun*pq3_cfrun
        self.G_cfrun_total   = G1_cfrun*pq1_cfrun + G2_cfrun*pq2_cfrun + G3_cfrun*pq3_cfrun

        self.index = ["GDP", "Import", "Export", "Net Export", "VAT", "PROTAX", "Investment", "Prv Cons.", "Gov. Cons."]
       
        self.base_values = [self.GDP_base_total, self.M_base_total,self.E_base_total, self.NE_base_total, self.Tva_base_total,
                             self.Tz_base_total, self.INV_base_total, self.CES_base_total, self.G_base_total]
        
        self.cfrun_values = [self.GDP_cfrun_total, self.M_cfrun_total,self.E_cfrun_total, self.NE_cfrun_total, self.Tva_cfrun_total,
                             self.Tz_cfrun_total, self.INV_cfrun_total, self.CES_cfrun_total, self.G_cfrun_total]
        
       
        self.GDP_Consumption = np.sum((np.array(self.C_cfrun) + 
                                       np.array(self.G_cfrun) + 
                                       np.array(self.INV_cfrun)) * np.array(self.pq_cfrun) + 
                                       np.array(self.E_cfrun) * np.array(self.pe_cfrun) - np.array(self.M_cfrun) * np.array(self.pm_cfrun))
        
                
        self.GDP_Income = self.m2.Lbar*1 + self.m2.Kbar*self.r_cfrun + Tva_cfrun + Tz_cfrun

        self.GDP_production = np.sum(np.array(self.Z_cfrun)*np.array(self.pz_cfrun))  - np.sum((np.array(self.I1_cfrun)*np.array(self.pq_cfrun) +
                                                                                                 np.array(self.I2_cfrun)*np.array(self.pq_cfrun) + 
                                                                                                 np.array(self.I3_cfrun)*np.array(self.pq_cfrun))) +  Tva_cfrun + Tz_cfrun
        
        self.GDP_production2 = np.sum(np.array(self.X_cfrun)*np.array(self.px_cfrun)) +  Tva_cfrun + Tz_cfrun
        

        self.arm_budget_fridges = {"Mb":self.pq_base[self.sec_code]*self.Q_base[self.sec_code]/(self.pm_base[self.sec_code]),
                                   "Mc":self.pq_cfrun[self.sec_code]*self.Q_cfrun[self.sec_code]/(self.pm_cfrun[self.sec_code]),
                                   "Db":self.pq_base[self.sec_code]*self.Q_base[self.sec_code]/self.pd_base[self.sec_code], 
                                   "Dc":self.pq_cfrun[self.sec_code]*self.Q_cfrun[self.sec_code]/self.pd_cfrun[self.sec_code]}
        
        self.trs_budget_fridges = {"Eb":(1+self.tz_base[self.sec_code] + self.tva_base[self.sec_code])*self.pz_base[self.sec_code]*self.Z_base[self.sec_code] / self.pe_base[self.sec_code],
                                   "Ec":(1+self.tz_cfrun[self.sec_code] + self.tva_cfrun[self.sec_code])*self.pz_cfrun[self.sec_code]*self.Z_cfrun[self.sec_code] / self.pe_cfrun[self.sec_code] ,
                                   "Db":(1+self.tz_base[self.sec_code] + self.tva_base[self.sec_code])*self.pz_base[self.sec_code]*self.Z_base[self.sec_code] / self.pd_base[self.sec_code], 
                                   "Dc":(1+self.tz_cfrun[self.sec_code] + self.tva_cfrun[self.sec_code])*self.pz_cfrun[self.sec_code]*self.Z_cfrun[self.sec_code] / self.pd_cfrun[self.sec_code]}
        
        self.ces_budget_fridges = {"Kb":self.px_base[self.sec_code]*self.X_base[self.sec_code] / self.r_base,
                                   "Kc":self.px_cfrun[self.sec_code]*self.X_cfrun[self.sec_code] / self.r_cfrun, 
                                   "Lb":self.px_base[self.sec_code]*self.X_base[self.sec_code],
                                   "Lc":self.px_cfrun[self.sec_code]*self.X_cfrun[self.sec_code]}
        
        self.arm_fridges =  {"Mb": ((self.Q_base[self.sec_code]/self.lambda_base[self.sec_code])**self.eta_base[self.sec_code]*(1/self.m_base[self.sec_code]))**(1/self.eta_base[self.sec_code]),
                             "Mc": ((self.Q_cfrun[self.sec_code]/self.lambda_cfrun[self.sec_code])**self.eta_cfrun[self.sec_code]*(1/self.m_cfrun[self.sec_code]))**(1/self.eta_cfrun[self.sec_code]),
                             "Db": ((self.Q_base[self.sec_code]/self.lambda_base[self.sec_code])**self.eta_base[self.sec_code]*1/self.da_base[self.sec_code])**(1/self.eta_base[self.sec_code]), 
                             "Dc": ((self.Q_cfrun[self.sec_code]/self.lambda_cfrun[self.sec_code])**self.eta_cfrun[self.sec_code]*(1/self.da_cfrun[self.sec_code]))**(1/self.eta_cfrun[self.sec_code])}
        
        self.trs_fridges =  {"Eb": ((self.Z_base[self.sec_code] / self.theta_base[self.sec_code])**self.rho_base[self.sec_code]*1/self.e_base[self.sec_code])**(1/self.rho_base[self.sec_code]),
                             "Ec": ((self.Z_cfrun[self.sec_code]/ self.theta_cfrun[self.sec_code])**self.rho_cfrun[self.sec_code]*1/self.e_cfrun[self.sec_code])**(1/self.rho_cfrun[self.sec_code]),
                             "Db": ((self.Z_base[self.sec_code] / self.theta_base[self.sec_code])**self.rho_base[self.sec_code]*1/self.dt_base[self.sec_code])**(1/self.rho_base[self.sec_code]), 
                             "Dc": ((self.Z_cfrun[self.sec_code]/ self.theta_cfrun[self.sec_code])**self.rho_cfrun[self.sec_code]*1/self.dt_cfrun[self.sec_code])**(1/self.rho_cfrun[self.sec_code])}
        
        self.ces_fridges =  {"Kb": ((self.X_base[self.sec_code]/self.A_base[self.sec_code])**self.alpha_base[self.sec_code]*(1/self.delta_base[self.sec_code]))**(1/self.alpha_base[self.sec_code]),
                             "Kc": ((self.X_cfrun[self.sec_code]/self.A_cfrun[self.sec_code])**self.alpha_cfrun[self.sec_code]*(1/self.delta_cfrun[self.sec_code]))**(1/self.alpha_cfrun[self.sec_code]),
                             "Lb": ((self.X_base[self.sec_code]/self.A_base[self.sec_code])**self.alpha_base[self.sec_code]*1/self.beta_base[self.sec_code])**(1/self.alpha_base[self.sec_code]), 
                             "Lc": ((self.X_cfrun[self.sec_code]/self.A_cfrun[self.sec_code])**self.alpha_cfrun[self.sec_code]*1/self.beta_cfrun[self.sec_code])**(1/self.alpha_cfrun[self.sec_code])}

    def ArmingtonFunction(self, X):

        return {"Base": ((self.Q_base[self.sec_code] / self.lambda_base[self.sec_code])**self.eta_base[self.sec_code] * 1/self.m_base[self.sec_code] - self.da_base[self.sec_code]/self.m_base[self.sec_code] * X **self.eta_base[self.sec_code])**(1/self.eta_base[self.sec_code]),
                "CFRun":((self.Q_cfrun[self.sec_code] / self.lambda_cfrun[self.sec_code])**self.eta_cfrun[self.sec_code] * 1/self.m_cfrun[self.sec_code] - self.da_cfrun[self.sec_code]/self.m_cfrun[self.sec_code] * X**self.eta_cfrun[self.sec_code])**(1/self.eta_cfrun[self.sec_code]) }
    
    def ArmingtonBudget(self, X):

        return {"Base" : self.pq_base[self.sec_code]*self.Q_base[self.sec_code] / (self.pm_base[self.sec_code] ) - self.pd_base[self.sec_code] / self.pm_base[self.sec_code] * X,
                "CFRun": self.pq_cfrun[self.sec_code]*self.Q_cfrun[self.sec_code] / (self.pm_cfrun[self.sec_code] ) - self.pd_cfrun[self.sec_code] / self.pm_cfrun[self.sec_code]* X }
    
    def TransformationFunction(self, X):

        return  {"Base": ((self.Z_base[self.sec_code] / self.theta_base[self.sec_code])**self.rho_base[self.sec_code] * (1/self.e_base[self.sec_code]) - (self.dt_base[self.sec_code]/self.e_base[self.sec_code] )* X**self.rho_base[self.sec_code])**(1/self.rho_base[self.sec_code]),
                 "CFRun":((self.Z_cfrun[self.sec_code] / self.theta_cfrun[self.sec_code])**self.rho_cfrun[self.sec_code] * (1/self.e_cfrun[self.sec_code]) - self.dt_cfrun[self.sec_code]/self.e_cfrun[self.sec_code]* X**self.rho_cfrun[self.sec_code])**(1/self.rho_cfrun[self.sec_code]) }  
    
    def TransformationBudget(self, X):

        return {"Base": (1+self.tva_base[self.sec_code]+self.tz_base[self.sec_code])*self.pz_base[self.sec_code]*self.Z_base[self.sec_code] / self.pe_base[self.sec_code] - self.pd_base[self.sec_code] / self.pe_base[self.sec_code] * X,
                "CFRun":(1+self.tva_cfrun[self.sec_code]+self.tz_cfrun[self.sec_code])*self.pz_cfrun[self.sec_code]*self.Z_cfrun[self.sec_code] / self.pe_cfrun[self.sec_code] - self.pd_cfrun[self.sec_code] / self.pe_cfrun[self.sec_code] * X}

    def ForeignTrade(self,X):

        return {"Base" : self.Sf_base[self.sec_code] / self.pm_base[self.sec_code]+ self.pe_base[self.sec_code]/self.pm_base[self.sec_code] * X ,
                "CFRun": self.Sf_cfrun[self.sec_code] / self.pm_cfrun[self.sec_code]  + self.pe_cfrun[self.sec_code]/self.pm_cfrun[self.sec_code] * X}
    
    def CES(self, X, sec_code):

         return {"Base" : ((self.X_base[sec_code] / self.A_base[sec_code])**self.alpha_base[sec_code] * 1/self.beta_base[sec_code] - self.delta_base[sec_code]/self.beta_base[sec_code] * X **self.alpha_base[sec_code])**(1/self.alpha_base[sec_code]),
                 "CFRun": ((self.X_cfrun[sec_code] / self.A_cfrun[sec_code])**self.alpha_cfrun[sec_code] * 1/self.beta_cfrun[sec_code] - self.delta_cfrun[sec_code]/self.beta_cfrun[sec_code] * X **self.alpha_cfrun[sec_code])**(1/self.alpha_cfrun[sec_code])}

    def CESFridges(self, sec_code):

        return {"ces_budget_fridges" : {"Kb":self.px_base[sec_code]*self.X_base[sec_code] / self.r_base,
                                        "Kc":self.px_cfrun[sec_code]*self.X_cfrun[sec_code] / self.r_cfrun, 
                                        "Lb":self.px_base[sec_code]*self.X_base[sec_code],
                                        "Lc":self.px_cfrun[sec_code]*self.X_cfrun[sec_code]},

                "ces_fridges" :  {"Kb": ((self.X_base[sec_code]/self.A_base[sec_code])**self.alpha_base[sec_code]*(1/self.delta_base[sec_code]))**(1/self.alpha_base[sec_code]),
                                  "Kc": ((self.X_cfrun[sec_code]/self.A_cfrun[sec_code])**self.alpha_cfrun[sec_code]*(1/self.delta_cfrun[sec_code]))**(1/self.alpha_cfrun[sec_code]),
                                  "Lb": ((self.X_base[sec_code]/self.A_base[sec_code])**self.alpha_base[sec_code]*1/self.beta_base[sec_code])**(1/self.alpha_base[sec_code]), 
                                  "Lc": ((self.X_cfrun[sec_code]/self.A_cfrun[sec_code])**self.alpha_cfrun[sec_code]*1/self.beta_cfrun[sec_code])**(1/self.alpha_cfrun[sec_code])}
  
        }
    
    def CESBudget(self, X, sec_code):

        return {"Base" : self.px_base[sec_code]*self.X_base[sec_code] / self.r_base - 1 / self.r_base * X,
                "CFRun": self.px_cfrun[sec_code]*self.X_cfrun[sec_code] / self.r_cfrun - 1 / self.r_cfrun * X}

    def ArmingtonValues(self):

        Db = self.arm_budget_fridges["Db"]
        Dc = self.arm_budget_fridges["Dc"]
        Mb = self.arm_budget_fridges["Mb"]
        Mc = self.arm_budget_fridges["Mc"]

        #-------------------------Base Run Equilibrium-------------------------------------

        self.arm_budget_1_x_base = np.linspace(0, Db, 100)
        self.arm_budget_1_y_base = self.ArmingtonBudget(self.arm_budget_1_x_base)["Base"]

        X = np.linspace(0, Db, 100)
        Y = self.ArmingtonFunction(X)["Base"]

        self.arm_1_x_base = []
        self.arm_1_y_base = []

        for i in range(len(Y)):
            if Y[i] > Mb or Y[i] <0:
                continue

            self.arm_1_x_base.append(X[i])
            self.arm_1_y_base.append(Y[i])

        #----------------------CounterFactual Eqquilibrium---------------------------------

        self.arm_budget_1_x_cfrun = np.linspace(0, Dc, 100)
        self.arm_budget_1_y_cfrun = self.ArmingtonBudget(self.arm_budget_1_x_cfrun)["CFRun"]
        

        X = np.linspace(0, Dc, 100)
        Y = self.ArmingtonFunction(X)["CFRun"]

        self.arm_1_x_cfrun= []
        self.arm_1_y_cfrun = []

        for i in range(len(Y)):
            if Y[i] > Mc or Y[i] < 0:
                continue

            self.arm_1_x_cfrun.append(X[i])
            self.arm_1_y_cfrun.append(Y[i])

    def axes1(self):

        self.ArmingtonValues()
        #----------------Base Year Equilibrium----------------------------
        label_base_arm =  "Baz Yıl\nQ: {:.2f}\nD: {:.2f}\nM: {:.2f}\npd: {:.2f}\npm: {:.2f}\npq: {:.2f}".format(self.Q_base[self.sec_code], 
                                                                                              self.D_base[self.sec_code], self.M_base[self.sec_code], 
                                                                                              self.pd_base[self.sec_code],
                                                                                              self.pm_base[self.sec_code], 
                                                                                              self.pq_base[self.sec_code])
        
        self.ax1.plot(self.arm_1_x_base, self.arm_1_y_base, "-k", linewidth = 3, label = label_base_arm)
        self.ax1.plot(self.arm_budget_1_x_base, self.arm_budget_1_y_base, "-k", linewidth = 3)
        self.ax1.plot([0, self.D_base[self.sec_code]], [self.M_base[self.sec_code], self.M_base[self.sec_code]], "--ok")
        self.ax1.plot([self.D_base[self.sec_code], self.D_base[self.sec_code]], [0, self.M_base[self.sec_code]], "--ok")

        #------------Counterfactual Equilibrium----------------------------
        label_cfrun_arm =  "Şok Durumu\nQ: {:.2f}\nD: {:.2f}\nM: {:.2f}\npd: {:.2f}\npm: {:.2f}\npq: {:.2f}".format(self.Q_cfrun[self.sec_code], 
                                                                                              self.D_cfrun[self.sec_code], self.M_cfrun[self.sec_code], 
                                                                                              self.pd_cfrun[self.sec_code],
                                                                                              self.pm_cfrun[self.sec_code],
                                                                                              self.pq_cfrun[self.sec_code])
        
        self.ax1.plot(self.arm_1_x_cfrun, self.arm_1_y_cfrun, "--k", linewidth = 1, label = label_cfrun_arm)
        self.ax1.plot(self.arm_budget_1_x_cfrun, self.arm_budget_1_y_cfrun, "--k", linewidth = 1)
        self.ax1.plot([0, self.D_cfrun[self.sec_code]], [self.M_cfrun[self.sec_code], self.M_cfrun[self.sec_code]], "--ok")
        self.ax1.plot([self.D_cfrun[self.sec_code], self.D_cfrun[self.sec_code]], [0, self.M_cfrun[self.sec_code]], "--ok")

        #----------------------Axes-------------------------------------
        self.ax1.set_title("{} Sektörü Armington Fonksiyonu".format(self.sectors[self.sec_code]), fontweight = "bold", fontsize = 13)
        self.ax1.set_xlabel("Yurtiçi Mal Talebi",fontweight = "bold", fontsize = 11)
        self.ax1.set_ylabel("İthal Mal Talebi",fontweight = "bold", fontsize = 11)
        self.ax1.spines["right"].set_visible(False)
        self.ax1.spines["top"].set_visible(False)
        self.ax1.legend(fontsize = 7, edgecolor = "black", ncol = 2)

    def TransformationValues(self):


        BaseBudgetMin = self.D_base[self.sec_code] - self.D_base[self.sec_code]/3
        BaseBudgetMax = self.D_base[self.sec_code] + self.D_base[self.sec_code]/3

        CfrunBudgetMin = self.D_cfrun[self.sec_code] - self.D_cfrun[self.sec_code]/3
        CfrunBudgetMax = self.D_cfrun[self.sec_code] + self.D_cfrun[self.sec_code]/3

        Db = self.trs_fridges["Db"]
        Dc = self.trs_fridges["Dc"]




        self.trs_budget_1_x_base = np.linspace(BaseBudgetMin, BaseBudgetMax , 100)
        self.trs_budget_1_y_base = self.TransformationBudget(self.trs_budget_1_x_base)["Base"]

        self.trs_1_x_base = np.linspace(0, Db, 100)
        self.trs_1_y_base = self.TransformationFunction(self.trs_1_x_base)["Base"]

        self.trs_budget_1_x_cfrun = np.linspace(CfrunBudgetMin, CfrunBudgetMax, 100)
        self.trs_budget_1_y_cfrun = self.TransformationBudget(self.trs_budget_1_x_cfrun)["CFRun"]

        self.trs_1_x_cfrun = np.linspace(0, Dc, 100)
        self.trs_1_y_cfrun = self.TransformationFunction(self.trs_1_x_cfrun)["CFRun"]
  
    def axes2(self):

        self.TransformationValues()
        #----------------Base Year Equilibrium----------------------------
        label_base_trs =  "Baz Yıl\nZ: {:.2f}\nD: {:.2f}\nE: {:.2f}\npd: {:.2f}\npe: {:.2f}\npz: {:.2f}".format(self.Z_base[self.sec_code], 
                                                                                              self.D_base[self.sec_code], self.E_base[self.sec_code],
                                                                                              self.pd_base[self.sec_code],
                                                                                              self.pe_base[self.sec_code],
                                                                                              self.pz_base[self.sec_code])

        self.ax2.plot(self.trs_1_x_base, self.trs_1_y_base, "-k", linewidth = 3, label = label_base_trs)
        self.ax2.plot(self.trs_budget_1_x_base, self.trs_budget_1_y_base, "-k", linewidth = 3)

        self.ax2.plot([0, self.D_base[self.sec_code]], [self.E_base[self.sec_code], self.E_base[self.sec_code]], "--ok")
        self.ax2.plot([self.D_base[self.sec_code], self.D_base[self.sec_code]], [0, self.E_base[self.sec_code]], "--ok")

        #------------Counterfactual Equilibrium----------------------------
        label_cfrun_trs =  "Şok Durumu\nZ: {:.2f}\nD: {:.2f}\nE: {:.2f}\npd: {:.2f}\npe: {:.2f}\npz: {:.2f}".format(self.Z_cfrun[self.sec_code], 
                                                                                              self.D_cfrun[self.sec_code], self.E_cfrun[self.sec_code],
                                                                                              self.pd_cfrun[self.sec_code],
                                                                                              self.pe_cfrun[self.sec_code], 
                                                                                              self.pz_cfrun[self.sec_code])
        

        self.ax2.plot(self.trs_1_x_cfrun, self.trs_1_y_cfrun, "--k", linewidth = 1, label = label_cfrun_trs)
        self.ax2.plot(self.trs_budget_1_x_cfrun, self.trs_budget_1_y_cfrun, "--k", linewidth = 1)
        self.ax2.plot([0, self.D_cfrun[self.sec_code]], [self.E_cfrun[self.sec_code], self.E_cfrun[self.sec_code]], "--ok")
        self.ax2.plot([self.D_cfrun[self.sec_code], self.D_cfrun[self.sec_code]], [0, self.E_cfrun[self.sec_code]], "--ok")


        #----------------------Axes-------------------------------------
        self.ax2.set_title("{} Sektörü Dönüşüm Fonksiyonu".format(self.sectors[self.sec_code]), fontweight = "bold", fontsize = 13)
        self.ax2.set_xlabel("Yurtiçi Arz",fontweight = "bold", fontsize = 11)
        self.ax2.set_ylabel("İhracat Arzı",fontweight = "bold", fontsize = 11)
        self.ax2.spines["right"].set_visible(False)
        self.ax2.spines["top"].set_visible(False)

        self.ax2.legend(fontsize = 7, edgecolor = "black", ncol = 2, bbox_to_anchor=(0.48, 0.285))

    def ForeignTradeValues(self):

        self.trade_line_1_x_base = np.linspace(0, self.ForeignTradeXLineMaxValue*2, 100)
        self.trade_line_1_y_base = self.ForeignTrade(self.trade_line_1_x_base)["Base"]

        self.trade_line_1_x_cfrun = np.linspace(0, self.ForeignTradeXLineMaxValue*2, 100)
        self.trade_line_1_y_cfrun = self.ForeignTrade(self.trade_line_1_x_cfrun)["CFRun"]
    
    def axes3(self):


        self.ForeignTradeValues()

        label_base_foreign =  "Baz Yıl\nE: {:.2f}\nM: {:.2f}\nSf/pm: {:.2f}\npe/pm: {:.2f}".format(self.E_base[self.sec_code], self.M_base[self.sec_code],
                                                                                                    self.Sf_base[self.sec_code]/self.pm_base[self.sec_code],
                                                                                                    self.pe_base[self.sec_code]/self.pm_base[self.sec_code])
        
        label_cfrun_foreign =  "Şok Durumu\nE: {:.2f}\nM: {:.2f}\nSf/pm: {:.2f}\npe/pm: {:.2f}".format(self.E_cfrun[self.sec_code], self.M_cfrun[self.sec_code], 
                                                                                                                   self.Sf_cfrun[self.sec_code]/self.pm_cfrun[self.sec_code], 
                                                                                                                   self.pe_cfrun[self.sec_code]/self.pm_cfrun[self.sec_code])

        self.ax3.plot(self.trade_line_1_x_base, self.trade_line_1_y_base, "-k", linewidth = 3, label = label_base_foreign)


        self.ax3.plot(self.trade_line_1_x_cfrun, self.trade_line_1_y_cfrun, "--k", linewidth = 1, label = label_cfrun_foreign)

        self.ax3.plot([0, self.E_base[self.sec_code]], [self.M_base[self.sec_code], self.M_base[self.sec_code]], "--ok")
        self.ax3.plot([self.E_base[self.sec_code], self.E_base[self.sec_code]], [0, self.M_base[self.sec_code]], "--ok")

        self.ax3.plot([0, self.E_cfrun[self.sec_code]], [self.M_cfrun[self.sec_code], self.M_cfrun[self.sec_code]], "--ok")
        self.ax3.plot([self.E_cfrun[self.sec_code], self.E_cfrun[self.sec_code]], [0, self.M_cfrun[self.sec_code]], "--ok")

        #----------------------Axes-------------------------------------
        self.ax3.set_title("{} Sektörü Dış Ticaret".format(self.sectors[self.sec_code]), fontweight = "bold", fontsize = 13)
        self.ax3.set_xlabel("İhracat",fontweight = "bold", fontsize = 11)
        self.ax3.set_ylabel("İthalat",fontweight = "bold", fontsize = 11)
        self.ax3.spines["right"].set_visible(False)
        self.ax3.spines["top"].set_visible(False)

        self.ax3.legend(fontsize = 7, edgecolor = "black", ncol = 2, bbox_to_anchor=(0.5, 0.4))

    def CesValues(self):

        Lb0 = self.CESFridges(0)["ces_budget_fridges"]["Lb"]
        Lc0 = self.CESFridges(0)["ces_budget_fridges"]["Lc"]
        Kb0 = self.CESFridges(0)["ces_budget_fridges"]["Kb"]
        Kc0 = self.CESFridges(0)["ces_budget_fridges"]["Kc"]

        Lb1 = self.CESFridges(1)["ces_budget_fridges"]["Lb"]
        Lc1 = self.CESFridges(1)["ces_budget_fridges"]["Lc"]
        Kb1 = self.CESFridges(1)["ces_budget_fridges"]["Kb"]
        Kc1 = self.CESFridges(1)["ces_budget_fridges"]["Kc"]

        Lb2 = self.CESFridges(2)["ces_budget_fridges"]["Lb"]
        Lc2 = self.CESFridges(2)["ces_budget_fridges"]["Lc"]
        Kb2 = self.CESFridges(2)["ces_budget_fridges"]["Kb"]
        Kc2 = self.CESFridges(2)["ces_budget_fridges"]["Kc"]

        #-------------------------Base Run Equilibrium-------------------------------------

        #Axes1
        self.ces_budget_0_x_base = np.linspace(0, Lb0, 100)
        self.ces_budget_0_y_base = self.CESBudget(self.ces_budget_0_x_base, 0)["Base"]

        X = np.linspace(0, Lb0, 100)
        Y = self.CES(X,0)["Base"]

        self.ces_0_x_base = []
        self.ces_0_y_base = []

        for i in range(len(Y)):
            if Y[i] > Kb0 or Y[i] < 0:
                continue

            self.ces_0_x_base.append(X[i])
            self.ces_0_y_base.append(Y[i])


        #Axes2
        self.ces_budget_1_x_base = np.linspace(0, Lb1, 100)
        self.ces_budget_1_y_base = self.CESBudget(self.ces_budget_1_x_base, 1)["Base"]

        X = np.linspace(0, Lb1, 111)
        Y = self.CES(X,1)["Base"]

        self.ces_1_x_base = []
        self.ces_1_y_base = []

        for i in range(len(Y)):
            if Y[i] > Kb1 or Y[i] < 0:
                continue

            self.ces_1_x_base.append(X[i])
            self.ces_1_y_base.append(Y[i])


        #Axes 3
        self.ces_budget_2_x_base = np.linspace(0, Lb2, 100)
        self.ces_budget_2_y_base = self.CESBudget(self.ces_budget_2_x_base, 2)["Base"]

        X = np.linspace(0, Lb2, 222)
        Y = self.CES(X,2)["Base"]

        self.ces_2_x_base = []
        self.ces_2_y_base = []

        for i in range(len(Y)):
            if Y[i] > Kb2 or Y[i] < 0:
                continue

            self.ces_2_x_base.append(X[i])
            self.ces_2_y_base.append(Y[i])



        #-------------------------CFRun Equilibrium-------------------------------------

        #Axes1
        self.ces_budget_0_x_cfrun = np.linspace(0, Lc0, 100)
        self.ces_budget_0_y_cfrun = self.CESBudget(self.ces_budget_0_x_cfrun, 0)["CFRun"]

        X = np.linspace(0, Lc0, 100)
        Y = self.CES(X,0)["CFRun"]

        self.ces_0_x_cfrun = []
        self.ces_0_y_cfrun = []

        for i in range(len(Y)):
            if Y[i] > Kc0 or Y[i] < 0:
                continue

            self.ces_0_x_cfrun.append(X[i])
            self.ces_0_y_cfrun.append(Y[i])


        #Axes2
        self.ces_budget_1_x_cfrun = np.linspace(0, Lc1, 100)
        self.ces_budget_1_y_cfrun = self.CESBudget(self.ces_budget_1_x_cfrun, 1)["CFRun"]

        X = np.linspace(0, Lc1, 111)
        Y = self.CES(X,1)["CFRun"]

        self.ces_1_x_cfrun = []
        self.ces_1_y_cfrun = []

        for i in range(len(Y)):
            if Y[i] > Kc1 or Y[i] < 0:
                continue

            self.ces_1_x_cfrun.append(X[i])
            self.ces_1_y_cfrun.append(Y[i])


        #Axes3
        self.ces_budget_2_x_cfrun = np.linspace(0, Lc2, 100)
        self.ces_budget_2_y_cfrun = self.CESBudget(self.ces_budget_2_x_cfrun, 2)["CFRun"]

        X = np.linspace(0, Lc2, 222)
        Y = self.CES(X,2)["CFRun"]

        self.ces_2_x_cfrun = []
        self.ces_2_y_cfrun = []

        for i in range(len(Y)):
            if Y[i] > Kc2 or Y[i] < 0:
                continue

            self.ces_2_x_cfrun.append(X[i])
            self.ces_2_y_cfrun.append(Y[i])

    def CESPlot(self):
        
        self.CesValues()

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15, 4))


        #-------------------------Base Run Equilibrium-------------------------------------

        #AGR
        label_base_ces_0 =  "Base Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_base[0], 
                                                                                              self.L_base[0], self.K_base[0], 
                                                                                              1/self.r_base)

        ax1.plot( self.ces_budget_0_x_base, self.ces_budget_0_y_base,  "-k", linewidth = 3, label = label_base_ces_0)
        ax1.plot( self.ces_0_x_base, self.ces_0_y_base, "-k", linewidth = 3)

        ax1.plot([0, self.L_base[0]], [self.K_base[0], self.K_base[0]], "--ok")
        ax1.plot([self.L_base[0], self.L_base[0]], [0, self.K_base[0]], "--ok")

        #SERVICES
        label_base_ces_1 =  "Base Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_base[1], 
                                                                                              self.L_base[1], self.K_base[1], 
                                                                                              1/self.r_base)
        ax2.plot( self.ces_budget_1_x_base, self.ces_budget_1_y_base,  "-k", linewidth = 3, label = label_base_ces_1)
        ax2.plot( self.ces_1_x_base, self.ces_1_y_base, "-k", linewidth = 3)

        ax2.plot([0, self.L_base[1]], [self.K_base[1], self.K_base[1]], "--ok")
        ax2.plot([self.L_base[1], self.L_base[1]], [0, self.K_base[1]], "--ok")


        #INDUSTRY
        label_base_ces_2 =  "Base Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_base[2], 
                                                                                              self.L_base[2], self.K_base[2], 
                                                                                              1/self.r_base)


        ax3.plot(self.ces_budget_2_x_base, self.ces_budget_2_y_base,  "-k", linewidth = 3, label = label_base_ces_2)
        ax3.plot(self.ces_2_x_base, self.ces_2_y_base, "-k", linewidth = 3)

        ax3.plot([0, self.L_base[2]], [self.K_base[2], self.K_base[2]], "--ok")
        ax3.plot([self.L_base[2], self.L_base[2]], [0, self.K_base[2]], "--ok")


        #-------------------------CFRun Equilibrium-------------------------------------
        
        #AGR
        label_cfrun_ces_0 =  "cfrun Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_cfrun[0], 
                                                                                              self.L_cfrun[0], self.K_cfrun[0], 
                                                                                              1/self.r_cfrun)
        
        ax1.plot( self.ces_budget_0_x_cfrun, self.ces_budget_0_y_cfrun,  "-c", linewidth = 1, label = label_cfrun_ces_0)
        ax1.plot( self.ces_0_x_cfrun, self.ces_0_y_cfrun, "-c", linewidth = 1)

        ax1.plot([0, self.L_cfrun[0]], [self.K_cfrun[0], self.K_cfrun[0]], "--oc")
        ax1.plot([self.L_cfrun[0], self.L_cfrun[0]], [0, self.K_cfrun[0]], "--oc")


        #SERVICES
        label_cfrun_ces_1 =  "cfrun Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_cfrun[1], 
                                                                                              self.L_cfrun[1], self.K_cfrun[1], 
                                                                                              1/self.r_cfrun)
        ax2.plot( self.ces_budget_1_x_cfrun, self.ces_budget_1_y_cfrun,  "-c", linewidth = 1, label = label_cfrun_ces_1)
        ax2.plot( self.ces_1_x_cfrun, self.ces_1_y_cfrun, "-c", linewidth = 1)

        ax2.plot([0, self.L_cfrun[1]], [self.K_cfrun[1], self.K_cfrun[1]], "--oc")
        ax2.plot([self.L_cfrun[1], self.L_cfrun[1]], [0, self.K_cfrun[1]], "--oc")


        #INDUSTRY
        label_cfrun_ces_2 =  "cfrun Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_cfrun[2], 
                                                                                              self.L_cfrun[2], self.K_cfrun[2], 
                                                                                              1/self.r_cfrun)
        ax3.plot( self.ces_budget_2_x_cfrun, self.ces_budget_2_y_cfrun,  "-c", linewidth = 2, label =label_cfrun_ces_2 )
        ax3.plot( self.ces_2_x_cfrun, self.ces_2_y_cfrun, "-c", linewidth = 2)

        ax3.plot([0, self.L_cfrun[2]], [self.K_cfrun[2], self.K_cfrun[2]], "--oc")
        ax3.plot([self.L_cfrun[2], self.L_cfrun[2]], [0, self.K_cfrun[2]], "--oc")



        # Axes limits---------------------------------------------------

        #AGR

        # axis1xmin = [self.L_base[0] if self.L_base[0] < self.L_cfrun[0] else self.L_cfrun[0]][0]
        # axis1xmax = [self.L_base[0] if self.L_base[0] > self.L_cfrun[00] else self.L_cfrun[0]][0]

        # axis1ymin = [self.K_base[0] if self.K_base[0] < self.K_cfrun[0] else self.K_cfrun[0]][0]
        # axis1ymax = [self.K_base[0] if self.K_base[0] > self.K_cfrun[0] else self.K_cfrun[0]][0]

        # ax1.set_xlim(xmin = axis1xmin - axis1xmin/5, xmax = axis1xmax + axis1xmax/7)
        # ax1.set_ylim(ymin = 0, ymax = axis1ymax)


        axis2xmin = [self.L_base[1] if self.L_base[1] < self.L_cfrun[1] else self.L_cfrun[1]][0]
        axis2xmax = [self.L_base[1] if self.L_base[1] > self.L_cfrun[1] else self.L_cfrun[1]][0]

        axis2ymin = [self.K_base[1] if self.K_base[1] < self.K_cfrun[1] else self.K_cfrun[1]][0]
        axis2ymax = [self.K_base[1] if self.K_base[1] > self.K_cfrun[1] else self.K_cfrun[1]][0]











        ax1.set_xlim(xmin = 0)
        ax1.set_ylim(ymin = 0)







        ax1.set_title("{} CES Production Function".format(self.sectors[0]), fontweight = "bold", fontsize = 13)
        ax1.set_xlabel("Labor",fontweight = "bold", fontsize = 11)
        ax1.set_ylabel("Capital",fontweight = "bold", fontsize = 11)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.legend(fontsize = 7, edgecolor = "black", ncol = 2)


        #SERVICES
        ax2.set_xlim(xmin = 0)
        ax2.set_ylim(ymin = 0)
        ax2.set_title("{} CES Production Function".format(self.sectors[1]), fontweight = "bold", fontsize = 13)
        ax2.set_xlabel("Labor",fontweight = "bold", fontsize = 11)
        ax2.set_ylabel("Capital",fontweight = "bold", fontsize = 11)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.legend(fontsize = 7, edgecolor = "black", ncol = 2)

        ax2.set_xlim(xmin = axis2xmin - axis2xmin/5, xmax = axis2xmax + axis2xmax/7)
        ax2.set_ylim(ymin = 0, ymax = axis2ymax)

        #INDUSTRY
        ax3.set_xlim(xmin = 0)
        ax3.set_ylim(ymin = 0)
        ax3.set_title("{} CES Production Function".format(self.sectors[2]), fontweight = "bold", fontsize = 13)
        ax3.set_xlabel("Labor",fontweight = "bold", fontsize = 11)
        ax3.set_ylabel("Capital",fontweight = "bold", fontsize = 11)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.legend(fontsize = 7, edgecolor = "black", ncol = 2)

        plt.tight_layout()

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
        df.to_excel("MacroVariables.xlsx")

        return df

    
    def flow_diagram(self, sector_code):

        BaserunResult = [round(float(x), 3) for x in self.result1.x]
        CfrunResult   = [round(float(x), 3) for x in self.result2.x]
    
        # Sektör kodu = 1: Tarım, 2: Ticaret ve hizmet, 3: Sanayi
        
        sectors = {1: "Tarım", 2: "Ticaret Hizmet", 3:"Sanayi"}
        
        values_base = []
        values_cfrun = []
        
        position = [(1,17), (3,17), (2,15), (5,13), 
                (2,10), (5,8), (2,8), (5,7), (2,6), 
                (1,4),
                (5,4)]
        
        variables1 = ["L", "K", "X", "I1", "I2", "I3", "Ir", "Ib",
                    "Z", "E", "D", "M", "Q", "C", "TPAO", "G", "INV",
                    "I_1", "I_2", "I_3", "I_r", "I_b"]
        
        variables = []
        for i in variables1:
            if "_" in i:
                val = i.replace("_", str(sector_code))
                variables.append(val)
            else:
                val = i + str(sector_code)
                variables.append(val)
                
        nodes1 = ["L", "K", "X", "I_cost",
                "Z", "E", "D", "M", "Q", "final",
                "I_income"]
    
        nodes = []
        for i in nodes1:
            if "_" in i:
                val = i.replace("_", str(sector_code))
                nodes.append(val)
            else:
                val = i + str(sector_code)
                nodes.append(val)
                

        Icost      = variables[3:7]
        Iincome    = variables[17:]
        final      = variables[13:17]
        
        
        for i in nodes:
            if i in CGE(SAM).init_values_str:
                values_base.append("{}: {}".format(i, BaserunResult[(CGE(SAM).init_values_str.index(i))]))
                values_cfrun.append("{}: {}".format(i, CfrunResult[(CGE(SAM).init_values_str.index(i))]))

            else:
                values_base.append("")
                values_cfrun.append("")
                
        val1_base = ""
        val1_cfrun = ""
        for y in Icost:
            val1_base = val1_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"
            val1_cfrun = val1_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        val2_base = ""
        val2_cfrun = ""
        for y in Iincome:
            val2_base = val2_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))])+ "\n"
            val2_cfrun = val2_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        val3_base = ""
        val3_cfrun = ""
        for y in final:
            val3_base = val3_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))])+ "\n"  
            val3_cfrun = val3_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"
                
        
        edges1 = [("L", "X"), ("K", "X"),
        ("X", "Z"), ("I_cost", "Z"), ("Z", "E"),("Z", "D"), ("D", "Q"), ("M", "Q"),
        ("Q","final"),
        ("Q", "I_income")]
        
        edges = []
        
        for i in edges1:
            k = []
            
            for j in i:
                
                if "_" in j:
                    val = j.replace("_", str(sector_code))
                    k.append(val)
                else:
                    val = j + str(sector_code)
                    k.append(val)
            edges.append(tuple(k))

        G1 = nx.Graph()  
        G2 = nx.Graph()
            
        for k in range(len(nodes)):
            G1.add_node(nodes[k], pos = position[k], val = values_base[k])
            G2.add_node(nodes[k], pos = position[k], val = values_cfrun[k])
        
        pos1 = nx.get_node_attributes(G1,'pos')
        pos2 = nx.get_node_attributes(G2,'pos')
        
        labels1 = nx.get_node_attributes(G1,'val')
        labels2 = nx.get_node_attributes(G2,'val')
        
        G1.add_edges_from(edges)
        G2.add_edges_from(edges)
        
            
        fig, (ax1, ax2) = plt.subplots(1,2,figsize = (10,6))

        nx.draw_networkx_nodes(G1, pos1, node_size = 1500, node_color = "none", ax = ax1)
        nx.draw_networkx_edges(G1, pos1, edgelist = G1.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax1);
        nx.draw_networkx_labels(G1, pos1,labels = labels1, font_weight = "bold", ax = ax1);
        
        nx.draw_networkx_nodes(G2, pos2, node_size = 1500, node_color = "none", ax = ax2)
        nx.draw_networkx_edges(G2, pos2, edgelist = G2.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax2);
        nx.draw_networkx_labels(G2, pos2,labels = labels2, font_weight = "bold", ax = ax2);
        

        ax1.text(5,11, val1_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax1.text(5,1, val2_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax1.text(.5,1.7, val3_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        
        ax2.text(5,11, val1_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax2.text(5,1, val2_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax2.text(.5,1.7, val3_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
            
        
        ax1.set_title("{} Sektörü Baz Yıl Verileri".format(sectors[sector_code]), fontweight = "bold", 
                    color = "grey")
        ax2.set_title("{} Sektörü Karşı Olgusal Denge Verileri".format(sectors[sector_code]), fontweight = "bold",
                    color = "grey")
        
        ax1.set_xlim(xmin = 0, xmax = 6.5)
        ax1.set_ylim(ymin = 0, ymax = 18)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        
        ax2.set_xlim(xmin = 0, xmax = 6.5)
        ax2.set_ylim(ymin = 0, ymax = 18)
        
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
            
        plt.tight_layout() 
        plt.show()

    def flow_raf(self):

        BaserunResult = [round(float(x), 3) for x in self.result1.x]
        CfrunResult   = [round(float(x), 3) for x in self.result2.x]
       
        values_base = []
        values_cfrun = []

        position = [(1,19), (3,19), (2,17), 
                    (5,19), (7,19), (6,17),
                    (4,15), (5,12),
                    (2,12), (1,10), (3,10), (3,8), (1,6),
                    (1,4), (3,3)             
                    ]

        variables = ["Lr", "Kr", "Xr",
                    "MCOr", "DCOr", "COr",
                    "XCOr", "I1r", "I2r", "I3r", "Irr", "Ibr",
                    "Zr", "Dr", "Er", "Mr", "Qr",
                    "Cr", "Ir1", "Ir2", "Ir3", "Irr"]

        nodes =  ["Lr", "Kr", "Xr",
                "MCOr", "DCOr", "COr",
                "XCOr", "I_cost",
                "Zr", "Dr", "Er", "Mr", "Qr",
                "Cr", "I_income"]

        Icost      = variables[7:12]
        Iincome    = variables[18:]



        for i in nodes:
                if i in CGE(SAM).init_values_str:
                    values_base.append("{}: {}".format(i, BaserunResult[(CGE(SAM).init_values_str.index(i))]))
                    values_cfrun.append("{}: {}".format(i, CfrunResult[(CGE(SAM).init_values_str.index(i))]))

                else:
                    values_base.append("")
                    values_cfrun.append("")

        val1_base = ""
        val1_cfrun = ""
        for y in Icost:
            val1_base = val1_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"
            val1_cfrun = val1_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        val2_base = ""
        val2_cfrun = ""
        for y in Iincome:
            val2_base = val2_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))])+ "\n"
            val2_cfrun = val2_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        edges = [
                ("Lr", "Xr"), ("Kr", "Xr"), ("MCOr", "COr"), ("DCOr", "COr"), ("Xr", "XCOr"), ("COr", "XCOr"),
                ("XCOr", "Zr"), ("I_cost", "Zr"), ("Zr", "Dr"), ("Zr", "Er"), ("Dr", "Qr"), ("Mr", "Qr"),("Qr", "Cr"), ("Qr", "I_income")

                ]   

        G1 = nx.Graph()  
        G2 = nx.Graph()

        for k in range(len(nodes)):
            G1.add_node(nodes[k], pos = position[k], val = values_base[k])
            G2.add_node(nodes[k], pos = position[k], val = values_cfrun[k])

        pos1 = nx.get_node_attributes(G1,'pos')
        pos2 = nx.get_node_attributes(G2,'pos')

        labels1 = nx.get_node_attributes(G1,'val')
        labels2 = nx.get_node_attributes(G2,'val')

        G1.add_edges_from(edges)
        G2.add_edges_from(edges)


        fig, (ax1, ax2) = plt.subplots(1,2,figsize = (10,6))

        nx.draw_networkx_nodes(G1, pos1, node_size = 1500, node_color = "none", ax = ax1)
        nx.draw_networkx_edges(G1, pos1, edgelist = G1.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax1);
        nx.draw_networkx_labels(G1, pos1,labels = labels1, font_weight = "bold", ax = ax1);

        nx.draw_networkx_nodes(G2, pos2, node_size = 1500, node_color = "none", ax = ax2)
        nx.draw_networkx_edges(G2, pos2, edgelist = G2.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax2);
        nx.draw_networkx_labels(G2, pos2,labels = labels2, font_weight = "bold", ax = ax2);


        ax1.text(5,9, val1_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax1.text(3,1, val2_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")

        ax2.text(5,9, val1_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax2.text(3,1, val2_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")



        ax1.set_title("Rafineriler Baz Yıl Verileri", fontweight = "bold", 
                    color = "grey")
        ax2.set_title("Rafineriler Karşı Olgusal Denge Verileri", fontweight = "bold",
                    color = "grey")

        ax1.set_xlim(xmin = 0, xmax = 8)
        ax1.set_ylim(ymin = 0, ymax = 20)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)

        ax2.set_xlim(xmin = 0, xmax = 8)
        ax2.set_ylim(ymin = 0, ymax = 20)

        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        plt.tight_layout()  
        plt.show()

    def flow_botas(self):  

        BaserunResult = [round(float(x), 3) for x in self.result1.x]
        CfrunResult   = [round(float(x), 3) for x in self.result2.x] 

        values_base = []
        values_cfrun = []

        position = [(1,19), (3,19), (2,17), 
                    (5,19), (7,19), (6,17),
                    (4,15), (5,12),
                    (2,12), (1,8), (3,8),
                    (1,4), (3,3)             
                    ]

        variables = ["Lb", "Kb", "Xb",
                    "MNGb", "DNGb", "NGb",
                    "XNGb", "I1b", "I2b", "I3b", "Ibb",
                    "Zb", "Db", "Eb", 
                    "Cb", "Ib1", "Ib2", "Ib3",  "Ibr"]

        nodes =  ["Lb", "Kb", "Xb",
                "MNGb", "DNGb", "NGb",
                "XNGb", "I_cost",
                "Zb", "Db", "Eb",
                "Cb", "I_income"]

        Icost      = variables[7:11]
        Iincome    = variables[15:]



        for i in nodes:
                if i in CGE(SAM).init_values_str:
                    values_base.append("{}: {}".format(i, BaserunResult[(CGE(SAM).init_values_str.index(i))]))
                    values_cfrun.append("{}: {}".format(i, CfrunResult[(CGE(SAM).init_values_str.index(i))]))

                else:
                    values_base.append("")
                    values_cfrun.append("")

        val1_base = ""
        val1_cfrun = ""
        for y in Icost:
            val1_base = val1_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"
            val1_cfrun = val1_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        val2_base = ""
        val2_cfrun = ""
        for y in Iincome:
            val2_base = val2_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))])+ "\n"
            val2_cfrun = val2_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        edges = [
                ("Lb", "Xb"), ("Kb", "Xb"), ("MNGb", "NGb"), ("DNGb", "NGb"), ("Xb", "XNGb"), ("NGb", "XNGb"),
                ("XNGb", "Zb"), ("I_cost", "Zb"), ("Zb", "Db"), ("Zb", "Eb"), ("Db", "Cb"), ("Db", "I_income")

                ]   

        G1 = nx.Graph()  
        G2 = nx.Graph()

        for k in range(len(nodes)):
            G1.add_node(nodes[k], pos = position[k], val = values_base[k])
            G2.add_node(nodes[k], pos = position[k], val = values_cfrun[k])

        pos1 = nx.get_node_attributes(G1,'pos')
        pos2 = nx.get_node_attributes(G2,'pos')

        labels1 = nx.get_node_attributes(G1,'val')
        labels2 = nx.get_node_attributes(G2,'val')

        G1.add_edges_from(edges)
        G2.add_edges_from(edges)


        fig, (ax1, ax2) = plt.subplots(1,2,figsize = (10,6))

        nx.draw_networkx_nodes(G1, pos1, node_size = 1500, node_color = "none", ax = ax1)
        nx.draw_networkx_edges(G1, pos1, edgelist = G1.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax1);
        nx.draw_networkx_labels(G1, pos1,labels = labels1, font_weight = "bold", ax = ax1);

        nx.draw_networkx_nodes(G2, pos2, node_size = 1500, node_color = "none", ax = ax2)
        nx.draw_networkx_edges(G2, pos2, edgelist = G2.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax2);
        nx.draw_networkx_labels(G2, pos2,labels = labels2, font_weight = "bold", ax = ax2);


        ax1.text(5,9, val1_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax1.text(3,1, val2_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")

        ax2.text(5,9, val1_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax2.text(3,1, val2_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")



        ax1.set_title("BOTAŞ Baz Yıl Verileri", fontweight = "bold", 
                    color = "grey")
        ax2.set_title("BOTAŞ Karşı Olgusal Denge Verileri", fontweight = "bold",
                    color = "grey")

        ax1.set_xlim(xmin = 0, xmax = 8)
        ax1.set_ylim(ymin = 0, ymax = 20)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)

        ax2.set_xlim(xmin = 0, xmax = 8)
        ax2.set_ylim(ymin = 0, ymax = 20)

        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        plt.tight_layout()     
        plt.show()  
    


result = CGEResults(2)     # 0: Tarım, 1: Ticaret ve Hizmet, 2: Sanayi
print(result.MacroVariables())
print("----------------------")
print(result.EV_CV_Calculation())

result.flow_diagram(3)     # 1: Tarım, 2: Ticaret ve Hizmet, 3: Sanayi
result.flow_raf()
result.flow_botas()




