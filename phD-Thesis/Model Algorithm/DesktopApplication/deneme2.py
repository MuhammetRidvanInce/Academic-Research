from scipy.optimize import minimize
import pandas as pd
import numpy as np
import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import tkinter.font as tkFont
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import matplotlib.pyplot as plt



SAM = pd.read_excel("SHM.xlsx", 
                  index_col = "index")

class ModelERExog():
    
    def __init__(self, SAM):  
        self.SAM = SAM
        shm = self.SAM.loc
        self.AlgorithmResult = None

        # DIŞŞSAl DEĞİŞKENLER
        ## Üretim Faktörleri
        self.DCOBar = shm["dco", "raf"] 
        self.DNGBar = shm["dng", "bts"]
        self.Lbar   = shm["hh", "lab"]
        self.Kbar   = shm["hh", "cap"]

        ## Yabancı Tasarruf # SF Yabancı Dışsal Değişken Değil

        ## Dünya Fiyatları
        self.Pwe1, self.Pwe2, self.Pwe3, self.Pwe4, self.Pwe5, self.Pwer, self.Pweb, \
        self.Pwm1, self.Pwm2, self.Pwm3, self.Pwm4, self.Pwm5, self.Pwmr, \
        self.Pwmco, self.Pwmng, self.epsilon = 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1

        ## Dönüşüm esneklikleri
        self.psi1 = 2
        self.psi2 = 2
        self.psi3 = 2
        self.psi4 = 2
        self.psi5 = 0.6
        self.psir = 0.6
        self.psib = 10 
        
        ## İkame Esneklikleri
        self.sigma1   = 2
        self.sigma2   = 2
        self.sigma3   = 2
        self.sigma4   = 2
        self.sigma5   = 3.2
        self.sigmar   = 2 # Rafineriler D / M arasındaki ikame petrol ürünleri
        self.sigmaco  = 10 # İthal hampetrol / yurtiçi hampetrol arasındaki ikame
        self.sigmang  = 10 # İthal doğalgaz / yurtiçi doğalgaz arasındaki ikame

        self.sigmaxco = 0.01 # Kompozit faktör / ham petrol arasındaki ikame
        self.sigmaxng = 0.01 # Kompozit faktör / ham petrol arasındaki ikame

        # BAŞLANGIÇ DEĞERLERİ (BAZ TIL DEĞERLERİ)
        ## Fiyatlar
        px1, px2, px3, px4, px5, pxr, pxb, pz1, pz2, pz3, pz4, pz5, pzr, pzb, \
        pe1, pe2, pe3, pe4, pe5, per, peb, pd1, pd2, pd3, pd4, pd5, pdr, pdb, \
        pq1, pq2, pq3, pq4, pq5, pqr, pm1, pm2, pm3, pm4, pm5, pmr, pmco, pdco, \
        pco, pxco, pmng, pdng, png, pxng, r, w = np.ones(50)

        ## Diğer Başlangıç (Baz Yıl) Değerleri
        Y = w*self.Lbar + r*self.Kbar
        OIL_INCOME = pdco*self.DCOBar + pdng*self.DNGBar
        L1 = shm["lab", "s1"]
        L2 = shm["lab", "s2"]
        L3 = shm["lab", "s3"]
        L4 = shm["lab", "s4"]
        L5 = shm["lab", "s5"]
        Lr = shm["lab", "raf"]
        Lb = shm["lab", "bts"]
        K1 = shm["cap", "s1"]
        K2 = shm["cap", "s2"]
        K3 = shm["cap", "s3"]
        K4 = shm["cap", "s4"]
        K5 = shm["cap", "s5"]
        Kr = shm["cap", "raf"]
        Kb = shm["cap", "bts"]
        X1 = L1 + K1
        X2 = L2 + K2
        X3 = L3 + K3
        X4 = L4 + K4
        X5 = L5 + K5
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
        I11 = shm["s1", "s1"]
        I21 = shm["s2", "s1"]
        I31 = shm["s3", "s1"]
        I41 = shm["s4", "s1"]
        I51 = shm["s5", "s1"]
        Ir1 = shm["raf", "s1"]
        Ib1 = shm["bts", "s1"]
        I12 = shm["s1", "s2"]
        I22 = shm["s2", "s2"]
        I32 = shm["s3", "s2"]
        I42 = shm["s4", "s2"]
        I52 = shm["s5", "s2"]
        Ir2 = shm["raf", "s2"]
        Ib2 = shm["bts", "s2"]
        I13 = shm["s1", "s3"]
        I23 = shm["s2", "s3"]
        I33 = shm["s3", "s3"]
        I43 = shm["s4", "s3"]
        I53 = shm["s5", "s3"]
        Ir3 = shm["raf", "s3"]
        Ib3 = shm["bts", "s3"]
        I14 = shm["s1", "s4"]
        I24 = shm["s2", "s4"]
        I34 = shm["s3", "s4"]
        I44 = shm["s4", "s4"]
        I54 = shm["s5", "s4"]
        Ir4 = shm["raf", "s4"]
        Ib4 = shm["bts", "s4"]
        I15 = shm["s1", "s5"]
        I25 = shm["s2", "s5"]
        I35 = shm["s3", "s5"]
        I45 = shm["s4", "s5"]
        I55 = shm["s5", "s5"]
        Ir5 = shm["raf", "s5"]
        Ib5 = shm["bts", "s5"]
        I1r = shm["s1", "raf"]
        I2r = shm["s2", "raf"]
        I3r = shm["s3", "raf"]
        I4r = shm["s4", "raf"]
        I5r = shm["s5", "raf"]
        Irr = shm["raf", "raf"]
        Ibr = shm["bts", "raf"]
        I1b = shm["s1", "bts"]
        I2b = shm["s2", "bts"]
        I3b = shm["s3", "bts"]
        I4b = shm["s4", "bts"]
        I5b = shm["s5", "bts"]
        Irb = shm["raf", "bts"]
        Ibb = shm["bts", "bts"]
        Z1 = X1 + I11 + I21 + I31 + I41 + I51 + Ir1 + Ib1
        Z2 = X2 + I12 + I22 + I32 + I42 + I52 + Ir2 + Ib2
        Z3 = X3 + I13 + I23 + I33 + I43 + I53 + Ir3 + Ib3
        Z4 = X4 + I14 + I24 + I34 + I44 + I54 + Ir4 + Ib4
        Z5 = X5 + I15 + I25 + I35 + I45 + I55 + Ir5 + Ib5
        Zr = XCOr + I1r + I2r + I3r + I4r + I5r + Irr + Ibr
        Zb = XNGb + I1b + I2b + I3b + I4b + I5b + Irb + Ibb
        E1 = shm["s1", "exp"]
        E2 = shm["s2", "exp"]
        E3 = shm["s3", "exp"]
        E4 = shm["s4", "exp"]
        E5 = shm["s5", "exp"]
        Er = shm["raf", "exp"]
        Eb = shm["bts", "exp"]
        M1 = shm["imp", "s1"]
        M2 = shm["imp", "s2"]
        M3 = shm["imp", "s3"]
        M4 = shm["imp", "s4"]
        M5 = shm["imp", "s5"]
        Mr = shm["imp", "raf"]
        Td = shm["dtax", "hh"]
        Tva1 = shm["gtax", "s1"]
        Tva2 = shm["gtax", "s2"]
        Tva3 = shm["gtax", "s3"]
        Tva4 = shm["gtax", "s4"]
        Tva5 = shm["gtax", "s5"]
        Tvar = shm["gtax", "raf"]
        Tvab = shm["gtax", "bts"]
        Tva = Tva1 + Tva2 + Tva3 + Tva4 + Tva5 + Tvar + Tvab
        Tz1 = shm["ptax", "s1"]
        Tz2 = shm["ptax", "s2"]
        Tz3 = shm["ptax", "s3"]
        Tz4 = shm["ptax", "s4"]
        Tz5 = shm["ptax", "s5"]
        Tzr = shm["ptax", "raf"]
        Tzb = shm["ptax", "bts"]
        Tz = Tz1 + Tz2 + Tz3 + Tz4 + Tz5 + Tzr + Tzb
        Tm1 = shm["trf", "s1"]
        Tm2 = shm["trf", "s2"]
        Tm3 = shm["trf", "s3"]
        Tm4 = shm["trf", "s4"]
        Tm5 = shm["trf", "s5"]
        Tm = Tm1 + Tm2 + Tm3 + Tm4 + Tm5 
        T = Td + Tz + Tva + Tm
        D1 = Z1 + (Tva1 + Tz1) - E1
        D2 = Z2 + (Tva2 + Tz2) - E2
        D3 = Z3 + (Tva3 + Tz3) - E3
        D4 = Z4 + (Tva4 + Tz4) - E4
        D5 = Z5 + (Tva5 + Tz5) - E5
        Dr = Zr + (Tvar + Tzr) - Er
        Db = Zb + (Tvab + Tzb) - Eb
        C1 = shm["s1", "hh"]
        C2 = shm["s2", "hh"]
        C3 = shm["s3", "hh"]
        C4 = shm["s4", "hh"]
        C5 = shm["s5", "hh"]
        Cr = shm["raf", "hh"]
        Cb = shm["bts", "hh"]
        TPAO1 = shm["s1", "TPAO"]
        TPAO2 = shm["s2", "TPAO"]
        TPAO3 = shm["s3", "TPAO"]
        TPAO4 = shm["s4", "TPAO"]
        TPAO5 = shm["s5", "TPAO"]
        G1 = shm["s1", "gov"]
        G2 = shm["s2", "gov"]
        G3 = shm["s3", "gov"]
        G4 = shm["s4", "gov"]
        G5 = shm["s5", "gov"]
        INV1 = shm["s1", "inv"]
        INV2 = shm["s2", "inv"]
        INV3 = shm["s3", "inv"]
        INV4 = shm["s4", "inv"]
        INV5 = shm["s5", "inv"]
        Q1 = C1 + TPAO1 + G1 + INV1 + I11 + I12 + I13 + I14 + I15 + I1r + I1b
        Q2 = C2 + TPAO2 + G2 + INV2 + I21 + I22 + I23 + I24 + I25 + I2r + I2b
        Q3 = C3 + TPAO3 + G3 + INV3 + I31 + I32 + I33 + I34 + I35 + I3r + I3b
        Q4 = C4 + TPAO4 + G4 + INV4 + I41 + I42 + I43 + I44 + I45 + I4r + I4b
        Q5 = C5 + TPAO5 + G5 + INV5 + I51 + I52 + I53 + I54 + I55 + I5r + I5b
        Qr = Cr + Ir1 + Ir2 + Ir3 + Ir4 + Ir5 + Irr + Irb
        Db = Cb + Ib1 + Ib2 + Ib3 + Ib4 + Ib5 + Ibr + Ibb
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
        self.tva4 = Tva4 / Z4
        self.tva5 = Tva5 / Z5
        self.tvar = Tvar / Zr
        self.tvab = Tvab / Zb

        self.tz1 = Tz1 / Z1
        self.tz2 = Tz2 / Z2
        self.tz3 = Tz3 / Z3
        self.tz4 = Tz4 / Z4
        self.tz5 = Tz5 / Z5
        self.tzr = Tzr / Zr
        self.tzb = Tzb / Zb

        self.tm1 = Tm1 / M1
        self.tm2 = Tm2 / M2
        self.tm3 = Tm3 / M3
        self.tm4 = Tm4 / M4
        self.tm5 = Tm5 / M5

        self.alphal1 = L1 / (L1 + K1)
        self.alphal2 = L2 / (L2 + K2)
        self.alphal3 = L3 / (L3 + K3)
        self.alphal4 = L4 / (L4 + K4)
        self.alphal5 = L5 / (L5 + K5)
        self.alphalr = Lr / (Lr + Kr)
        self.alphalb = Lb / (Lb + Kb)

        self.alphak1 = K1 / (L1 + K1)
        self.alphak2 = K2 / (L2 + K2)
        self.alphak3 = K3 / (L3 + K3)
        self.alphak4 = K4 / (L4 + K4)
        self.alphak5 = K5 / (L5 + K5)
        self.alphakr = Kr / (Lr + Kr)
        self.alphakb = Kb / (Lb + Kb)

        self.A1 = X1 / (L1**self.alphal1 * K1**self.alphak1)
        self.A2 = X2 / (L2**self.alphal2 * K2**self.alphak2)
        self.A3 = X3 / (L3**self.alphal3 * K3**self.alphak3)
        self.A4 = X4 / (L4**self.alphal4 * K4**self.alphak4)
        self.A5 = X5 / (L5**self.alphal5 * K5**self.alphak5)
        self.Ar = Xr / (Lr**self.alphalr * Kr**self.alphakr)
        self.Ab = Xb / (Lb**self.alphalb * Kb**self.alphakb)

        # self.alphaxr = Xr / (Xr + COr)
        # self.alphaco = COr / (Xr + COr)
        # self.Axco = XCOr / (Xr**self.alphaxr * COr**self.alphaco)

        self.etaxco = (self.sigmaxco - 1) / self.sigmaxco
        self.xr   = Xr**(1-self.etaxco) / (Xr**(1-self.etaxco) + COr**(1-self.etaxco))
        self.co   = COr**(1-self.etaxco) / (Xr**(1-self.etaxco) + COr**(1-self.etaxco))
        self.lambdaxco = XCOr / (self.xr*Xr**self.etaxco + self.co*COr**self.etaxco)**(1/self.etaxco)

        # self.alphaxb = Xb / (Xb + NGb)
        # self.alphang = NGb / (Xb + NGb)
        # self.Axng = XNGb / (Xb**self.alphaxb * NGb**self.alphang)

        self.etaxng = (self.sigmaxng - 1) / self.sigmaxng
        self.xb   = Xb**(1-self.etaxng) / (Xb**(1-self.etaxng) + NGb**(1-self.etaxng))
        self.ng   = NGb**(1-self.etaxng) / (Xb**(1-self.etaxng) + NGb**(1-self.etaxng))
        self.lambdaxng = XNGb / (self.xb*Xb**self.etaxng + self.ng*NGb**self.etaxng)**(1/self.etaxng)

        self.a11 = I11 / Z1
        self.a21 = I21 / Z1
        self.a31 = I31 / Z1
        self.a41 = I41 / Z1
        self.a51 = I51 / Z1
        self.ar1 = Ir1 / Z1
        self.ab1 = Ib1 / Z1
        self.a12 = I12 / Z2
        self.a22 = I22 / Z2
        self.a32 = I32 / Z2
        self.a42 = I42 / Z2
        self.a52 = I52 / Z2
        self.ar2 = Ir2 / Z2
        self.ab2 = Ib2 / Z2
        self.a13 = I13 / Z3
        self.a23 = I23 / Z3
        self.a33 = I33 / Z3
        self.a43 = I43 / Z3
        self.a53 = I53 / Z3
        self.ar3 = Ir3 / Z3
        self.ab3 = Ib3 / Z3
        self.a14 = I14 / Z4
        self.a24 = I24 / Z4
        self.a34 = I34 / Z4
        self.a44 = I44 / Z4
        self.a54 = I54 / Z4
        self.ar4 = Ir4 / Z4
        self.ab4 = Ib4 / Z4
        self.a15 = I15 / Z5
        self.a25 = I25 / Z5
        self.a35 = I35 / Z5
        self.a45 = I45 / Z5
        self.a55 = I55 / Z5
        self.ar5 = Ir5 / Z5
        self.ab5 = Ib5 / Z5
        self.a1r = I1r / Zr
        self.a2r = I2r / Zr
        self.a3r = I3r / Zr
        self.a4r = I4r / Zr
        self.a5r = I5r / Zr
        self.arr = Irr / Zr
        self.abr = Ibr / Zr
        self.a1b = I1b / Zb
        self.a2b = I2b / Zb
        self.a3b = I3b / Zb
        self.a4b = I4b / Zb
        self.a5b = I5b / Zb
        self.arb = Irb / Zb
        self.abb = Ibb / Zb

        self.x1 = X1 / Z1
        self.x2 = X2 / Z2
        self.x3 = X3 / Z3
        self.x4 = X4 / Z4
        self.x5 = X5 / Z5
        
        self.xcor = XCOr / Zr
        self.xngb = XNGb / Zb

        self.rho1 = (self.psi1 + 1) / self.psi1
        self.rho2 = (self.psi2 + 1) / self.psi2
        self.rho3 = (self.psi3 + 1) / self.psi3
        self.rho4 = (self.psi4 + 1) / self.psi4
        self.rho5 = (self.psi5 + 1) / self.psi5
        self.rhor = (self.psir + 1) / self.psir
        self.rhob = (self.psib + 1) / self.psib

        self.eta1 = (self.sigma1 - 1) / self.sigma1
        self.eta2 = (self.sigma2 - 1) / self.sigma2
        self.eta3 = (self.sigma3 - 1) / self.sigma3
        self.eta4 = (self.sigma4 - 1) / self.sigma4
        self.eta5 = (self.sigma5 - 1) / self.sigma5
        self.etar = (self.sigmar - 1) / self.sigmar
        self.etaco = (self.sigmaco - 1) / self.sigmaco
        self.etang = (self.sigmang - 1) / self.sigmang


        self.e1 = E1**(1-self.rho1) / (E1**(1-self.rho1) + D1**(1-self.rho1))
        self.e2 = E2**(1-self.rho2) / (E2**(1-self.rho2) + D2**(1-self.rho2))
        self.e3 = E3**(1-self.rho3) / (E3**(1-self.rho3) + D3**(1-self.rho3))
        self.e4 = E4**(1-self.rho4) / (E4**(1-self.rho4) + D4**(1-self.rho4))
        self.e5 = E5**(1-self.rho5) / (E5**(1-self.rho5) + D5**(1-self.rho5))
        self.er = Er**(1-self.rhor) / (Er**(1-self.rhor) + Dr**(1-self.rhor))
        self.eb = Eb**(1-self.rhob) / (Eb**(1-self.rhob) + Db**(1-self.rhob))

        self.dt1 = D1**(1-self.rho1) / (E1**(1-self.rho1) + D1**(1-self.rho1))
        self.dt2 = D2**(1-self.rho2) / (E2**(1-self.rho2) + D2**(1-self.rho2))
        self.dt3 = D3**(1-self.rho3) / (E3**(1-self.rho3) + D3**(1-self.rho3))
        self.dt4 = D4**(1-self.rho4) / (E4**(1-self.rho4) + D4**(1-self.rho4))
        self.dt5 = D5**(1-self.rho5) / (E5**(1-self.rho5) + D5**(1-self.rho5))
        self.dtr = Dr**(1-self.rhor) / (Er**(1-self.rhor) + Dr**(1-self.rhor))
        self.dtb = Db**(1-self.rhob) / (Eb**(1-self.rhob) + Db**(1-self.rhob))

        self.theta1 = Z1 / (self.e1*E1**self.rho1 + self.dt1*D1**self.rho1)**(1/self.rho1)
        self.theta2 = Z2 / (self.e2*E2**self.rho2 + self.dt2*D2**self.rho2)**(1/self.rho2)
        self.theta3 = Z3 / (self.e3*E3**self.rho3 + self.dt3*D3**self.rho3)**(1/self.rho3)
        self.theta4 = Z4 / (self.e4*E4**self.rho4 + self.dt4*D4**self.rho4)**(1/self.rho4)
        self.theta5 = Z5 / (self.e5*E5**self.rho5 + self.dt5*D5**self.rho5)**(1/self.rho5)
        self.thetar = Zr / (self.er*Er**self.rhor + self.dtr*Dr**self.rhor)**(1/self.rhor)
        self.thetab = Zb / (self.eb*Eb**self.rhob + self.dtb*Db**self.rhob)**(1/self.rhob)

        self.m1   = (1+self.tm1)*M1**(1-self.eta1) / ((1+self.tm1)*M1**(1-self.eta1) + D1**(1-self.eta1))
        self.m2   = (1+self.tm2)*M2**(1-self.eta2) / ((1+self.tm2)*M2**(1-self.eta2) + D2**(1-self.eta2))
        self.m3   = (1+self.tm3)*M3**(1-self.eta3) / ((1+self.tm3)*M3**(1-self.eta3) + D3**(1-self.eta3))
        self.m4   = (1+self.tm4)*M4**(1-self.eta4) / ((1+self.tm4)*M4**(1-self.eta4) + D4**(1-self.eta4))
        self.m5   = (1+self.tm5)*M5**(1-self.eta5) / ((1+self.tm5)*M5**(1-self.eta5) + D5**(1-self.eta5))
        self.mr   = Mr**(1-self.etar) / (Mr**(1-self.etar) + Dr**(1-self.etar))
        self.mcor = MCOr**(1-self.etaco) / (MCOr**(1-self.etaco) + DCOr**(1-self.etaco))
        self.mngb = MNGb**(1-self.etang) / (MNGb**(1-self.etang) + DNGb**(1-self.etang))

        self.da1   = D1**(1-self.eta1) / ((1+self.tm1)*M1**(1-self.eta1) + D1**(1-self.eta1))
        self.da2   = D2**(1-self.eta2) / ((1+self.tm2)*M2**(1-self.eta2) + D2**(1-self.eta2))
        self.da3   = D3**(1-self.eta3) / ((1+self.tm3)*M3**(1-self.eta3) + D3**(1-self.eta3))
        self.da4   = D4**(1-self.eta4) / ((1+self.tm4)*M4**(1-self.eta4) + D4**(1-self.eta4))
        self.da5   = D5**(1-self.eta5) / ((1+self.tm5)*M5**(1-self.eta5) + D5**(1-self.eta5))
        self.dar   = Dr**(1-self.etar) / (Mr**(1-self.etar) + Dr**(1-self.etar))
        self.dcor  = DCOr**(1-self.etaco) / (MCOr**(1-self.etaco) + DCOr**(1-self.etaco))
        self.dngb  = DNGb**(1-self.etang) / (MNGb**(1-self.etang) + DNGb**(1-self.etang))

        self.lambda1 = Q1 / (self.m1*M1**self.eta1 + self.da1*D1**self.eta1)**(1/self.eta1)
        self.lambda2 = Q2 / (self.m2*M2**self.eta2 + self.da2*D2**self.eta2)**(1/self.eta2)
        self.lambda3 = Q3 / (self.m3*M3**self.eta3 + self.da3*D3**self.eta3)**(1/self.eta3)
        self.lambda4 = Q4 / (self.m4*M4**self.eta4 + self.da4*D4**self.eta4)**(1/self.eta4)
        self.lambda5 = Q5 / (self.m5*M5**self.eta5 + self.da5*D5**self.eta5)**(1/self.eta5)
        self.lambdar = Qr / (self.mr*Mr**self.etar + self.dar*Dr**self.etar)**(1/self.etar)
        self.lambdaco = COr / (self.mcor*MCOr**self.etaco + self.dcor*DCOr**self.etaco)**(1/self.etaco)
        self.lambdang = NGb / (self.mngb*MNGb**self.etang + self.dngb*DNGb**self.etang)**(1/self.etang)

        self.c1 = C1 / Yd
        self.c2 = C2 / Yd
        self.c3 = C3 / Yd
        self.c4 = C4 / Yd
        self.c5 = C5 / Yd
        self.cr = Cr / Yd
        self.cb = Cb / Yd

        self.mu1 = TPAO1 / OIL_INCOME 
        self.mu2 = TPAO2 / OIL_INCOME 
        self.mu3 = TPAO3 / OIL_INCOME 
        self.mu4 = TPAO4 / OIL_INCOME 
        self.mu5 = TPAO5 / OIL_INCOME 

        self.g1 = G1 / (T - Sg)
        self.g2 = G2 / (T - Sg)
        self.g3 = G3 / (T - Sg)
        self.g4 = G4 / (T - Sg)
        self.g5 = G5 / (T - Sg)

        self.inv1 = INV1 / S
        self.inv2 = INV2 / S
        self.inv3 = INV3 / S
        self.inv4 = INV4 / S
        self.inv5 = INV5 / S

        self.sp = Sp / Y
        self.sg = Sg / T
        
        self.init_values =  [
            X1, L1, K1, I11, I21, I31, I41, I51, Ir1, Ib1, Z1,
            E1, D1, Q1, M1, X2, L2, K2, I12, I22, I32, I42, 
            I52, Ir2, Ib2, Z2, E2, D2, Q2, M2, X3, L3, K3, 
            I13, I23, I33, I43, I53, Ir3, Ib3, Z3, E3, D3, Q3, 
            M3, X4, L4, K4, I14, I24, I34, I44, I54, Ir4, 
            Ib4, Z4, E4, D4, Q4, M4, X5, L5, K5, I15, I25, 
            I35, I45, I55, Ir5, Ib5, Z5, E5, D5, Q5, M5, Xr, 
            Lr, Kr, COr, MCOr, DCOr, XCOr, I1r, I2r, I3r, I4r, 
            I5r, Irr, Ibr, Zr, Er, Dr, Qr, Mr, Xb, Lb, Kb, 
            NGb, MNGb, DNGb, XNGb, I1b, I2b, I3b, I4b, I5b, Irb, 
            Ibb, Zb, Eb, Db, C1, C2, C3, C4, C5, Cr, Cb, Yd,
            Y, TPAO1, TPAO2, TPAO3, TPAO4, TPAO5, OIL_INCOME, G1, G2, 
            G3, G4, G5, T, Td, Tz, Tva, Tm, Tz1, Tz2, Tz3, 
            Tz4, Tz5, Tzr, Tzb, Tva1, Tva2, Tva3, Tva4, Tva5, Tvar, 
            Tvab, Tm1, Tm2, Tm3, Tm4, Tm5, INV1, INV2, INV3, INV4, 
            INV5, S, Sp, Sg, px1, px2, px3, px4, px5, pxr, pxb, 
            pz1, pz2, pz3, pz4, pz5, pzr, pzb, pe1, pe2, pe3, 
            pe4, pe5, per, peb, pd1, pd2, pd3, pd4, pd5, pdr, pdb, 
            pq1, pq2, pq3, pq4, pq5, pqr, pm1, pm2, pm3, pm4, 
            pm5, pmr, pmco, pdco, pco, pxco, pmng, pdng, png, pxng, 
            Sf, r 
        ]
        
        self.init_values_str = {
            "X1": "Tarım sektörü kompozit faktör kullanım miktarı",
            "L1": "Tarım sektörü emek kullanım miktarı",
            "K1":"Tarım sektörü sermaye kullanım miktarı",
            "I11":"Ara girdi miktarı (Tarım --> Tarım)",
            "I21":"Ara girdi miktarı (Tic ve hizmet --> Tarım)",
            "I31": "Ara girdi miktarı (Ulaşım --> Tarım)",
            "I41": "Ara girdi miktarı (İnşaat --> Tarım)",
            "I51": "Ara girdi miktarı (Sanayi --> Tarım)",
            "Ir1": "Ara girdi miktarı (Rafineriler --> Tarım)",
            "Ib1": "Ara girdi miktarı (BOTAŞ --> Tarım)",
            "Z1": "Tarım sektörü toplam gayrisafi üretim miktarı",
            "E1": "Tarım sektörü ihracat miktarı",
            "D1": "Tarım sektörü yurtiçi arz miktarı",
            "Q1": "Tarım sektörü kompozit mal üretim miktarı",
            "M1":"Tarım sektörü ithalat miktarı",

            "X2": "Tic. ve hizmet sektörü kompozit faktör kullanım miktar",
            "L2": "Tic. ve hizmet sektörü emek kullanım miktarı",
            "K2": "Tic. ve hizmet sektörü sermaye kullanım miktarı",
            "I12": "Ara girdi miktarı (Tarım --> Tic. ve hizmet)",
            "I22": "Ara girdi miktarı (Tic. ve hizmet --> Tic. ve hizmet)",
            "I32": "Ara girdi miktarı (Ulşaım --> Tic. ve hizmet)",
            "I42": "Ara girdi miktarı (İnşaat --> Tic. ve hizmet)",
            "I52": "Ara girdi miktarı (Sanayi --> Tic. ve hizmet)",
            "Ir2": "Ara girdi miktarı (Rafineriler --> Tic. ve hizmet)",
            "Ib2": "Ara girdi miktarı (BOTAŞ --> Tic. ve hizmet)",
            "Z2": "Tic. ve hizmet sektörü toplam gayrisafi üretim miktarı",
            "E2":"Tic. ve hizmet sektörü ihracat miktarı",
            "D2":"Tic. ve hizmet sektörü yurtiçi arz miktarı",
            "Q2":"Tic. ve hizmet sektörü kompozit mal üretim miktarı",
            "M2":"Tic. ve hizmet sektörü ithalat miktarı",

            "X3":"Ulaşım sektörü kompozit faktör kullanım miktarı",
            "L3":"Ulaşım sektörü emek kullanım miktarı",
            "K3":"Ulaşım sektörü sermaye kullanım miktarı",
            "I13":"Ara girdi miktarı (Tarım --> Ulaşım)",
            "I23":"Ara girdi miktarı (Tic ve hizmet --> Ulaşım)",
            "I33":"Ara girdi miktarı (Ulaşım --> Ulaşım)",
            "I43":"Ara girdi miktarı (İnşaat --> Ulaşım)",
            "I53":"Ara girdi miktarı (Sanayi --> Ulaşım)",
            "Ir3":"Ara girdi miktarı (Rafineriler --> Ulaşım)",
            "Ib3":"Ara girdi miktarı (BOTAŞ --> Ulaşım)",
            "Z3":"Ulaşım sektörü toplam gayrisafi üretim miktarı",
            "E3":"Ulaşım sektörü ihracat miktarı",
            "D3":"Ulaşım sektörü yurtiçi arz miktarı",
            "Q3":"Ulaşım sektörü kompozit mal üretim miktarı",
            "M3":"Ulaşım sektörü ithalat miktarı",

            "X4":"İnşaat sektörü kompozit faktör kullanım miktarı",
            "L4":"İnşaat sektörü emek kullanım miktarı",
            "K4":"İnşaat sektörü sermaye kullanım miktarı",
            "I14":"Ara girdi miktarı (Tarım --> İnşaat)",
            "I24":"Ara girdi miktarı (Tic ve hizmet --> İnşaat)",
            "I34":"Ara girdi miktarı (Ulaşım --> İnşaat)",
            "I44":"Ara girdi miktarı (İnşaat --> İnşaat)",
            "I54":"Ara girdi miktarı (Sanayi --> İnşaat)",
            "Ir4":"Ara girdi miktarı (Rafineriler --> İnşaat)",
            "Ib4":"Ara girdi miktarı (BOTAŞ --> İnşaat)",
            "Z4":"İnşaat sektörü toplam gayrisafi üretim miktarı",
            "E4":"İnşaat sektörü ihracat miktarı",
            "D4":"İnşaat sektörü yurtiçi arz miktarı",
            "Q4":"İnşaat sektörü kompozit mal üretim miktarı",
            "M4":"İnşaat sektörü ithalat miktarı",

            "X5":"Sanayi sektörü kompozit faktör kullanım miktarı",
            "L5":"Sanayi sektörü emek kullanım miktarı",
            "K5":"Sanayi sektörü sermaye kullanım miktarı",
            "I15":"Ara girdi miktarı (Tarım --> Sanayi)",
            "I25":"Ara girdi miktarı (Tic ve hizmet --> Sanayi)",
            "I35":"Ara girdi miktarı (Ulaşım --> Sanayi)",
            "I45":"Ara girdi miktarı (İnşaat --> Sanayi)",
            "I55":"Ara girdi miktarı (Sanayi --> Sanayi)",
            "Ir5":"Ara girdi miktarı (Rafineriler --> Sanayi)",
            "Ib5":"Ara girdi miktarı (BOTAŞ --> Sanayi)",
            "Z5":"Sanayi sektörü toplam gayrisafi üretim miktarı",
            "E5":"Sanayi sektörü ihracat miktarı",
            "D5":"Sanayi sektörü yurtiçi arz miktarı",
            "Q5":"Sanayi sektörü kompozit mal üretim miktarı",
            "M5":"Sanayi sektörü ithalat miktarı",

            "Xr":"Rafineriler kompozit faktör kullanım miktarı",
            "Lr":"Rafineriler emek kullanım miktarı",
            "Kr":"Rafineriler sermaye kullanım miktarı",
            "COr":"Rafineriler kompozit ham petrol miktarı",
            "MCOr":"Rafineriler ithal ham petrol miktarı",
            "DCOr":"Rafineriler yurtiçi ham petrol miktarı",
            "XCOr":"Rafineriler kompozit hampetrol enerji miktarı",
            "I1r":"Ara girdi miktarı (Tarım --> Rafineriler)",
            "I2r":"Ara girdi miktarı (Tic ve hizmet --> Rafineriler)",
            "I3r":"Ara girdi miktarı (Ulaşım --> Rafineriler)",
            "I4r":"Ara girdi miktarı (İnşaat --> Rafineriler)",
            "I5r":"Ara girdi miktarı (Sanayi --> Rafineriler)",
            "Irr":"Ara girdi miktarı (Rafineriler --> Rafineriler)",
            "Ibr":"Ara girdi miktarı (BOTAŞ --> Rafineriler)",
            "Zr":"Rafineriler toplam gayrisafi üretim miktarı",
            "Er":"Rafineriler ihracat miktarı",
            "Dr":"Rafineriler yurtiçi arz miktarı",
            "Qr":"Rafineriler kompozit mal üretim miktarı",
            "Mr":"Rafineriler ithalat miktarı",

            "Xb":"BOTAŞ kompozit faktör kullanım miktarı",
            "Lb":"BOTAŞ emek kullanım miktarı",
            "Kb":"BOTAŞ sermaye kullanım miktarı",
            "NGb":"BOTAŞ kompozit doğalgaz miktarı",
            "MNGb":"BOTAŞ ithal doğalgaz miktarı",
            "DNGb":"BOTAŞ yurtiçi dogalgaz miktarı",
            "XNGb":"BOTAŞ kompozit doğalgaz enerji miktarı",
            "I1b":"Ara girdi miktarı (Tarım --> BOTAŞ)",
            "I2b":"Ara girdi miktarı (Tic ve hizmet --> BOTAŞ)",
            "I3b":"Ara girdi miktarı (Ulaşım --> BOTAŞ)",
            "I4b":"Ara girdi miktarı (İnşaat --> BOTAŞ)",
            "I5b":"Ara girdi miktarı (Sanayi --> BOTAŞ)",
            "Irb":"Ara girdi miktarı (Rafineriler --> BOTAŞ)",
            "Ibb":"Ara girdi miktarı (BOTAŞ --> BOTAŞ)",
            "Zb":"BOTAŞ toplam gayrisafi üretim miktarı",
            "Eb":"BOTAŞ ihracat miktarı",
            "Db":"BOTAŞ yurtiçi arz miktarı",

            "C1":"Hanehalkı tarım sektörü malına ait talap miktarı",
            "C2":"Hanehalkı Tic ve hizmet sektörü malına ait talap miktarı",
            "C3":"Hanehalkı Ulaşım sektörü malına ait talap miktarı",
            "C4":"Hanehalkı inşaat sektörü malına ait talap miktarı",
            "C5":"Hanehalkı sanayi sektörü malına ait talap miktarı",
            "Cr":"Hanehalkı petrol ürünü talap miktarı",
            "Cb":"Hanehalkı doğalgaz talap miktarı",
            "Yd":"Hanehalkı harcanabilir geliri",
            "Y":"Hanehalkı toplam geliri",
            "TPAO1":"TPAO tarım ürünü talep miktarı",
            "TPAO2":"TPAO tic ve hizmet talep miktarı",
            "TPAO3":"TPAO ulaşım sektörü talep miktarı",
            "TPAO4":"TPAO inşaat talep miktarı",
            "TPAO5":"TPAO sanayi ürünü talep miktarı",
            "OIL_INCOME":"TPAO enerji gelirleri",
            "G1":"Kamu kesimi tarım ürünü talep miktarı",
            "G2":"Kamu kesimi tic ve hizmet talep miktarı",
            "G3":"Kamu kesimi ulaşım talep miktarı",
            "G4":"Kamu kesimi inşaat talep miktarı",
            "G5":"Kamu kesimi sanayi ürünü talep miktarı",
            "T":"Toplam vergi gelirleri",
            "Td":"Toplam doğrudan vergi geliri",
            "Tz":"Toplam üretim vergisi geliri",
            "Tva":"Toplam katma değer vergisi (KDV) geliri",
            "Tm":"Toplam ithalat vergisi gelirleri",
            "Tz1":"Tarım sektörü üretim vergisi",
            "Tz2":"Tic ve hizmedt sektörü üretim vergisi",
            "Tz3":"Ulaşım sektörü üretim vergisi",
            "Tz4":"İnşaat sektörü üretim vergisi",
            "Tz5":"Sanayi sektörü üretim vergisi",
            "Tzr":"Rafineriler üretim vergisi",
            "Tzb":"BOTAŞ üretim vergisi",
            "Tva1":"Tarım sektörü ürünü KDV",
            "Tva2":"Tic ve hizmet KDV",
            "Tva3":"Ulaşım KDV",
            "Tva4":"İnşaat KDV",
            "Tva5":"Sanayi ürünü KDV",
            "Tvar":"Petrol ürünleri KDV",
            "Tvab":"Doğalgaz KDV",
            "Tm1":"Tarım sektörü ithalat vergisi",
            "Tm2":"Tic ve hizmet sektörü ithalat vergisi",
            "Tm3":"Ulaşım sektörü ithalat vergisi",
            "Tm4":"İnşaat sektörü ithalat vergisi",
            "Tm5":"Sanayi sektörü ithalat vergisi",
            "INV1":"Yatırım birimi tarım ürünü talep miktarı",
            "INV2":"Yatırım birimi tic ve hizmet talep miktarı",
            "INV3":"Yatırım birimi ulaşım talep miktarı",
            "INV4":"Yatırım birimi inşaat talep miktarı",
            "INV5":"Yatırım birimi sanayi talep miktarı",
            "S":"Toplam tasarruflar",
            "Sp":"Özel kesim tasarrufları",
            "Sg":"Kamu tasarufu",
            "px1":"Tarım sektörü kompozit faktör fiyatı",
            "px2":"Tic ve hizmet sektörü kompozit faktör fiyatı",
            "px3":"Ulaşım sektörü kompozit faktör fiyatı",
            "px4":"İnşaat sektörü kompozit faktör fiyatı",
            "px5":"Sanayi sektörü kompozit faktör fiyatı",
            "pxr":"Rafineriler kompozit faktör fiyatı",
            "pxb":"BOTAŞ kompozit faktör fiyatı",
            "pz1":"Tarım sektörü yurtiçi üretim fiyatı",
            "pz2":"Tic ve Hizmet sektörü yurtiçi üretim fiyatı",
            "pz3":"Ulaşım sektörü yurtiçi üretim fiyatı",
            "pz4":"İnşaat sektörü yurtiçi üretim fiyatı",
            "pz5":"Sanayi sektörü yurtiçi üretim fiyatı",
            "pzr":"Petrol ürünleri yurtiçi üretim fiyatı",
            "pzb":"Doğalgaz yurtiçi üretim fiyatı",
            "pe1":"Tarım sektörü TL ihracat fiyatı",
            "pe2":"Tic ve hizmet sektörü TL ihracat fiyatı",
            "pe3":"Ulaşım sektörü TL ihracat fiyatı",
            "pe4":"İnşaat sektörü TL ihracat fiyatı",
            "pe5":"Sanayi sektörü TL ihracat fiyatı",
            "per":"Petrol ürünleri TL ihracat fiyatı",
            "peb":"Doğalgaz TL ihracat fiyatı",
            "pd1":"Tarım sektörü yurtiçi arz fiyatı",
            "pd2":"Tic ve hizmet sektörü yurtiçi arz fiyatı",
            "pd3":"Ulaşım sektörü yurtiçi arz fiyatı",
            "pd4":"İnşaat sektörü yurtiçi arz fiyatı",
            "pd5":"Sanayi sektörü yurtiçi arz fiyatı",
            "pdr":"Petrol ürünü yurtiçi arz fiyatı",
            "pdb": "Doğalgazyurtiçi arz fiyatı",
            "pq1":"Tarım sektörü kompozit mal fiyatı",
            "pq2":"Tic ve hizmet sektörü kompozit mal fiyatı",
            "pq3":"Ulaşım sektörü kompozit mal fiyatı",
            "pq4":"İnşaat sektörü kompozit mal fiyatı",
            "pq5":"Sanayi sektörü kompozit mal fiyatı",
            "pqr":"Rafineriler kompozit mal fiyatı",
            "pm1":"Tarım sektörü TL ithalat fiyatı",
            "pm2":"Tic ve hizmet sektörü TL ithalat fiyatı",
            "pm3":"Ulaşım sektörü TL ithalat fiyatı",
            "pm4":"İnşaat sektörü TL ithalat fiyatı",
            "pm5":"Sanayi sektörü TL ithalat fiyatı",
            "pmr":"Petrol ürünleri TL ithalat fiyatı",
            "pmco":"İthal ham petrol fiyatı",
            "pdco":"Yurtiçi ham petrol fiyatı",
            "pco":"Kompozit ham petrol fiyatı",
            "pxco":"Kompozit ham petrol enerji fiyatı",
            "pmng":"İthal doğalgaz fiyatı",
            "pdng":"Yurtiçi doğalgaz fiyatı",
            "png":"Kompozit doğalgaz fiyatı",
            "pxng":"Kompozit doğalgaz enerjisi fiyatı",
            "Sf":"Yabancı tasarruflar",
            "r":"Sermayenin fiyatı"
        }
        self.parameters = []
        self.parameters_str = {}
        self.exog_variables = []
        self.exog_variables_str = {}

    def UpdateVariables(self):

        self.parameters = [self.td ,self.tva1 ,self.tva2 ,self.tva3 ,self.tva4 ,self.tva5 ,self.tvar ,self.tvab ,self.tz1 ,self.tz2 ,self.tz3 ,self.tz4 ,self.tz5 ,self.tzr ,self.tzb ,
            self.tm1 ,self.tm2 ,self.tm3 ,self.tm4 ,self.tm5 ,self.alphal1 ,self.alphal2 ,self.alphal3 ,self.alphal4 ,self.alphal5 ,
            self.alphalr ,self.alphalb ,self.alphak1 ,self.alphak2 ,self.alphak3 ,self.alphak4 ,self.alphak5 ,self.alphakr ,self.alphakb ,
            self.A1 ,self.A2 ,self.A3 ,self.A4 ,self.A5 ,self.Ar ,self.Ab ,
            # self.alphaxr ,self.alphaco, self.Axco, 
            self.lambdaxco, self.etaxco, self.xr, self.co,
            # self.alphaxb ,self.alphang ,self.Axng ,
            self.lambdaxng, self.etaxng, self.xb, self.ng, 
            self.a11 ,self.a21 ,self.a31 ,self.a41 ,self.a51 ,self.ar1 ,self.ab1 ,self.a12 ,
            self.a22 ,self.a32 ,self.a42 ,self.a52 ,self.ar2 ,self.ab2 ,self.a13 ,self.a23 ,self.a33 ,self.a43 ,self.a53 ,self.ar3 ,self.ab3 ,
            self.a14 ,self.a24 ,self.a34 ,self.a44 ,self.a54 ,self.ar4 ,self.ab4 ,self.a15 ,self.a25 ,self.a35 ,self.a45 ,self.a55 ,self.ar5 ,self.ab5 ,
            self.a1r ,self.a2r ,self.a3r ,self.a4r ,self.a5r ,self.arr ,self.abr ,self.a1b ,self.a2b ,self.a3b ,self.a4b ,
            self.a5b ,self.arb ,self.abb ,self.x1 ,self.x2 ,self.x3 ,self.x4 ,self.x5 ,self.xcor ,self.xngb ,self.rho1 ,
            self.rho2 ,self.rho3 ,self.rho4 ,self.rho5 ,self.rhor ,self.rhob ,self.eta1 ,self.eta2 ,self.eta3 ,self.eta4 ,self.eta5 ,self.etar ,self.etaco ,
            self.etang ,self.e1 ,self.e2 ,self.e3 ,self.e4 ,self.e5 ,self.er ,self.eb ,self.dt1 ,self.dt2 ,self.dt3 ,self.dt4 ,self.dt5 ,
            self.dtr ,self.dtb ,self.theta1 ,self.theta2 ,self.theta3 ,self.theta4 ,self.theta5 ,self.thetar ,self.thetab ,self.m1   ,
            self.m2   ,self.m3   ,self.m4   ,self.m5   ,self.mr   ,self.mcor ,self.mngb , self.da1   ,self.da2   ,self.da3   ,self.da4   ,self.da5   ,
            self.dar   ,self.dcor  ,self.dngb  ,self.lambda1 ,
            self.lambda2 ,self.lambda3 ,self.lambda4 ,self.lambda5 ,self.lambdar ,self.lambdaco ,self.lambdang ,self.c1 ,self.c2 ,self.c3 ,self.c4 ,
            self.c5 ,self.cr ,self.cb ,self.mu1 , self.mu2 ,self.mu3 ,self.mu4 ,self.mu5 ,self.g1 ,self.g2 ,self.g3 ,self.g4 ,self.g5 ,
            self.inv1 ,self.inv2 ,self.inv3 ,self.inv4 ,self.inv5 ,self.sp ,self.sg ,
                ]
        
        self.parameters_str = { "td" : "Doğrudan Vergi Oranı",
                                'tva1' : "Tarım sektörü ürün KDV" ,
                                'tva2': "Tic. ve Hizmet sektörü ürün KDV" ,
                                'tva3': "Ulaşım sektörü ürün KDV" ,
                                'tva4': "İnşaat sektörü ürün KDV" ,
                                "tva5": "Sanayi sektörü ürün KDV" ,
                                "tvar": "Rafineriler ürün KDV" ,
                                "tvab": "BOTAŞ sektörü ürün KDV" ,
                                "tz1" : "Tarım sektörü üretim vergisi" ,
                                "tz2" : "Tic. ve Hizmet sektörü üretim vergisi",
                                "tz3" : "Ulaşım sektörü üretim vergisi",
                                "tz4" : "İnşaat sektörü üretim vergisi",
                                "tz5" : "Sanayi sektörü üretim vergisi",
                                "tzr" : "Rafineriler üretim vergisi",
                                "tzb" : "BOTAŞ üretim vergisi",
                                "tm1" : "Tarım sektörü ithalat vergisi",
                                "tm2" : "Tic. ve Hizmet sektörü ithalat vergisi",
                                "tm3" : "Ulaşım sektörü ithalat vergisi",
                                "tm4" : "İnşaat sektörü ithalat vergisi",
                                "tm5" : "Sanayi sektörü ithalat vergisi",
                                "alphal1" : "Tarım sektörü emek girdi parametresi" ,
                                "alphal2" : "Tic. ve Hizmet sektörü emek girdi parametresi",
                                "alphal3" : "Ulaşım sektörü emek girdi parametresi",
                                "alphal4" : "İnşaat sektörü emek girdi parametresi",
                                "alphal5" : "Sanayi sektörü emek girdi parametresi",
                                "alphalr" : "Rafineriler emek girdi parametresi",
                                "alphalb" : "BOTAŞ sektörü emek girdi parametresi",
                                "alphak1" : "Tarım sektörü sermaye girdi parametresi",
                                "alphak2" : "Tic ve Hizmet sektörü sermaye girdi parametresi",
                                "alphak3" : "Ulaşım sektörü sermaye girdi parametresi",
                                "alphak4" : "İnşaat sektörü sermaye girdi parametresi",
                                "alphak5" : "Sanayi sektörü sermaye girdi parametresi",
                                "alphakr" : "Rafineriler sermaye girdi parametresi",
                                "alphakb" : "BOTAŞ sermaye girdi parametresi",
                                "A1" : "Tarım sektörü kompozit faktör üretimi teknoloji parametresi",
                                "A2" : "Tic. ve Hizmet kompozit faktör üretimi sektörü teknoloji parametresi",
                                "A3" : "Ulaşım sektörü kompozit faktör üretimi teknoloji parametresi",
                                "A4" : "İnşaat sektörü kompozit faktör üretimi teknoloji parametresi",
                                "A5" : "Sanayi sektörü kompozit faktör üretimi teknoloji parametresi",
                                "Ar" : "Rafineriler kompozit faktör üretimi teknoloji parametresi",
                                "Ab" : "BOTAŞ kompozit faktör üretimi teknoloji parametresi",
                                # "alphaxr": "Rafineriler kompozit faktör girdi parametresi" ,
                                # "alphaco": "Rafineriler enerji girdi parametresi" ,
                                # "Axco": "Rafineriler kompozit enerji üretimi teknoloji parametresi"  ,
                                
                                "lambdaxco": "Hampetrol Kompozit Enerji üretimi teknoloji parametresi",
                                "etaxco": "Hampetrol Kompozit Enerji üretimi ikame esneklik parametresi", 
                                "xr": "Hampetrol Kompozit Enerji üretimi kompozit faktör girdi oranı", 
                                "co": "Hampetrol Kompozit Enerji üretimi ham petrol faktör girdi oranı",


                                # "alphaxb": "BOTAŞ kompozit faktör girdi parametresi" ,
                                # "alphang" : "BOTAŞ enerji faktör girdi parametresi" ,
                                # "Axng": "BOTAŞ kompozit enerji üretimi teknoloji parametresi"  , 

                                "lambdaxng" :"Doğalgaz Kompozit Enerji üretimi teknoloji parametresi",
                                "etaxng" : "Doğalgaz Kompozit Enerji üretimi ikame esneklik parametresi",
                                "xb" : "Doğalgaz Kompozit Enerji üretimi kompozit faktör girdi oranı", 
                                "ng" : "Doğalgaz Kompozit Enerji üretimi ham petrol faktör girdi oranı",
                                
                                "a11": "Ara girdi Parametresi (Tarım --> Tarım)" ,
                                "a21": "Ara girdi Parametresi (Tic ve Hizmet --> Tarım)" ,
                                "a31": "Ara girdi Parametresi (Ulaşım --> Tarım)" ,
                                "a41": "Ara girdi Parametresi (İnşaat --> Tarım)" ,
                                "a51": "Ara girdi Parametresi (Sanayi --> Tarım)" ,
                                "ar1": "Ara girdi Parametresi (Rafineriler --> Tarım)" ,
                                "ab1": "Ara girdi Parametresi (BOTAŞ --> Tarım)" ,
                                "a12": "Ara girdi Parametresi (Tarım --> Tic. ve Hizmet)" ,
                                "a22": "Ara girdi Parametresi (Tic. ve Hizmet --> Tic. ve Hizmet)" ,
                                "a32": "Ara girdi Parametresi (Ulaşım --> Tic. ve Hizmet)" ,
                                "a42": "Ara girdi Parametresi (İnşaat --> Tic. ve Hizmet)" ,
                                "a52": "Ara girdi Parametresi (Sanayi --> Tic. ve Hizmet)" ,
                                "ar2": "Ara girdi Parametresi (Rafineriler --> Tic. ve Hizmet)" ,
                                "ab2": "Ara girdi Parametresi (BOTAŞ --> Tic. ve Hizmet)" ,
                                "a13": "Ara girdi Parametresi (Tarım --> Ulaşım)" ,
                                "a23": "Ara girdi Parametresi (Tic. ve Hizmet --> Ulaşım)",
                                "a33": "Ara girdi Parametresi (Ulaşım --> Ulaşım)",
                                "a43": "Ara girdi Parametresi (İnşaat --> Ulaşım)" ,
                                "a53": "Ara girdi Parametresi (Sanayi --> Ulaşım)" ,
                                "ar3": "Ara girdi Parametresi (Rafineriler --> Ulaşım)" ,
                                "ab3": "Ara girdi Parametresi (BOTAŞ --> Ulaşım)" ,
                                "a14": "Ara girdi Parametresi (Tarım --> İnşaat)" ,
                                "a24": "Ara girdi Parametresi (Tic. ve Hizmet --> İnşaat)" ,
                                "a34": "Ara girdi Parametresi (Ulaşım --> İnşaat)" ,
                                "a44": "Ara girdi Parametresi (İnşaat --> İnşaat)" ,
                                "a54": "Ara girdi Parametresi (Sanayi --> İnşaat)" ,
                                "ar4": "Ara girdi Parametresi (Rafineriler --> İnşaat)" ,
                                "ab4": "Ara girdi Parametresi (BOTAŞ --> İnşaat)" ,
                                "a15": "Ara girdi Parametresi (Tarım --> Sanayi)" ,
                                "a25": "Ara girdi Parametresi (Tic. ve Hizmet --> Sanayi)" ,
                                "a35": "Ara girdi Parametresi (Ulaşım --> Sanayi)" ,
                                "a45": "Ara girdi Parametresi (İnşaat --> Sanayi)" ,
                                "a55": "Ara girdi Parametresi (Sanayi --> Sanayi)" ,
                                "ar5": "Ara girdi Parametresi (Rafineriler --> Sanayi)" ,
                                "ab5": "Ara girdi Parametresi (BOTAŞ --> Sanayi)" ,
                                "a1r": "Ara girdi Parametresi (Tarım --> Rafineriler)" ,
                                "a2r": "Ara girdi Parametresi (Tic. ve Hizmet --> Rafineriler)" ,
                                "a3r": "Ara girdi Parametresi (Ulaşım --> Rafineriler)" ,
                                "a4r": "Ara girdi Parametresi (İnşaat --> Rafineriler)" ,
                                "a5r": "Ara girdi Parametresi (Sanayi --> Rafineriler)" ,
                                "arr": "Ara girdi Parametresi (Rafineriler --> Rafineriler)" ,
                                "abr": "Ara girdi Parametresi (BOTAŞ --> Rafineriler)",
                                "a1b": "Ara girdi Parametresi (Tarım --> BOTAŞ)" ,
                                "a2b": "Ara girdi Parametresi (Tic. ve Hizmet --> BOTAŞ)" ,
                                "a3b": "Ara girdi Parametresi (Ulaşım --> BOTAŞ)" ,
                                "a4b": "Ara girdi Parametresi (İnşaat --> BOTAŞ)" ,
                                "a5b": "Ara girdi Parametresi (Sanayi --> BOTAŞ)" ,
                                "arb": "Ara girdi Parametresi (Rafineriler --> BOTAŞ)" ,
                                "abb": "Ara girdi Parametresi (BOTAŞ --> BOTAŞ)" ,
                                "x1": "Kompozit faktör Parametresi (Tarım)" ,
                                "x2": "Kompozit faktör Parametresi (Tic. ve Hizmet)" ,
                                "x3": "Kompozit faktör Parametresi (Ulaşım)" ,
                                "x4": "Kompozit faktör Parametresi (İnşaat)" ,
                                "x5": "Kompozit faktör Parametresi (Sanayi)" ,
                                "xcor": "Kompozit faktör Parametresi (Rafineriler)" ,
                                "xngb": "Kompozit faktör Parametresi (BOTAŞ)" ,
                                "rho1": "Tarım sektörü dönüşüm esneklik parametresi" ,
                                "rho2": "Tic. ve Hizmet sektörü dönüşüm esneklik parametresi" ,
                                "rho3": "Ulaşım sektörü dönüşüm esneklik parametresi" ,
                                "rho4": "İnşaat sektörü dönüşüm esneklik parametresi" ,
                                "rho5": "Sanayi sektörü dönüşüm esneklik parametresi" ,
                                "rhor": "Rafineriler dönüşüm esneklik parametresi" ,
                                "rhob": "BOTAŞ dönüşüm esneklik parametresi" ,
                                "eta1": "Tarım sektörü ikame esneklik parametresi" ,
                                "eta2": "Tic. ve Hizmet sektörü ikame esneklik parametresi" ,
                                "eta3": "Ulaşım sektörü ikame esneklik parametresi" ,
                                "eta4": "İnşaat sektörü ikame esneklik parametresi" ,
                                "eta5": "Sanayi sektörü ikame esneklik parametresi" ,
                                "etar": "Rafineriler ikame esneklik parametresi" ,
                                "etaco": "Yurtiçi - İthal ham petrol ikame esnekliği" ,
                                "etang": "Yurtiçi - İthal doğalgaz ikame esnekliği"  ,
                                "e1": "Tarım sektörü dönüşüm fonksiyonu ihracat girdi oranı" ,
                                "e2": "Tic. ve Hizmet sektörü dönüşüm fonksiyonu ihracat girdi oranı" ,
                                "e3": "Ulaşım sektörü dönüşüm fonksiyonu ihracat girdi oranı" ,
                                "e4": "İnşaat sektörü dönüşüm fonksiyonu ihracat girdi oranı" ,
                                "e5": "Sanayi sektörü dönüşüm fonksiyonu ihracat girdi oranı" ,
                                "er": "Rafineriler dönüşüm fonksiyonu ihracat girdi oranı" ,
                                "eb": "BOTAŞ dönüşüm fonksiyonu ihracat girdi oranı" ,
                                "dt1": "Tarım sektörü dönüşüm fonksiyonu yurtiçi arz girdi oranı" ,
                                "dt2": "Tic. ve Hizmet sektörü dönüşüm fonksiyonu yurtiçi arz girdi oranı"  ,
                                "dt3": "Ulaşım sektörü dönüşüm fonksiyonu yurtiçi arz girdi oranı"  ,
                                "dt4": "İnşaat sektörü dönüşüm fonksiyonu yurtiçi arz girdi oranı"  ,
                                "dt5": "Sanayi sektörü dönüşüm fonksiyonu yurtiçi arz girdi oranı"  ,
                                "dtr": "Rafineriler dönüşüm fonksiyonu yurtiçi arz girdi oranı"  ,
                                "dtb": "BOTAŞ dönüşüm fonksiyonu yurtiçi arz girdi oranı"  ,
                                "theta1": "Tarım sektörü Dönüşüm fonksiyonu teknoloji parametresi"  ,
                                "theta2": "Tic. ve Hizmet sektörü Dönüşüm fonksiyonu teknoloji parametresi"   ,
                                "theta3": "Ulaşım sektörü Dönüşüm fonksiyonu teknoloji parametresi"   ,
                                "theta4": "İnşaat sektörü Dönüşüm fonksiyonu teknoloji parametresi"   ,
                                "theta5": "Sanayi sektörü Dönüşüm fonksiyonu teknoloji parametresi"   ,
                                "thetar": "Rafineriler Dönüşüm fonksiyonu teknoloji parametresi"   ,
                                "thetab": "BOTAŞ Dönüşüm fonksiyonu teknoloji parametresi"   ,
                                "m1": "Tarım sektörü Armington fonksiyonu ithal girdi parametresi"   ,  
                                "m2": "Tic. ve Hizmet sektörü Armington fonksiyonu ithal girdi parametresi"   ,
                                "m3": "Ulaşım sektörü Armington fonksiyonu ithal girdi parametresi"   ,
                                "m4": "İnşaat sektörü Armington fonksiyonu ithal girdi parametresi"  ,
                                "m5": "Sanayi sektörü Armington fonksiyonu ithal girdi parametresi"   ,
                                "mr": "Rafineriler Armington fonksiyonu ithal girdi parametresi"   ,
                                "mcor": "Kompozit ham petrol üretimi ithal girdi parametresi" ,
                                "mngb": "Kompozit doğalgaz üretimi ithal girdi parametresi" ,
                                "da1": "Tarım sektörü Armington fonksiyonu yurtiçi üretim girdi parametresi" , 
                                "da2": "Tic. ve Hizmet sektörü Armington fonksiyonu yurtiçi üretim girdi parametresi"    ,
                                "da3": "Ulaşım sektörü Armington fonksiyonu yurtiçi üretim girdi parametresi"    ,
                                "da4": "İnşaat sektörü Armington fonksiyonu yurtiçi üretim girdi parametresi"    ,
                                "da5": "Sanayi sektörü Armington fonksiyonu yurtiçi üretim girdi parametresi"    ,
                                "dar": "Rafineriler sektörü Armington fonksiyonu yurtiçi üretim girdi parametresi"    ,
                                "dcor": "Kompozit ham petrol üretimi yurtiçi üretim girdi parametresi" ,
                                "dngb": "Kompozit doğalgaz üretimi yurtiçi üretim girdi parametresi" ,
                                "lambda1": "Tarım sektörü Armington fonksiyonu teknoloji parametresi"  ,
                                "lambda2": "Tic. ve Hizmet sektörü Dönüşüm fonksiyonu teknoloji parametresi"  ,
                                "lambda3": "Ulaşım sektörü Dönüşüm fonksiyonu teknoloji parametresi"  ,
                                "lambda4": "İnşaat sektörü Dönüşüm fonksiyonu teknoloji parametresi"  ,
                                "lambda5": "Sanayi sektörü Dönüşüm fonksiyonu teknoloji parametresi"  ,
                                "lambdar": "Rafineriler sektörü Dönüşüm fonksiyonu teknoloji parametresi"  ,
                                "lambdaco": "Kompozit ham petrol teknoloji parametresi"  ,
                                "lambdang": "Kompozit doğalgaz üretimi teknoloji parametresi"   ,
                                "c1": "Hanehalkı tarım sektörü tüketim tercihi parametresi" ,
                                "c2": "Hanehalkı tic. ve hizmet sektörü tüketim tercihi parametresi" ,
                                "c3": "Hanehalkı ulaşım sektörü tüketim tercihi parametresi" , 
                                "c4": "Hanehalkı inşaat sektörü tüketim tercihi parametresi" ,
                                "c5": "Hanehalkı sanayi sektörü tüketim tercihi parametresi" , 
                                "cr": "Hanehalkı rafineriler sektörü tüketim tercihi parametresi" , 
                                "cb": "Hanehalkı BOTAŞ sektörü tüketim tercihi parametresi" , 
                                "mu1": "TPAO tarım sektörü tüketim tercihi parametresi" , 
                                "mu2": "TPAO tic. ve hizmet sektörü tüketim tercihi parametresi"  ,
                                "mu3": "TPAO ulaşım sektörü tüketim tercihi parametresi"  ,
                                "mu4": "TPAO inşaat sektörü tüketim tercihi parametresi"  ,
                                "mu5": "TPAO sanayi sektörü tüketim tercihi parametresi"  ,
                                "g1": "Devlet tarım sektörü tüketim tercihi parametresi"  ,
                                "g2": "Devlet tic. ve hizmet sektörü tüketim tercihi parametresi"  , 
                                "g3": "Devlet ulaşım sektörü tüketim tercihi parametresi"  , 
                                "g4": "Devlet inşaat sektörü tüketim tercihi parametresi"  ,
                                "g5": "Devlet sanayi sektörü tüketim tercihi parametresi"  , 
                                "inv1": "Yatırım kesimi tarım sektörü yatırım tercihi parametresi"  , 
                                "inv2": "Yatırım kesimi tic. ve hizmet sektörü yatırım tercihi parametresi"  , 
                                "inv3": "Yatırım kesimi ulaşım sektörü yatırım tercihi parametresi"  , 
                                "inv4": "Yatırım kesimi inşaat sektörü yatırım tercihi parametresi"  , 
                                "inv5": "Yatırım kesimi sanayi sektörü yatırım tercihi parametresi"  , 
                                "sp": "Özel kesim tasarruf oranı" ,
                                "sg": "Devlet tasarruf oranı" }
      
        self.exog_variables = [ self.DCOBar, self.DNGBar, self.Lbar, self.Kbar, self.Pwe1, self.Pwe2, self.Pwe3, 
                        self.Pwe4, self.Pwe5, self.Pwer, self.Pweb, self.Pwm1, self.Pwm2, self.Pwm3, self.Pwm4, 
                        self.Pwm5, self.Pwmr, self.Pwmco, self.Pwmng, self.epsilon, self.psi1,self.psi2,self.psi3,
                        self.psi4,self.psi5,self.psir,self.psib,self.sigma1,self.sigma2,self.sigma3,self.sigma4,
                        self.sigma5,self.sigmar,self.sigmaco,self.sigmang]
        
        self.exog_variables_str = { "DCOBar": "Yurtiçi Ham Petrol Üretimi Miktarı",
                                    "DNGBar": "Yurtiçi Doğalgaz Üretimi Miktarı",
                                    "Lbar": "Ekonomideki toplam emek miktarı",
                                    "Kbar": "Ekonomideki toplam sermaye miktarı",
                                    "Pwe1": "Tarım sektörü ihracat dünya fiyatı",
                                    "Pwe2": "Tic. ve Hizmet sektörü ihracat dünya fiyatı",
                                    "Pwe3": "Ulaşım sektörü ihracat dünya fiyatı", 
                                    "Pwe4": "İnşaat sektörü ihracat dünya fiyatı", 
                                    "Pwe5": "Sanayi sektörü ihracat dünya fiyatı", 
                                    "Pwer": "Rafineriler sektörü ihracat dünya fiyatı", 
                                    "Pweb": "BOTAŞ sektörü ihracat dünya fiyatı", 
                                    "Pwm1": "Tarım sektörü ithalat dünya fiyatı", 
                                    "Pwm2": "Tic. ve Hizmet sektörü ithalat dünya fiyatı", 
                                    "Pwm3": "Ulaşım sektörü ithalat dünya fiyatı", 
                                    "Pwm4": "İnşaat sektörü ithalat dünya fiyatı", 
                                    "Pwm5": "Sanayi sektörü ithalat dünya fiyatı", 
                                    "Pwmr": "Rafineriler sektörü ithalat dünya fiyatı", 
                                    "Pwmco": "Ham Petrol ithalat dünya fiyatı", 
                                    "Pwmng": "Doğalgaz ithalat dünya fiyatı", 
                                    "epsilon": "Döviz kuru seviyesi",
                                    "psi1": "Tarım sektörü dönüşüm esnekliği",
                                    "psi2": "Tic ve hizmet sektörü dönüşüm esnekliği",
                                    "psi3": "Ulaşım sektörü dönüşüm esnekliği",
                                    "psi4": "İnşaat sektörü dönüşüm esnekliği",
                                    "psi5": "Sanayi sektörü dönüşüm esnekliği",
                                    "psir": "Rafineriler dönüşüm esnekliği",
                                    "psib": "BOTAŞ sektörü dönüşüm esnekliği",
                                    "sigma1": "Tarım sektörü ikame esnekliği",
                                    "sigma2": "Tic ve hizmet sektörü ikame esnekliği",
                                    "sigma3": "Ulaşım sektörü ikame esnekliği",
                                    "sigma4": "İnşaat sektörü ikame esnekliği",
                                    "sigma5": "Sanayi sektörü ikame esnekliği",
                                    "sigmar": "Rafineriler sektörü ikame esnekliği",
                                    "sigmaco": "Yurtiçi-İthal ham petrol ikame esnekliği",
                                    "sigmang": "Yurtiçi-İthal doğalgaz ikame esnekliği"}

    def objValue(self, x):
        C1 = x[111]
        C2 = x[112]
        C3 = x[113]
        C4 = x[114]
        C5 = x[115]
        Cr = x[116]
        Cb = x[117]

        return -C1**self.c1 * C2**self.c2 * C3**self.c3 *C4**self.c4 * C5**self.c5 * Cr**self.cr * Cb**self.cb
        
    def constraints(self, x):
        X1  = x[0]
        L1  = x[1]
        K1  = x[2]
        I11 = x[3]
        I21 = x[4]
        I31 = x[5]
        I41 = x[6]
        I51 = x[7]
        Ir1 = x[8]
        Ib1 = x[9]
        Z1  = x[10]
        E1  = x[11]
        D1  = x[12]
        Q1  = x[13]
        M1  = x[14]

        X2  = x[15]
        L2  = x[16]
        K2  = x[17]
        I12 = x[18]
        I22 = x[19]
        I32 = x[20]
        I42 = x[21]
        I52 = x[22]
        Ir2 = x[23]
        Ib2 = x[24]
        Z2  = x[25]
        E2  = x[26]
        D2  = x[27]
        Q2  = x[28]
        M2  = x[29]

        X3  = x[30]
        L3  = x[31]
        K3  = x[32]
        I13 = x[33]
        I23 = x[34]
        I33 = x[35]
        I43 = x[36]
        I53 = x[37]
        Ir3 = x[38]
        Ib3 = x[39]
        Z3  = x[40]
        E3  = x[41]
        D3  = x[42]
        Q3  = x[43]
        M3  = x[44]

        X4  = x[45]
        L4  = x[46]
        K4  = x[47]
        I14 = x[48]
        I24 = x[49]
        I34 = x[50]
        I44 = x[51]
        I54 = x[52]
        Ir4 = x[53]
        Ib4 = x[54]
        Z4  = x[55]
        E4  = x[56]
        D4  = x[57]
        Q4  = x[58]
        M4  = x[59]

        X5  = x[60]
        L5  = x[61]
        K5  = x[62]
        I15 = x[63]
        I25 = x[64]
        I35 = x[65]
        I45 = x[66]
        I55 = x[67]
        Ir5 = x[68]
        Ib5 = x[69]
        Z5  = x[70]
        E5  = x[71]
        D5  = x[72]
        Q5  = x[73]
        M5  = x[74]

        Xr   = x[75]
        Lr   = x[76]
        Kr   = x[77]
        COr  = x[78]
        MCOr = x[79]
        DCOr = x[80]
        XCOr = x[81]
        I1r  = x[82]
        I2r  = x[83]
        I3r  = x[84]
        I4r  = x[85]
        I5r  = x[86]
        Irr  = x[87]
        Ibr  = x[88]
        Zr   = x[89]
        Er   = x[90]
        Dr   = x[91]
        Qr   = x[92]
        Mr   = x[93]

        Xb   = x[94]
        Lb   = x[95]
        Kb   = x[96]
        NGb  = x[97]
        MNGb = x[98]
        DNGb = x[99]
        XNGb = x[100]
        I1b = x[101]
        I2b = x[102]
        I3b = x[103]
        I4b = x[104]
        I5b = x[105]
        Irb = x[106]
        Ibb = x[107]
        Zb  = x[108]
        Eb  = x[109]
        Db  = x[110]

        C1 = x[111]
        C2 = x[112]
        C3 = x[113]
        C4 = x[114]
        C5 = x[115]
        Cr = x[116]
        Cb = x[117]
        Yd  = x[118]
        Y = x[119]

        TPAO1 = x[120]
        TPAO2 = x[121]
        TPAO3 = x[122]
        TPAO4 = x[123]
        TPAO5 = x[124]
        OIL_INCOME = x[125]

        G1   = x[126]
        G2   = x[127]
        G3   = x[128]
        G4   = x[129]
        G5   = x[130]
        T    = x[131]
        Td   = x[132]
        Tz   = x[133]
        Tva  = x[134]
        Tm   = x[135]
        Tz1  = x[136]
        Tz2  = x[137]
        Tz3  = x[138]
        Tz4  = x[139]
        Tz5  = x[140]
        Tzr  = x[141]
        Tzb  = x[142]
        Tva1 = x[143]
        Tva2 = x[144]
        Tva3 = x[145]
        Tva4 = x[146]
        Tva5 = x[147]
        Tvar = x[148]
        Tvab = x[149]
        Tm1  = x[150]
        Tm2  = x[151]
        Tm3  = x[152]
        Tm4  = x[153]
        Tm5  = x[154]

        INV1 = x[155]
        INV2 = x[156]
        INV3 = x[157]
        INV4 = x[158]
        INV5 = x[159]
        S    = x[160]
        Sp   = x[161]
        Sg   = x[162]

        px1  = x[163]
        px2  = x[164]
        px3  = x[165]
        px4  = x[166]
        px5  = x[167]
        pxr  = x[168]
        pxb  = x[169]
        pz1  = x[170]
        pz2  = x[171]
        pz3  = x[172]
        pz4  = x[173]
        pz5  = x[174]
        pzr  = x[175]
        pzb  = x[176]
        pe1  = x[177]
        pe2  = x[178]
        pe3  = x[179]
        pe4  = x[180]
        pe5  = x[181]
        per  = x[182]
        peb  = x[183]
        pd1  = x[184]
        pd2  = x[185]
        pd3  = x[186]
        pd4  = x[187]
        pd5  = x[188]
        pdr  = x[189]
        pdb  = x[190]
        pq1  = x[191]
        pq2  = x[192]
        pq3  = x[193]
        pq4  = x[194]
        pq5  = x[195]
        pqr  = x[196]
        pm1  = x[197]
        pm2  = x[198]
        pm3  = x[199]
        pm4  = x[200]
        pm5  = x[201]
        pmr  = x[202]
        pmco = x[203]
        pdco = x[204]
        pco  = x[205]
        pxco = x[206]
        pmng = x[207]
        pdng = x[208]
        png  = x[209]
        pxng = x[210]
        Sf = x[211]
        r    = x[212]
        w    = 1
        
        return [
            
        X1  - self.A1*L1**self.alphal1 * K1**self.alphak1,
        L1  - self.alphal1 / w * px1 * X1,
        K1  - self.alphak1 / r * px1 * X1,
        I11 - self.a11*Z1,
        I21 - self.a21*Z1,
        I31 - self.a31*Z1,
        I41 - self.a41*Z1,
        I51 - self.a51*Z1,
        Ir1 - self.ar1*Z1,
        Ib1 - self.ab1*Z1,
        X1  - self.x1*Z1,
        pz1 - (px1*self.x1 + self.a11*pq1 + self.a21*pq2 + self.a31*pq3 + self.a41*pq4 + self.a51*pq5 + self.ar1*pqr + self.ab1*pdb),
        Z1  - self.theta1 * (self.e1*E1**self.rho1 + self.dt1*D1**self.rho1)**(1/self.rho1),
        E1  - (self.theta1 ** self.rho1 * self.e1 * (1+self.tz1 + self.tva1)*pz1 / pe1 )**(1/(1-self.rho1))*Z1,
        D1  - (self.theta1 ** self.rho1 * self.dt1 * (1+self.tz1 + self.tva1)*pz1 / pd1 )**(1/(1-self.rho1))*Z1,
        Q1  - self.lambda1*(self.m1*M1**self.eta1 + self.da1*D1**self.eta1)**(1/self.eta1),
        M1  - (self.lambda1**self.eta1 * self.m1  *pq1 / ((1+self.tm1)*pm1))**(1/(1-self.eta1))*Q1,
        D1  - (self.lambda1**self.eta1 * self.da1 *pq1 / pd1)**(1/(1-self.eta1))*Q1,

        X2  - self.A2*L2**self.alphal2 * K2**self.alphak2,
        L2  - self.alphal2 / w * px2 * X2,
        K2  - self.alphak2 / r * px2 * X2,
        I12 - self.a12*Z2,
        I22 - self.a22*Z2,
        I32 - self.a32*Z2,
        I42 - self.a42*Z2,
        I52 - self.a52*Z2,
        Ir2 - self.ar2*Z2,
        Ib2 - self.ab2*Z2,
        X2  - self.x2*Z2,
        pz2 - (px2*self.x2 + self.a12*pq1 + self.a22*pq2 + self.a32*pq3 + self.a42*pq4 + self.a52*pq5 + self.ar2*pqr + self.ab2*pdb),
        Z2  - self.theta2 * (self.e2*E2**self.rho2 + self.dt2*D2**self.rho2)**(1/self.rho2),
        E2  - (self.theta2 ** self.rho2 * self.e2 * (1+self.tz2 + self.tva2)*pz2 / pe2 )**(1/(1-self.rho2))*Z2,
        D2  - (self.theta2 ** self.rho2 * self.dt2 * (1+self.tz2 + self.tva2)*pz2 / pd2 )**(1/(1-self.rho2))*Z2,
        Q2  - self.lambda2*(self.m2*M2**self.eta2 + self.da2*D2**self.eta2)**(1/self.eta2),
        M2  - (self.lambda2**self.eta2 * self.m2  *pq2 / ((1+self.tm2)*pm2))**(1/(1-self.eta2))*Q2,
        D2  - (self.lambda2**self.eta2 * self.da2 *pq2 / pd2)**(1/(1-self.eta2))*Q2,

        X3  - self.A3*L3**self.alphal3 * K3**self.alphak3,
        L3  - self.alphal3 / w * px3 * X3,
        K3  - self.alphak3 / r * px3 * X3,
        I13 - self.a13*Z3,
        I23 - self.a23*Z3,
        I33 - self.a33*Z3,
        I43 - self.a43*Z3,
        I53 - self.a53*Z3,
        Ir3 - self.ar3*Z3,
        Ib3 - self.ab3*Z3,
        X3  - self.x3*Z3,
        pz3 - (px3*self.x3 + self.a13*pq1 + self.a23*pq2 + self.a33*pq3 + self.a43*pq4 + self.a53*pq5 + self.ar3*pqr + self.ab3*pdb),
        Z3  - self.theta3 * (self.e3*E3**self.rho3 + self.dt3*D3**self.rho3)**(1/self.rho3),
        E3  - (self.theta3 ** self.rho3 * self.e3 * (1+self.tz3 + self.tva3)*pz3 / pe3 )**(1/(1-self.rho3))*Z3,
        D3  - (self.theta3 ** self.rho3 * self.dt3 * (1+self.tz3 + self.tva3)*pz3 / pd3 )**(1/(1-self.rho3))*Z3,
        Q3  - self.lambda3*(self.m3*M3**self.eta3 + self.da3*D3**self.eta3)**(1/self.eta3),
        M3  - (self.lambda3**self.eta3 * self.m3  *pq3 / ((1+self.tm3)*pm3))**(1/(1-self.eta3))*Q3,
        D3  - (self.lambda3**self.eta3 * self.da3 *pq3 / pd3)**(1/(1-self.eta3))*Q3,

        X4  - self.A4*L4**self.alphal4 * K4**self.alphak4,
        L4  - self.alphal4 / w * px4 * X4,
        K4  - self.alphak4 / r * px4 * X4,
        I14 - self.a14*Z4,
        I24 - self.a24*Z4,
        I34 - self.a34*Z4,
        I44 - self.a44*Z4,
        I54 - self.a54*Z4,
        Ir4 - self.ar4*Z4,
        Ib4 - self.ab4*Z4,
        X4  - self.x4*Z4,
        pz4 - (px4*self.x4 + self.a14*pq1 + self.a24*pq2 + self.a34*pq3 + self.a44*pq4 + self.a54*pq5 + self.ar4*pqr + self.ab4*pdb),
        Z4  - self.theta4 * (self.e4*E4**self.rho4 + self.dt4*D4**self.rho4)**(1/self.rho4),
        E4  - (self.theta4 ** self.rho4 * self.e4 * (1+self.tz4 + self.tva4)*pz4 / pe4 )**(1/(1-self.rho4))*Z4,
        D4  - (self.theta4 ** self.rho4 * self.dt4 * (1+self.tz4 + self.tva4)*pz4 / pd4 )**(1/(1-self.rho4))*Z4,
        Q4  - self.lambda4*(self.m4*M4**self.eta4 + self.da4*D4**self.eta4)**(1/self.eta4),
        M4  - (self.lambda4**self.eta4 * self.m4  *pq4 / ((1+self.tm4)*pm4))**(1/(1-self.eta4))*Q4,
        D4  - (self.lambda4**self.eta4 * self.da4 *pq4 / pd4)**(1/(1-self.eta4))*Q4,

        X5  - self.A5*L5**self.alphal5 * K5**self.alphak5,
        L5  - self.alphal5 / w * px5 * X5,
        K5  - self.alphak5 / r * px5 * X5,
        I15 - self.a15*Z5,
        I25 - self.a25*Z5,
        I35 - self.a35*Z5,
        I45 - self.a45*Z5,
        I55 - self.a55*Z5,
        Ir5 - self.ar5*Z5,
        Ib5 - self.ab5*Z5,
        X5  - self.x5*Z5,
        pz5 - (px5*self.x5 + self.a15*pq1 + self.a25*pq2 + self.a35*pq3 + self.a45*pq4 + self.a55*pq5 + self.ar5*pqr + self.ab5*pdb),
        Z5  - self.theta5 * (self.e5*E5**self.rho5 + self.dt5*D5**self.rho5)**(1/self.rho5),
        E5  - (self.theta5 ** self.rho5 * self.e5 * (1+self.tz5 + self.tva5)*pz5 / pe5 )**(1/(1-self.rho5))*Z5,
        D5  - (self.theta5 ** self.rho5 * self.dt5 * (1+self.tz5 + self.tva5)*pz5 / pd5 )**(1/(1-self.rho5))*Z5,
        Q5  - self.lambda5*(self.m5*M5**self.eta5 + self.da5*D5**self.eta5)**(1/self.eta5),
        M5  - (self.lambda5**self.eta5 * self.m5  *pq5 / ((1+self.tm5)*pm5))**(1/(1-self.eta5))*Q5,
        D5  - (self.lambda5**self.eta5 * self.da5 *pq5 / pd5)**(1/(1-self.eta5))*Q5,

        Xr   - self.Ar*Lr**self.alphalr * Kr**self.alphakr,
        Lr   - self.alphalr / w * pxr * Xr,
        Kr   - self.alphakr / r * pxr * Xr,
        COr  - self.lambdaco*(self.mcor*MCOr**self.etaco + self.dcor*DCOr**self.etaco)**(1/self.etaco),
        MCOr - (self.lambdaco**self.etaco*self.mcor*pco/pmco)**(1 / (1-self.etaco)) * COr,
        DCOr - (self.lambdaco**self.etaco*self.dcor*pco/pdco)**(1 / (1-self.etaco)) * COr,

        # XCOr - self.Axco*Xr**self.alphaxr * COr**self.alphaco,
        # Xr   - self.alphaxr / pxr * pxco * XCOr,
        # COr  - self.alphaco / pco * pxco * XCOr,

        XCOr - self.lambdaxco * (self.xr*Xr**self.etaxco + self.co*COr**self.etaxco)**(1/self.etaxco),
        Xr - (self.lambdaxco**self.etaxco * self.xr * pxco  / pxr) ** (1 / (1-self.etaxco)) * XCOr,
        COr - (self.lambdaxco**self.etaxco * self.co * pxco  / pco) ** (1 / (1-self.etaxco)) * XCOr,

        I1r  - self.a1r*Zr,
        I2r  - self.a2r*Zr,
        I3r  - self.a3r*Zr,
        I4r  - self.a4r*Zr,
        I5r  - self.a5r*Zr,
        Irr  - self.arr*Zr,
        Ibr  - self.abr*Zr,
        XCOr - self.xcor*Zr,
        pzr  - (pxco*self.xcor + self.a1r*pq1 + self.a2r*pq2 + self.a3r*pq3 + self.a4r*pq4 + self.a5r*pq5 + self.arr*pqr + self.abr*pdb),
        Zr   - self.thetar * (self.er*Er**self.rhor + self.dtr*Dr**self.rhor)**(1/self.rhor),
        Er   - (self.thetar ** self.rhor * self.er * (1+self.tzr + self.tvar)*pzr / per )**(1/(1-self.rhor))*Zr,
        Dr   - (self.thetar ** self.rhor * self.dtr * (1+self.tzr + self.tvar)*pzr / pdr )**(1/(1-self.rhor))*Zr,
        Qr   - self.lambdar*(self.mr*Mr**self.etar + self.dar*Dr**self.etar)**(1/self.etar),
        Mr   - (self.lambdar**self.etar * self.mr  *pqr / pmr)**(1/(1-self.etar))*Qr,
        Dr   - (self.lambdar**self.etar * self.dar *pqr / pdr)**(1/(1-self.etar))*Qr,

        Xb   - self.Ab*Lb**self.alphalb * Kb**self.alphakb,
        Lb   - self.alphalb / w * pxb * Xb,
        Kb   - self.alphakb / r * pxb * Xb,
        NGb  - self.lambdang * (self.mngb*MNGb**self.etang + self.dngb*DNGb**self.etang)**(1/self.etang),
        MNGb - (self.lambdang**self.etang * self.mngb*png / pmng)**(1/(1-self.etang)) * NGb,
        DNGb - (self.lambdang**self.etang * self.dngb * png / pdng)**(1/(1-self.etang)) * NGb,

        # XNGb - self.Axng * Xb**self.alphaxb * NGb**self.alphang,
        # Xb   - self.alphaxb / pxb * pxng * XNGb,
        # NGb  - self.alphang / png * pxng * XNGb,

        XNGb - self.lambdaxng * (self.xb*Xb**self.etaxng + self.ng*NGb**self.etaxng)**(1/self.etaxng),
        Xb - (self.lambdaxng**self.etaxng * self.xb * pxng  / pxb) ** (1 / (1-self.etaxng)) * XNGb,
        NGb - (self.lambdaxng**self.etaxng * self.ng * pxng  / png) ** (1 / (1-self.etaxng)) * XNGb,


        I1b  - self.a1b*Zb,
        I2b  - self.a2b*Zb,
        I3b  - self.a3b*Zb,
        I4b  - self.a4b*Zb,
        I5b  - self.a5b*Zb,
        Irb  - self.arb*Zb,
        Ibb  - self.abb*Zb,
        XNGb - self.xngb*Zb,
        pzb  - (pxng*self.xngb + self.a1b*pq1 + self.a2b*pq2 + self.a3b*pq3 + self.a4b*pq4 + self.a5b*pq5 + self.arb*pqr + self.abb*pdb),
        Zb   - self.thetab * (self.eb*Eb**self.rhob + self.dtb*Db**self.rhob)**(1/self.rhob),
        Eb   - (self.thetab ** self.rhob * self.eb  * (1+self.tzb + self.tvab) *pzb / peb )**(1/(1-self.rhob))*Zb,
        Db   - (self.thetab ** self.rhob * self.dtb * (1+self.tzb + self.tvab) *pzb / pdb )**(1/(1-self.rhob))*Zb,

        C1 - self.c1 / pq1 * Yd,
        C2 - self.c2 / pq2 * Yd,
        C3 - self.c3 / pq3 * Yd,
        C4 - self.c4 / pq4 * Yd,
        C5 - self.c5 / pq5 * Yd,
        Cr - self.cr / pqr * Yd,
        Cb - self.cb / pdb * Yd,
        Yd - (Y - Sp - Td),
        Y  - (w*self.Lbar + r*self.Kbar),

        TPAO1 - self.mu1 / pq1 * OIL_INCOME,
        TPAO2 - self.mu2 / pq2 * OIL_INCOME,
        TPAO3 - self.mu3 / pq3 * OIL_INCOME,
        TPAO4 - self.mu4 / pq4 * OIL_INCOME,
        TPAO5 - self.mu5 / pq5 * OIL_INCOME,
        OIL_INCOME - (pdco * self.DCOBar + pdng*self.DNGBar),

        G1   - self.g1 / pq1 * (T-Sg),
        G2   - self.g2 / pq2 * (T-Sg),
        G3   - self.g3 / pq3 * (T-Sg),
        G4   - self.g4 / pq4 * (T-Sg),
        G5   - self.g5 / pq5 * (T-Sg),
        T    - (Td + Tz + Tva + Tm),
        Td   - self.td*Y,
        Tz   - (Tz1 + Tz2 + Tz3 + Tz4 + Tz5 + Tzr + Tzb),
        Tz1  - self.tz1 * pz1 * Z1,
        Tz2  - self.tz2 * pz2 * Z2,
        Tz3  - self.tz3 * pz3 * Z3,
        Tz4  - self.tz4 * pz4 * Z4,
        Tz5  - self.tz5 * pz5 * Z5,
        Tzr  - self.tzr * pzr * Zr,
        Tzb  - self.tzb * pzb * Zb,
        Tva  - (Tva1 + Tva2 + Tva3 + Tva4 + Tva5 + Tvar + Tvab), 
        Tva1 - self.tva1 * pz1 * Z1,
        Tva2 - self.tva2 * pz2 * Z2,
        Tva3 - self.tva3 * pz3 * Z3,
        Tva4 - self.tva4 * pz4 * Z4,
        Tva5 - self.tva5 * pz5 * Z5,
        Tvar - self.tvar * pzr * Zr,
        Tvab - self.tvab * pzb * Zb,
        Tm   - (Tm1 + Tm2 + Tm3 + Tm4 + Tm5), 
        Tm1  - self.tm1 * pm1 * M1,
        Tm2  - self.tm2 * pm2 * M2,
        Tm3  - self.tm3 * pm3 * M3,
        Tm4  - self.tm4 * pm4 * M4,
        Tm5  - self.tm5 * pm5 * M5,
        
        INV1 - self.inv1 / pq1 * S,
        INV2 - self.inv2 / pq2 * S,
        INV3 - self.inv3 / pq3 * S,
        INV4 - self.inv4 / pq4 * S,
        INV5 - self.inv5 / pq5 * S,
        S    - (Sp + Sg + Sf*self.epsilon),
        Sp   - self.sp*Y,
        Sg   - self.sg*T,

        pe1 - self.epsilon * self.Pwe1,
        pe2 - self.epsilon * self.Pwe2,
        pe3 - self.epsilon * self.Pwe3,
        pe4 - self.epsilon * self.Pwe4,
        pe5 - self.epsilon * self.Pwe5,
        per - self.epsilon * self.Pwer,
        peb - self.epsilon * self.Pweb,
        pm1 - self.epsilon * self.Pwm1,
        pm2 - self.epsilon * self.Pwm2,
        pm3 - self.epsilon * self.Pwm3,
        pm4 - self.epsilon * self.Pwm4,
        pm5 - self.epsilon * self.Pwm5,
        pmr - self.epsilon * self.Pwmr,
        pmco - self.epsilon * self.Pwmco,
        pmng - self.epsilon * self.Pwmng,
        self.Pwe1*E1 + self.Pwe2*E2 + self.Pwe3*E3 + self.Pwe4*E4 + self.Pwe5*E5 + self.Pwer*Er + self.Pweb*Eb + Sf - \
        (self.Pwm1*M1 + self.Pwm2*M2 + self.Pwm3*M3 + self.Pwm4*M4 + self.Pwm5*M5 + self.Pwmr*Mr +  \
        self.Pwmco * MCOr + self.Pwmng * MNGb),

        Q1 - (C1 + TPAO1 + G1 + INV1 + I11 + I12 + I13 + I14 + I15 + I1r + I1b),
        Q2 - (C2 + TPAO2 + G2 + INV2 + I21 + I22 + I23 + I24 + I25 + I2r + I2b),
        Q3 - (C3 + TPAO3 + G3 + INV3 + I31 + I32 + I33 + I34 + I35 + I3r + I3b),
        Q4 - (C4 + TPAO4 + G4 + INV4 + I41 + I42 + I43 + I44 + I45 + I4r + I4b),
        #         Q5 - (C5 + TPAO5 + G5 + INV5 + I51 + I52 + I53 + I54 + I55 + I5r + I5b),
        Qr - (Cr + Ir1 + Ir2 + Ir3 + Ir4 + Ir5 + Irr + Irb),
        Db - (Cb + Ib1 + Ib2 + Ib3 + Ib4 + Ib5 + Ibr + Ibb),

        self.Lbar - (L1 + L2 + L3 + L4 + L5 + Lr + Lb),
        self.Kbar - (K1 + K2 + K3 + K4 + K5 + Kr + Kb),
        self.DCOBar - DCOr,
        self.DNGBar - DNGb,

        ]
    
    def SolveModel(self):
        
        self.UpdateVariables()

        cons = {"type":"eq", "fun":self.constraints}
        x0 = self.init_values
        bnds = []

        for val in self.init_values_str.keys():
            if val == "Tz1" or val == "Tva1" or val == "Sf":
                bnds.append((None, None))
            else:
                bnds.append((0.000000000001, None))
        
        result = minimize(self.objValue,
                          x0, 
                          constraints = cons,
                         bounds = bnds) 
        
        return result

m = ModelERExog(SAM)
m.DCOBar = 54
m.DNGBar = 8

print(m.SolveModel().message)
for i in m.SolveModel().x:
    print(i)
