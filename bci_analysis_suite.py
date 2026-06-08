"""BCI-IM Analysis Suite.

A Tkinter desktop application for offline analysis and classification of the
8-channel motor-imagery / executed-movement EEG recorded by the BCI-IM Data
Collection Suite. It loads the commented CSV files produced during acquisition,
segments cued trials and rest periods, and runs a configurable feature
extraction + classification pipeline with leakage-safe, subject/trial-grouped
cross-validation.

Feature families
----------------
* Common Spatial Patterns (CSP) and Filter-Bank CSP (FBCSP).
* Band power, Hjorth parameters, spectral connectivity (coherence).
* Nonlinear / complexity features (Lempel-Ziv complexity, Katz fractal
  dimension, sample entropy, DFA, Hjorth mobility/complexity).
* Optional Riemannian tangent-space features (if ``pyriemann`` is installed).

Classifiers: LDA, SVM (RBF), Random Forest, and optionally XGBoost.

The interface is a split pane: acquisition/processing controls on the left and
a tabbed visualization area on the right (raw-signal preview and a "Functional
Analysis" tab offering ERD/ERS, connectivity, recurrence plots, reconstructed
attractors, CSP spatial maps, nonlinear-metric summaries and phase-amplitude
coupling). Batch runs export PSD, ERD/ERS, topographic and summary figures plus
a results CSV.

Method references
-----------------
* ERD/ERS: Pfurtscheller & Lopes da Silva (1999).
* RQA / STR connectivity: Rodrigues et al. (2019), Webber & Zbilut (2005),
  Marwan et al. (2007).
* Nonlinear dynamics: Stam (2005), Abasolo et al. (2006).
* Phase-amplitude coupling: Tort et al. (2010), Canolty et al. (2006).

Dependencies
------------
    pip install numpy pandas scipy scikit-learn matplotlib seaborn
    # optional: xgboost, pyriemann

Usage
-----
    python bci_analysis_suite_v2.py
"""
# ── Splash screen (instant — only uses tkinter) ─────────────
import tkinter as _tk_splash
import sys, os

def _show_splash():
    sp = _tk_splash.Tk(); sp.title("BCI-IM Analysis Suite")
    sp.overrideredirect(True); w, h = 440, 150
    sx = sp.winfo_screenwidth()//2 - w//2; sy = sp.winfo_screenheight()//2 - h//2
    sp.geometry(f"{w}x{h}+{sx}+{sy}"); sp.configure(bg="#0F1318"); sp.attributes("-topmost", True)
    _tk_splash.Label(sp,text="📊  BCI-IM Analysis Suite v2.0",font=("Segoe UI",14,"bold"),fg="#4FC3F7",bg="#0F1318").pack(pady=(24,4))
    _lbl=_tk_splash.Label(sp,text="Loading...",font=("Segoe UI",10),fg="#6B7A94",bg="#0F1318"); _lbl.pack(pady=(0,4))
    _bar=_tk_splash.Canvas(sp,width=320,height=8,bg="#1A1F28",highlightthickness=0); _bar.pack(pady=(4,0))
    sp.update(); return sp, _lbl, _bar

def _sp_prog(sp,lbl,bar,text,pct):
    try: lbl.config(text=text); bar.delete("all"); bar.create_rectangle(0,0,int(3.2*pct),8,fill="#4FC3F7",outline=""); sp.update()
    except: pass

_sp, _sp_lbl, _sp_bar = _show_splash()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, time, re, glob, copy, json, warnings, itertools, queue, gc
from datetime import datetime

_sp_prog(_sp,_sp_lbl,_sp_bar,"NumPy / Pandas...",10)
import numpy as np; import pandas as pd

_sp_prog(_sp,_sp_lbl,_sp_bar,"Matplotlib / Seaborn...",25)
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

_sp_prog(_sp,_sp_lbl,_sp_bar,"SciPy...",45)
from scipy.linalg import eigh
from scipy.signal import welch, butter, sosfiltfilt, coherence as sp_coherence, hilbert
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.interpolate import CloughTocher2DInterpolator

_sp_prog(_sp,_sp_lbl,_sp_bar,"Scikit-learn...",65)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.covariance import OAS

_sp_prog(_sp,_sp_lbl,_sp_bar,"Extras...",85)
_np_integrate = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except ImportError: HAS_XGB = False
try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace; HAS_PYRIEMANN = True
except ImportError: HAS_PYRIEMANN = False
warnings.filterwarnings("ignore")
_sp_prog(_sp,_sp_lbl,_sp_bar,"Iniciando interface...",100)


#  CONSTANTS

VERSION = "2.0"; SFREQ = 250.0; RANDOM_STATE = 42
CH_ORDER = ["C3","Cz","C4","P3","P4","F3","F4","Pz"]; N_CH = len(CH_ORDER)
CH_COLORS = ["#4FC3F7","#81C784","#FF8A65","#BA68C8","#FFD54F","#4DD0E1","#F48FB1","#A1887F"]
MARKERS = {"session_start":100,"session_end":200,"block_start":90,"block_end":91,
           "fixation":1,"beep":2,"rest_start":50,"rest_end":51,"pause_start":99,"pause_end":98,
           "mi_left":10,"mi_right":11,"mov_hand":20,"mov_elbow":21,"mov_shoulder":22}
MOV_EN = {1:"Hand Grip",2:"Elbow Flexion",3:"Shoulder Extension"}
GRAZ_N = {10:"MI Left",11:"MI Right"}
_TH = np.array([-90,0,90,-126,126,-54,54,180]); _RD = np.array([0.5,0,0.5,0.5,0.5,0.5,0.5,0.5])
CH_XY = {c:(r*np.sin(np.deg2rad(t)),r*np.cos(np.deg2rad(t))) for c,t,r in zip(CH_ORDER,_TH,_RD)}
CH_X = np.array([CH_XY[c][0] for c in CH_ORDER]); CH_Y = np.array([CH_XY[c][1] for c in CH_ORDER])
HR = 0.55; CC = ['#2166AC','#B2182B','#4DAF4A','#FF7F00','#984EA3']
TH = {"bg":"#0F1318","sf":"#1A1F28","sf2":"#242A36","bd":"#2E3648","tx":"#D4DAE6",
      "tm":"#6B7A94","ac":"#4FC3F7","ad":"#1A3A52","gn":"#66BB6A","gd":"#1A3D1C",
      "rd":"#EF5350","rdd":"#3D1A1A","yl":"#FFD54F","or":"#FFA726","pu":"#B39DDB"}

def bandpass_filter(data, lo, hi, fs=SFREQ, order=4):
    return sosfiltfilt(butter(order,[lo,hi],btype='band',fs=fs,output='sos'), data, axis=-1)


#  FEATURE EXTRACTORS

class CSP_Reg(BaseEstimator, TransformerMixin):
    def __init__(self, nf=2): self.nf=nf; self.W=None
    def fit(self, X, y):
        cl=np.unique(y)
        def sc(t): return np.mean([OAS().fit(x.T).covariance_ for x in t],0)
        C0=sc(X[y==cl[0]]); C1=sc(X[y==cl[1]]); Cc=C0+C1+np.eye(C0.shape[0])*1e-6
        v,w=eigh(C0,Cc); idx=np.concatenate([np.arange(self.nf),np.arange(len(v)-self.nf,len(v))])
        self.W=w[:,idx]; return self
    def transform(self, X):
        return np.array([np.log(np.maximum(np.var(self.W.T@x,axis=1),1e-12)) for x in X])

class FBCSP(BaseEstimator, TransformerMixin):
    def __init__(self, bands=None, nf=2, ns=6):
        self.bands=bands or [(4,8),(8,12),(12,16),(16,20),(20,24),(24,30)]
        self.nf=nf; self.ns=ns; self.csps=[]; self.sel=None
    def fit(self, X, y):
        self.csps=[]; af=[]
        for lo,hi in self.bands:
            Xf=np.array([bandpass_filter(x,lo,hi) for x in X]); c=CSP_Reg(self.nf)
            try: c.fit(Xf,y); f=c.transform(Xf)
            except: c.W=None; f=np.zeros((len(X),2*self.nf))
            self.csps.append(c); af.append(f)
        af=np.hstack(af)
        sc=[np.log(np.var(af[:,j])/(sum(np.var(af[y==c,j])*np.sum(y==c) for c in np.unique(y))/len(y)+1e-12)+1e-12) for j in range(af.shape[1])]
        self.sel=np.argsort(sc)[::-1][:min(self.ns,len(sc))]; return self
    def transform(self, X):
        af=[]
        for (lo,hi),c in zip(self.bands,self.csps):
            Xf=np.array([bandpass_filter(x,lo,hi) for x in X])
            af.append(c.transform(Xf) if c.W is not None else np.zeros((len(X),2*self.nf)))
        return np.hstack(af)[:,self.sel]

class BP_Feat(BaseEstimator, TransformerMixin):
    def __init__(self): self.bands=[('th',4,8),('al',8,12),('lb',12,20),('hb',20,30)]
    def fit(self,X,y=None): return self
    def transform(self,X):
        feats=[]
        for t in X:
            tf=[]
            for ch in range(t.shape[0]):
                fr,ps=welch(t[ch],fs=SFREQ,nperseg=min(256,t.shape[1]))
                for _,lo,hi in self.bands:
                    m=(fr>=lo)&(fr<=hi); tf.append(np.log(_np_integrate(ps[m],fr[m])+1e-12))
            feats.append(tf)
        return np.array(feats)

class Hjorth_Feat(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X):
        feats=[]
        for t in X:
            tf=[]
            for ch in range(t.shape[0]):
                s=t[ch]; d1=np.diff(s); d2=np.diff(d1); a=np.var(s); m=np.sqrt(np.var(d1)/(a+1e-12))
                tf.extend([a,m,np.sqrt(np.var(d2)/(np.var(d1)+1e-12))/(m+1e-12)])
            feats.append(tf)
        return np.array(feats)

class NL_Feat(BaseEstimator, TransformerMixin):
    def _lz(self,s):
        b=''.join(map(str,(s>np.median(s)).astype(int))); n=len(b)
        if n==0: return 0
        i,c,l=0,1,1
        while i+l<=n:
            if b[i:i+l] in b[:i+l-1]: l+=1
            else: c+=1; i+=l; l=1
        return c/(n/max(np.log2(n),1e-12))
    def _kfd(self,s):
        n=len(s)-1
        if n<1: return 0
        L=np.sum(np.abs(np.diff(s))); d=np.max(np.abs(s-s[0]))
        return np.log10(n)/(np.log10(d/L)+np.log10(n)) if d>1e-12 and L>1e-12 else 0
    def _se(self,s,m=2,rf=0.2):
        r=rf*np.std(s); N=len(s)
        if N<m+2 or r<1e-12: return 0.0
        sg=s[::max(1,N//300)]; N2=len(sg)
        if N2<m+2: return 0.0
        def _c(tl):
            t=np.array([sg[i:i+tl] for i in range(N2-tl)]); cnt=0
            for i in range(len(t)): cnt+=np.sum(np.max(np.abs(t[i]-t),axis=1)<r)-1
            return cnt
        A=_c(m+1); B=_c(m)
        return -np.log((A+1e-12)/(B+1e-12)) if B else 0.0
    def fit(self,X,y=None): return self
    def transform(self,X):
        feats=[]
        for t in X:
            tf=[]
            for ch in range(t.shape[0]):
                s=t[ch]; d1=np.diff(s); d2=np.diff(d1); a=np.var(s); mb=np.sqrt(np.var(d1)/(a+1e-12))
                tf.extend([self._lz(s),self._kfd(s),np.log(a+1e-12),a,mb,
                           np.sqrt(np.var(d2)/(np.var(d1)+1e-12))/(mb+1e-12),self._se(s)])
            feats.append(tf)
        return np.array(feats)

class Conn_Feat(BaseEstimator, TransformerMixin):
    def __init__(self): self.pairs=list(itertools.combinations(range(N_CH),2)); self.bands=[('mu',8,13),('beta',13,30)]
    def fit(self,X,y=None): return self
    def transform(self,X):
        feats=[]
        for t in X:
            tf=[]
            for i,j in self.pairs:
                f,Cxy=sp_coherence(t[i],t[j],fs=SFREQ,nperseg=min(128,t.shape[1]))
                for _,lo,hi in self.bands: tf.append(np.mean(Cxy[(f>=lo)&(f<=hi)]))
            feats.append(tf)
        return np.array(feats)


#  RQA ENGINE — Webber & Zbilut (2005), Marwan et al. (2007)

def _embed(signal, dim, tau):
    N = len(signal)
    if N < (dim-1)*tau+1: return signal.reshape(-1,1)
    return np.array([signal[i*tau:i*tau + N - (dim-1)*tau] for i in range(dim)]).T

def _recurrence_matrix(trajectory, eps_pct=5):
    D = squareform(pdist(trajectory, 'euclidean'))
    eps = np.percentile(D[D > 0], eps_pct) if np.any(D > 0) else 1e-12
    return (D <= eps).astype(np.int8), eps

def compute_rqa(signal, dim=10, tau=2, eps_pct=5, l_min=2):
    """11 RQA metrics: RR, DET, L, Lmax, ENTR, LAM, TT, Vmax, LWVL, W, RTE"""
    traj = _embed(signal, dim, tau)
    zero = {k: 0.0 for k in ['RR','DET','L','Lmax','ENTR','LAM','TT','Vmax','LWVL','W','RTE']}
    if traj.shape[0] < 5: return zero
    R, eps = _recurrence_matrix(traj, eps_pct)
    N = R.shape[0]; total = N * N; rsum = np.sum(R)
    if rsum == 0: return zero
    RR = rsum / total
    def _lines(M, axis=0):
        lines = []
        it = range(N)
        for idx in (range(-N+1,N) if axis==2 else range(N)):
            seq = np.diag(R, idx) if axis==2 else (R[:,idx] if axis==0 else R[idx,:])
            ll = 0
            for v in seq:
                if v: ll += 1
                elif ll >= l_min: lines.append(ll); ll = 0
            if ll >= l_min: lines.append(ll)
        return lines
    def _entropy(lines):
        if not lines: return 0.0
        c = np.bincount(lines); p = c[c>0]/sum(c[c>0])
        return -np.sum(p*np.log(p+1e-12))
    diags = _lines(R, 2)
    DET = sum(diags)/rsum if diags else 0; L = np.mean(diags) if diags else 0
    Lmax = max(diags) if diags else 0; ENTR = _entropy(diags)
    verts = _lines(R, 0)
    LAM = sum(verts)/rsum if verts else 0; TT = np.mean(verts) if verts else 0
    Vmax = max(verts) if verts else 0
    wv = []
    for j in range(N):
        ll = 0
        for i in range(N):
            if not R[i,j]: ll += 1
            elif ll >= l_min: wv.append(ll); ll = 0
        if ll >= l_min: wv.append(ll)
    LWVL = max(wv) if wv else 0; W = np.mean(wv) if wv else 0; RTE = _entropy(wv)
    return {'RR':RR,'DET':DET,'L':L,'Lmax':Lmax,'ENTR':ENTR,'LAM':LAM,'TT':TT,'Vmax':Vmax,'LWVL':LWVL,'W':W,'RTE':RTE}

def cross_recurrence_rqa(sig1, sig2, dim=10, tau=2, eps_pct=5):
    """Cross-recurrence RR and DET — Rodrigues et al. (2019) STR method."""
    t1 = _embed(sig1, dim, tau); t2 = _embed(sig2, dim, tau)
    n = min(len(t1), len(t2)); t1 = t1[:n]; t2 = t2[:n]
    if n < 5: return 0, 0
    D = cdist(t1, t2, 'euclidean')
    eps = np.percentile(D[D>0], eps_pct) if np.any(D>0) else 1e-12
    CR = (D <= eps).astype(np.int8)
    crr = np.sum(CR)/(n*n)
    diags = []
    for k in range(-n+1, n):
        d = np.diag(CR, k); ll = 0
        for v in d:
            if v: ll += 1
            elif ll >= 2: diags.append(ll); ll = 0
        if ll >= 2: diags.append(ll)
    cdet = sum(diags)/max(np.sum(CR),1) if diags else 0
    return crr, cdet


#  NONLINEAR DYNAMICS — Stam (2005), Abasolo et al. (2006)

NL_METRIC_NAMES = ['LZC','KFD','SampEn','Variance','Mobility','Complexity','DFA_alpha']
NL_INTERPRETATIONS = {
    'LZC': 'Lempel-Ziv Complexity — signal randomness. ↑=complex dynamics. Reduced in AD (Abasolo 2006).',
    'KFD': 'Katz Fractal Dimension — waveform complexity. Sensitive to cognitive load (Stam 2005).',
    'SampEn': 'Sample Entropy — regularity. ↑=unpredictable. ↓ in neurodegeneration (Richman & Moorman 2000).',
    'Variance': 'Signal variance — oscillation amplitude. Reflects band power.',
    'Mobility': 'Hjorth Mobility — mean frequency proxy (Hjorth 1970).',
    'Complexity': 'Hjorth Complexity — bandwidth proxy (Hjorth 1970).',
    'DFA_alpha': 'DFA exponent — long-range correlations. α≈0.5=noise, α≈1=1/f, α≈1.5=Brownian (Peng 1994).'
}

def compute_nl_metrics(signal):
    nl = NL_Feat()
    lzc = nl._lz(signal); kfd = nl._kfd(signal); se = nl._se(signal)
    var_ = np.var(signal); d1 = np.diff(signal); d2 = np.diff(d1)
    mob = np.sqrt(np.var(d1)/(var_+1e-12))
    comp = np.sqrt(np.var(d2)/(np.var(d1)+1e-12))/(mob+1e-12)
    # DFA (Peng et al. 1994)
    try:
        n = len(signal); y = np.cumsum(signal - np.mean(signal))
        scales = np.unique(np.logspace(0.5, np.log10(max(n//4,5)), 15).astype(int))
        scales = scales[scales >= 4]; flucts = []
        for s in scales:
            segs = n // s
            if segs < 1: continue
            rms_list = []
            for seg in range(segs):
                ys = y[seg*s:(seg+1)*s]; x = np.arange(s)
                coeffs = np.polyfit(x, ys, 1); trend = np.polyval(coeffs, x)
                rms_list.append(np.sqrt(np.mean((ys-trend)**2)))
            flucts.append(np.mean(rms_list))
        dfa = np.polyfit(np.log(scales[:len(flucts)]), np.log(np.array(flucts)+1e-12), 1)[0] if len(flucts)>=3 else 0
    except: dfa = 0
    return [lzc, kfd, se, var_, mob, comp, dfa]


#  DATA I/O

def load_csv(fp):
    meta={}
    with open(fp,'r',encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                l=line.strip('# \n')
                if ':' in l: k,v=l.split(':',1); meta[k.strip()]=v.strip()
            else: break
    return pd.read_csv(fp,comment='#'), meta

def _cmap(df):
    m={}
    for col in df.columns:
        for std in CH_ORDER:
            if col.upper()==std.upper(): m[std]=col; break
    return m

def extract_graz(df):
    mk=df['Marker'].values if 'Marker' in df.columns else np.zeros(len(df))
    cm=_cmap(df); trials=[]; labels=[]; mi=[MARKERS['mi_left'],MARKERS['mi_right']]
    i=0
    while i<len(mk):
        if mk[i] in mi:
            lb=mk[i]; s=i; e=s+1
            while e<len(mk) and mk[e]==0: e+=1
            if e-s>int(SFREQ):
                td=np.zeros((N_CH,e-s))
                for ci,cn in enumerate(CH_ORDER):
                    if cn in cm: td[ci]=df[cm[cn]].values[s:e]
                trials.append(td); labels.append(lb)
            i=e
        else: i+=1
    return trials, np.array(labels) if labels else np.array([])

def extract_mov(df):
    mk=df['Marker'].values if 'Marker' in df.columns else np.zeros(len(df))
    cm=_cmap(df); trials=[]; labels=[]; mm=[MARKERS['mov_hand'],MARKERS['mov_elbow'],MARKERS['mov_shoulder']]
    i=0
    while i<len(mk):
        if mk[i] in mm:
            lb=mk[i]; s=i; e=s+1
            while e<len(mk) and mk[e]==0: e+=1
            if e-s>int(SFREQ):
                td=np.zeros((N_CH,e-s))
                for ci,cn in enumerate(CH_ORDER):
                    if cn in cm: td[ci]=df[cm[cn]].values[s:e]
                trials.append(td); labels.append(lb)
            i=e
        else: i+=1
    return trials, np.array(labels) if labels else np.array([])

def extract_rest(df):
    mk=df['Marker'].values if 'Marker' in df.columns else np.zeros(len(df))
    cm=_cmap(df); segs=[]; i=0
    while i<len(mk):
        if mk[i]==MARKERS['rest_start']:
            s=i; e=s+1
            while e<len(mk) and mk[e]!=MARKERS['rest_end']: e+=1
            if e-s>int(SFREQ*0.5):
                seg=np.zeros((N_CH,e-s))
                for ci,cn in enumerate(CH_ORDER):
                    if cn in cm: seg[ci]=df[cm[cn]].values[s:e]
                segs.append(seg)
            i=e+1
        else: i+=1
    return segs

def seg_trials(trials,labels,ca,cb,ws=3.0,ov=0.5,max_windows=5000):
    wn=int(SFREQ*ws); hop=max(1,int(wn*(1-ov)))
    idx=np.where((labels==ca)|(labels==cb))[0]
    if len(idx)<2: return None,None,None
    Xw,yw,gw=[],[],[]
    for i,j in enumerate(idx):
        t=trials[j]; lb=0 if labels[j]==ca else 1
        for s in range(0,t.shape[1]-wn+1,hop):
            Xw.append(t[:,s:s+wn]); yw.append(lb); gw.append(i)
            if len(Xw)>=max_windows: break
        if len(Xw)>=max_windows: break
    if len(Xw)<4: return None,None,None
    return np.stack(Xw),np.array(yw),np.array(gw)

def seg_trials_multi(trials, labels, classes, ws=3.0, ov=0.5, max_windows=5000):
    """Multi-class segmentation for 3+ movement types."""
    wn=int(SFREQ*ws); hop=max(1,int(wn*(1-ov)))
    cl_sorted=sorted(classes); cl_map={c:i for i,c in enumerate(cl_sorted)}
    idx=np.where(np.isin(labels,cl_sorted))[0]
    if len(idx)<3: return None,None,None
    Xw,yw,gw=[],[],[]
    for i,j in enumerate(idx):
        t=trials[j]; lb=cl_map[labels[j]]
        for s in range(0,t.shape[1]-wn+1,hop):
            Xw.append(t[:,s:s+wn]); yw.append(lb); gw.append(i)
            if len(Xw)>=max_windows: break
        if len(Xw)>=max_windows: break
    if len(Xw)<6: return None,None,None
    return np.stack(Xw),np.array(yw),np.array(gw)


#  PREPROCESSING

def pp_bp(t,lo,hi,fs=SFREQ,o=4): return [bandpass_filter(x,lo,hi,fs,o) for x in t]
def pp_notch(t,freq=60.0,fs=SFREQ,Q=30):
    from scipy.signal import iirnotch, filtfilt
    b,a=iirnotch(freq,Q,fs); return [filtfilt(b,a,x,axis=-1) for x in t]
def pp_car(t): return [x-np.mean(x,axis=0) for x in t]
def pp_zs(t): return [(x-np.mean(x,1,keepdims=True))/(np.std(x,1,keepdims=True)+1e-12) for x in t]

def rej_art(X,y,g,pct=85):
    nw,nc,ns=X.shape; rms=np.sqrt(np.mean(X**2,axis=2)); ptp=np.ptp(X,axis=2)
    hbr=np.zeros((nw,nc))
    for i in range(nw):
        for ch in range(nc):
            f,psd_=welch(X[i,ch],fs=SFREQ,nperseg=min(128,ns))
            mu_p=_np_integrate(psd_[(f>=8)&(f<=13)],f[(f>=8)&(f<=13)]) if np.any((f>=8)&(f<=13)) else 1e-12
            hb_p=_np_integrate(psd_[(f>=20)&(f<=30)],f[(f>=20)&(f<=30)]) if np.any((f>=20)&(f<=30)) else 0
            hbr[i,ch]=hb_p/(mu_p+1e-12)
    zs={}
    for nm,v in {'rms':rms,'ptp':ptp,'hbr':hbr}.items():
        mpw=np.max(v,1); mu=np.mean(mpw); sg=np.std(mpw)+1e-12; zs[nm]=(mpw-mu)/sg
    comp_=np.max(np.column_stack(list(zs.values())),1); thr=np.percentile(comp_,pct)
    rej=comp_>thr; keep=~rej
    cs={}
    for c in np.unique(y):
        cm_=y==c; n=int(np.sum(cm_)); nr=int(np.sum(cm_&rej))
        cs[c]={'total':n,'rejected':nr,'pct':nr/max(n,1)*100}
    return X[keep],y[keep],g[keep],{'n_total':len(X),'n_rej':int(np.sum(rej)),'n_kept':int(np.sum(keep)),
           'pct':np.sum(rej)/len(X)*100,'cs':cs}


#  CLASSIFICATION

def get_clfs():
    c={'LDA':LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'),
       'SVM':SVC(kernel='rbf',C=10,gamma='scale',random_state=RANDOM_STATE),
       'RF':RandomForestClassifier(n_estimators=200,max_depth=15,min_samples_split=5,random_state=RANDOM_STATE)}
    if HAS_XGB: c['XGB']=XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,subsample=0.8,
                                         colsample_bytree=0.8,random_state=RANDOM_STATE,eval_metric='logloss',verbosity=0)
    return c

def get_fmethods():
    m={'CSP':None,'FB-CSP':None,'Band Power':BP_Feat(),'Hjorth':Hjorth_Feat(),
       'Nonlinear':NL_Feat(),'Connectivity':Conn_Feat()}
    if HAS_PYRIEMANN: m['Riemannian']=None
    return m

def extr_feat(Xtr,ytr,Xte,mn):
    if mn=='CSP': m=CSP_Reg(2); m.fit(Xtr,ytr); return m.transform(Xtr),m.transform(Xte),m
    elif mn=='FB-CSP': m=FBCSP(nf=2,ns=8); m.fit(Xtr,ytr); return m.transform(Xtr),m.transform(Xte),m
    elif mn=='Riemannian':
        cov=Covariances(estimator='oas'); ts=TangentSpace()
        ctr=cov.fit_transform(Xtr); cte=cov.transform(Xte); xtr=ts.fit_transform(ctr); xte=ts.transform(cte)
        if xtr.shape[1]>30: p=PCA(n_components=min(20,xtr.shape[0]-1)); xtr=p.fit_transform(xtr); xte=p.transform(xte)
        return xtr,xte,None
    else:
        mm={'Band Power':BP_Feat(),'Hjorth':Hjorth_Feat(),'Nonlinear':NL_Feat(),'Connectivity':Conn_Feat()}
        e=mm[mn]; e.fit(Xtr,ytr); return e.transform(Xtr),e.transform(Xte),None

def run_clf(X,y,g,mnames,cnames,nsp=5,log=None):
    n_classes = len(np.unique(y))
    cd={k:v for k,v in get_clfs().items() if k in cnames}; res=[]
    ns=min(nsp,len(np.unique(g)))
    if ns<2:
        if log: log("[WARNING] Not enough groups.")
        return res
    # Filter methods: CSP/FB-CSP only work with binary classification
    valid_mnames = []
    for mn in mnames:
        if n_classes > 2 and mn in ('CSP', 'FB-CSP', 'Riemannian'):
            if log: log(f"  [!] {mn} skipped (requires 2 classes, found {n_classes})")
            continue
        valid_mnames.append(mn)
    if not valid_mnames:
        if log: log("[WARNING] No method compatible with multi-class.")
        return res
    gkf=GroupKFold(n_splits=ns)
    for mn in valid_mnames:
        fr={cn:{'acc':[],'kp':[],'yt':[],'yp':[],'inv':0,'tot':0} for cn in cd}
        for fi,(tri,tei) in enumerate(gkf.split(X,y,g)):
            Xtr,ytr=X[tri],y[tri]; Xte,yte=X[tei],y[tei]
            if len(np.unique(ytr))<2: continue
            try: xtr,xte,ex=extr_feat(Xtr,ytr,Xte,mn)
            except Exception as e:
                if log: log(f"  [!] {mn} fold {fi}: {e}")
                continue
            if xtr.shape[0]==0: continue
            sc=StandardScaler(); xtr=np.nan_to_num(sc.fit_transform(xtr)); xte=np.nan_to_num(sc.transform(xte))
            for cn,ct in cd.items():
                clf=copy.deepcopy(ct)
                try:
                    clf.fit(xtr,ytr)
                    yp=clf.predict(xte)
                    # Inversion check only for binary
                    inv = False
                    if n_classes == 2:
                        if accuracy_score(ytr, clf.predict(xtr)) < 0.5:
                            yp = 1 - yp; inv = True
                    fr[cn]['tot']+=1
                    if inv: fr[cn]['inv']+=1
                    a=accuracy_score(yte,yp)
                    try: k=cohen_kappa_score(yte,yp)
                    except: k=0.0
                    if np.isnan(k): k=0.0
                    fr[cn]['acc'].append(a); fr[cn]['kp'].append(k)
                    fr[cn]['yt'].extend(yte.tolist()); fr[cn]['yp'].extend(yp.tolist())
                except Exception as clf_e:
                    if log: log(f"    [!] {mn}+{cn} fold {fi}: {clf_e}")
        for cn in cd:
            a=fr[cn]['acc']
            if not a: continue
            res.append({'method':mn,'clf':cn,'acc_mean':np.mean(a),'acc_std':np.std(a),
                'kappa_mean':np.nanmean(fr[cn]['kp']),'fold_accs':a,
                'y_true_all':fr[cn]['yt'],'y_pred_all':fr[cn]['yp'],
                'folds_inverted':fr[cn]['inv'],'folds_total':fr[cn]['tot']})
    return res


#  PLOT FUNCTIONS (Agg backend — file output)

def _pstyle():
    import matplotlib as _m; _m.use("Agg",force=True); plt.switch_backend("Agg")
    plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
        'font.family':'serif','font.size':10,'axes.titlesize':12,'axes.labelsize':11,
        'axes.titleweight':'bold','axes.grid':True,'grid.alpha':0.3,'legend.fontsize':9})

def _dhead(ax):
    th=np.linspace(0,2*np.pi,200); ax.plot(HR*np.sin(th),HR*np.cos(th),color='black',lw=1.2,zorder=5)
    ax.plot([-0.06,0,0.06],[HR,HR+0.07,HR],color='black',lw=1.2,zorder=5)

def _topo(vals,ax,cmap='RdBu_r',vmin=None,vmax=None,title=''):
    x,y=CH_X.copy(),CH_Y.copy()
    if vmin is None: va=max(abs(np.nanmin(vals)),abs(np.nanmax(vals))); vmin,vmax=-va,va
    xi=np.linspace(-HR-0.05,HR+0.05,100); yi=xi; Xi,Yi=np.meshgrid(xi,yi)
    bt=np.linspace(0,2*np.pi,36,endpoint=False); bx=(HR+0.01)*np.sin(bt); by=(HR+0.01)*np.cos(bt)
    d=cdist(np.column_stack([bx,by]),np.column_stack([x,y])); w=1.0/(d+1e-6); w/=w.sum(1,keepdims=True)
    try:
        interp=CloughTocher2DInterpolator(np.column_stack([np.concatenate([x,bx]),np.concatenate([y,by])]),np.concatenate([vals,w@vals]))
        Zi=interp(Xi,Yi); Zi[np.sqrt(Xi**2+Yi**2)>HR]=np.nan
        ax.pcolormesh(Xi,Yi,Zi,cmap=cmap,vmin=vmin,vmax=vmax,shading='gouraud',zorder=1)
    except: pass
    _dhead(ax); ax.scatter(x,y,c='black',s=22,zorder=10,edgecolors='white',linewidths=0.5)
    for c,cx,cy in zip(CH_ORDER,x,y): ax.text(cx,cy+(0.04 if cy>=0 else -0.05),c,ha='center',va='center',fontsize=6.5,fontweight='bold',zorder=11)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title,fontweight='bold',fontsize=10,pad=8)

def gen_psd(X,y,lm,od,tag):
    _pstyle(); fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True); axes=axes.flatten(); fig.patch.set_facecolor('white')
    for ci,cn in enumerate(CH_ORDER):
        ax=axes[ci]; f0=welch(X[0][ci],fs=SFREQ,nperseg=min(256,X[0].shape[1]))[0]
        for cv in sorted(lm.keys()):
            psds=np.array([10*np.log10(welch(t[ci],fs=SFREQ,nperseg=min(256,t.shape[1]))[1]+1e-20) for t in X[y==cv]])
            mp=np.mean(psds,0); sp=np.std(psds,0); c=CC[cv%len(CC)]
            ax.plot(f0,mp,color=c,label=str(lm[cv]),lw=1.3); ax.fill_between(f0,mp-sp,mp+sp,color=c,alpha=0.15)
        ax.set_xlim(2,40); ax.set_title(cn,fontweight='bold')
        if ci==0: ax.legend(fontsize=7)
        if ci>=4: ax.set_xlabel('Freq (Hz)')
        if ci%4==0: ax.set_ylabel('PSD (dB/Hz)')
    fig.suptitle(f'PSD — {tag}',fontsize=13,fontweight='bold',y=1.01); fig.tight_layout()
    p=os.path.join(od,f'psd_{tag}.png'); fig.savefig(p); plt.close(fig); del fig; gc.collect(); return p

def gen_erd(X, y, lm, rest_segs, od, tag):
    _pstyle(); tchs=['C3','Cz','C4']; bands=[('mu (8-13 Hz)',8,13),('beta (13-30 Hz)',13,30)]
    has_rest = rest_segs is not None and len(rest_segs) > 0
    ref={}
    if has_rest:
        for bi,(bn,lo,hi) in enumerate(bands):
            for ci,cn in enumerate(tchs):
                chi=CH_ORDER.index(cn) if cn in CH_ORDER else 0; pws=[]
                for seg in rest_segs:
                    if seg.shape[1]<int(SFREQ*0.3): continue
                    sf=bandpass_filter(seg[chi],lo,hi); pws.append(np.mean(np.abs(hilbert(sf))**2))
                ref[(bi,ci)]=np.mean(pws) if pws else None
    else:
        for bi in range(len(bands)):
            for ci in range(len(tchs)): ref[(bi,ci)]=None
    fig,axes=plt.subplots(len(bands),len(tchs),figsize=(14,7),sharex=True); fig.patch.set_facecolor('white')
    wn=X.shape[2]; t=np.arange(wn)/SFREQ; fb=False
    for bi,(bn,lo,hi) in enumerate(bands):
        for ci,cn in enumerate(tchs):
            chi=CH_ORDER.index(cn) if cn in CH_ORDER else 0; ax=axes[bi,ci]; R=ref.get((bi,ci))
            for cv in sorted(lm.keys()):
                erds=[]
                for trial in X[y==cv]:
                    sf=bandpass_filter(trial[chi],lo,hi); pw=np.abs(hilbert(sf))**2
                    if R is not None and R>1e-12: erds.append(((pw-R)/R)*100)
                    else: fb=True; bl=np.mean(pw[:max(int(0.5*SFREQ),1)]); erds.append(((pw-bl)/(bl+1e-12))*100)
                erds=np.array(erds); me=np.mean(erds,0); se=np.std(erds,0)/np.sqrt(max(len(erds),1))
                c=CC[cv%len(CC)]; ax.plot(t,me,color=c,label=str(lm[cv]),lw=1.2); ax.fill_between(t,me-se,me+se,color=c,alpha=0.15)
            ax.axhline(0,color='gray',ls='--',lw=0.8)
            if bi==0: ax.set_title(cn,fontweight='bold',fontsize=12)
            if ci==0: ax.set_ylabel(f'{bn}\nERD/ERS (%)')
            if bi==len(bands)-1: ax.set_xlabel('Time (s)')
            if bi==0 and ci==0: ax.legend(fontsize=8)
    bl_txt="REST segments" if has_rest and not fb else "first 0.5s fallback"
    fig.suptitle(f'ERD/ERS — {tag}\nPfurtscheller & Lopes da Silva (1999) | Baseline: {bl_txt}',fontsize=12,fontweight='bold',y=1.04)
    fig.tight_layout(); p=os.path.join(od,f'erd_ers_{tag}.png'); fig.savefig(p); plt.close(fig); del fig; gc.collect(); return p,fb

def gen_topo(X,y,lm,od,tag):
    _pstyle(); bands=[('mu',8,13),('beta',13,30)]
    nc=len(lm); ncols=nc+1 if nc>=2 else 1
    fig,axes=plt.subplots(len(bands),ncols,figsize=(4*ncols,4*len(bands)),squeeze=False); fig.patch.set_facecolor('white')
    for bi,(bn,lo,hi) in enumerate(bands):
        cp={}
        for cv in sorted(lm.keys()):
            pw=[]
            for t in X[y==cv]:
                cbp=[]
                for ci in range(N_CH):
                    fr,ps=welch(t[ci],fs=SFREQ,nperseg=min(256,t.shape[1]))
                    m=(fr>=lo)&(fr<=hi); cbp.append(np.log10(_np_integrate(ps[m],fr[m])+1e-12))
                pw.append(cbp)
            cp[cv]=np.mean(pw,0)
        vn=min(v.min() for v in cp.values()); vx=max(v.max() for v in cp.values())
        keys=sorted(lm.keys())
        for ci,cv in enumerate(keys): _topo(cp[cv],axes[bi,ci],cmap='YlOrRd',vmin=vn,vmax=vx,title=f'{bn} — {lm[cv]}')
        if nc>=2: _topo(cp[keys[0]]-cp[keys[1]],axes[bi,nc],cmap='RdBu_r',title=f'{bn} — Diff ({lm[keys[0]]}-{lm[keys[1]]})')
    fig.suptitle(f'Topo BP — {tag}',fontsize=13,fontweight='bold',y=1.02); fig.tight_layout()
    p=os.path.join(od,f'topo_{tag}.png'); fig.savefig(p); plt.close(fig); del fig; gc.collect(); return p

def gen_summary_plot(allr, lm, od, tag):
    """Consolidated classification summary: boxplot (top) + heatmap + best CM (bottom)."""
    _pstyle()
    n_methods = len(set(r['method'] for r in allr))
    n_clfs = len(set(r['clf'] for r in allr))
    n_classes = len(lm)
    fig_h = max(10, 5 + n_methods * 0.6)
    fig = plt.figure(figsize=(18, fig_h)); fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], width_ratios=[1.5, 1], hspace=0.35, wspace=0.3)

    # Top-left: Boxplot
    ax1 = fig.add_subplot(gs[0, :])
    rows = [{'Label': f"{r['method']}+{r['clf']}", 'Accuracy': a} for r in allr for a in r['fold_accs']]
    if rows:
        df = pd.DataFrame(rows); sns.boxplot(data=df, x='Label', y='Accuracy', ax=ax1, palette='Set2')
        chance = 1.0 / max(n_classes, 2)
        ax1.axhline(chance, ls='--', color='red', lw=1, label=f'Chance ({chance:.0%})')
        ax1.set_ylim(0, 1.05); ax1.legend(fontsize=7)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax1.set_title('Fold Accuracies', fontweight='bold')
    ax1.set_xlabel('')

    # Bottom-left: Heatmap (full space)
    ax2 = fig.add_subplot(gs[1, 0])
    pv = pd.DataFrame(allr).pivot_table(values='acc_mean', index='method', columns='clf', aggfunc='mean')
    if not pv.empty:
        # Custom annotation with automatic text color based on cell value
        sns.heatmap(pv * 100, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
                    vmin=30, vmax=100, linewidths=1, linecolor='white', square=False,
                    annot_kws={"size": 11, "weight": "bold"},
                    cbar_kws={"shrink": 0.7, "label": "Acc %"})
        
        for text in ax2.texts:
            try:
                val = float(text.get_text())
                text.set_color('white' if val > 75 else 'black')
            except: pass
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=9)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=9)
    ax2.set_title('Mean Accuracy (%) — Method × Classifier', fontweight='bold')

    # Bottom-right: Best CM
    ax3 = fig.add_subplot(gs[1, 1])
    if allr:
        best = max(allr, key=lambda r: r['acc_mean'])
        if best['y_true_all']:
            cm_ = confusion_matrix(best['y_true_all'], best['y_pred_all'])
            cn = [str(lm.get(c, c)) for c in sorted(lm.keys())]
            cp = cm_.astype(float) / (cm_.sum(1, keepdims=True) + 1e-12) * 100
            ax3.imshow(cp, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
            fs = 9 if n_classes <= 3 else 7
            for i in range(cm_.shape[0]):
                for j in range(cm_.shape[1]):
                    ax3.text(j, i, f'{cp[i,j]:.1f}%\n({cm_[i,j]})', ha='center', va='center', fontsize=fs,
                             color='white' if cp[i,j] > 55 else 'black')
            ax3.set_xticks(range(len(cn))); ax3.set_yticks(range(len(cn)))
            ax3.set_xticklabels(cn, fontsize=8); ax3.set_yticklabels(cn, fontsize=8)
            ax3.set_xlabel('Predicted', fontweight='bold'); ax3.set_ylabel('True', fontweight='bold')
            ax3.set_title(f"Best: {best['method']}+{best['clf']}\nAcc={best['acc_mean']:.1%} κ={best['kappa_mean']:.3f}",
                          fontweight='bold', fontsize=10)

    fig.suptitle(f'Classification Summary — {tag}', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(od, f'summary_{tag}.png'); fig.savefig(p); plt.close(fig); del fig; gc.collect(); return p

def save_csv(res,od,tag):
    if not res: return None
    df=pd.DataFrame(res)
    cols=[c for c in ['pair','method','clf','acc_mean','acc_std','kappa_mean','folds_inverted','folds_total'] if c in df.columns]
    p=os.path.join(od,f'results_{tag}.csv'); df[cols].to_csv(p,index=False); return p


#  GUI — Split-Pane with Tabbed Right Panel

class AnalysisApp:
    def __init__(self):
        global _sp
        if _sp is not None:
            try: _sp.destroy()
            except: pass
            _sp = None
        self.root=tk.Tk(); self.root.title(f"BCI-IM Analysis Suite v{VERSION}")
        self.root.configure(bg=TH["bg"]); self.root.geometry("1440x920"); self.root.minsize(1050,750)
        self.loaded_files=[]; self.raw_trials=[]; self.filt_trials=[]; self.trial_labels=np.array([])
        self.rest_segs=[]; self.is_running=False; self.log_q=queue.Queue()
        self.pfig=None; self.pax=None; self.pcv=None
        self._pv_offset=0; self._pv_win=10.0  # preview window seconds
        # Functional analysis state
        self._fa_fig=None; self._fa_cv=None
        self.V=type('V',(),{})()
        for n,v in [('mode','single'),('proto','graz_b'),
                    ('opath',os.path.join(os.path.expanduser("~"),"BCI_Analysis_Output")),
                    ('oname',''),
                    ('bp',True),('bp_lo','1.0'),('bp_hi','40.0'),('bp_ord','4'),
                    ('notch',True),('notch_f','60'),('car',True),('zs',False),
                    ('art',True),('art_p','85'),
                    ('win','3.0'),('ov','0.5'),('kf','5'),('feat','All'),('csp_nf','2'),('fb_ns','8'),
                    ('c_lda',True),('c_svm',True),('c_rf',True),('c_xgb',HAS_XGB),
                    ('c_3class',False),
                    ('p_psd',True),('p_erd',True),('p_topo',True),('p_summary',True),
                    ('pv_ch','All'),
                    # Individual preview preprocessing toggles
                    ('pv_bp',False),('pv_notch',False),('pv_car',False),('pv_zs',False),
                    # Functional analysis
                    ('fa_technique','ERD/ERS Interactive'),('fa_win','3.0'),('fa_band','mu (8-13)'),
                    ('fa_ch','C3'),('fa_rqa_dim','10'),('fa_rqa_tau','2')]:
            if isinstance(v,bool): setattr(self.V,n,tk.BooleanVar(value=v))
            else: setattr(self.V,n,tk.StringVar(value=str(v)))
        self._build(); self._poll()

    # Widget helpers 
    def _sec(self,p,t,ico=""):
        f=tk.Frame(p,bg=TH["sf"],highlightbackground=TH["bd"],highlightthickness=1,padx=12,pady=8)
        f.pack(fill="x",pady=(0,8))
        tk.Label(f,text=f"{ico}  {t}" if ico else t,font=("Segoe UI",9,"bold"),fg=TH["ac"],bg=TH["sf"]).pack(anchor="w",pady=(0,6))
        return f
    def _lbl(self,p,t,**k):
        d={"font":("Segoe UI",9),"fg":TH["tx"],"bg":TH["sf"]}; d.update(k); return tk.Label(p,text=t,**d)
    def _ent(self,p,v,w=10):
        return tk.Entry(p,textvariable=v,width=w,font=("Segoe UI",9),bg=TH["sf2"],fg=TH["tx"],
                        insertbackground=TH["tx"],relief="flat",highlightthickness=1,
                        highlightbackground=TH["bd"],highlightcolor=TH["ac"])
    def _btn(self,p,t,cmd,s="normal"):
        c={"normal":(TH["sf2"],TH["tx"],TH["ac"]),"accent":(TH["ad"],TH["ac"],TH["ac"]),
           "green":(TH["gd"],TH["gn"],TH["gn"]),"red":(TH["rdd"],TH["rd"],TH["rd"])}
        bg,fg,bc=c.get(s,c["normal"])
        return tk.Button(p,text=t,command=cmd,font=("Segoe UI",9,"bold"),bg=bg,fg=fg,
                         activebackground=bc,activeforeground="#FFF",relief="flat",cursor="hand2",
                         padx=10,pady=4,highlightthickness=1,highlightbackground=bc)
    def _chk(self,p,t,v):
        return tk.Checkbutton(p,text=t,variable=v,font=("Segoe UI",9),fg=TH["tx"],bg=TH["sf"],
                              selectcolor=TH["sf2"],activebackground=TH["sf"])

    # Main layout 
    def _build(self):
        bar=tk.Frame(self.root,bg=TH["sf2"],height=24); bar.pack(side="bottom",fill="x"); bar.pack_propagate(False)
        self.st_lbl=tk.Label(bar,text="Ready",font=("Segoe UI",8),fg=TH["tm"],bg=TH["sf2"]); self.st_lbl.pack(side="left",padx=12)
        pw=tk.PanedWindow(self.root,orient=tk.HORIZONTAL,bg=TH["bg"],sashwidth=6,sashrelief="flat")
        pw.pack(fill="both",expand=True)
        # LEFT panel — scrollable controls
        lf=tk.Frame(pw,bg=TH["bg"]); pw.add(lf,width=520,minsize=420)
        lc=tk.Canvas(lf,bg=TH["bg"],highlightthickness=0)
        ls=ttk.Scrollbar(lf,orient="vertical",command=lc.yview)
        cf=tk.Frame(lc,bg=TH["bg"])
        cf.bind("<Configure>",lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.create_window((0,0),window=cf,anchor="nw"); lc.configure(yscrollcommand=ls.set)
        lc.pack(side="left",fill="both",expand=True); ls.pack(side="right",fill="y")
        lc.bind("<Enter>",lambda e: lc.bind_all("<MouseWheel>",lambda ev: lc.yview_scroll(int(-1*(ev.delta/120)),"units")))
        lc.bind("<Leave>",lambda e: lc.unbind_all("<MouseWheel>"))
        ct=tk.Frame(cf,bg=TH["bg"]); ct.pack(fill="x",padx=14,pady=8)
        self._bld_ctrl(ct)
        # RIGHT panel — tabbed (Signals / Functional Analysis)
        rf=tk.Frame(pw,bg=TH["sf"]); pw.add(rf,minsize=400)
        self._bld_right_tabs(rf)

    # Left panel controls
    def _bld_ctrl(self, ct):
        h=tk.Frame(ct,bg=TH["bg"]); h.pack(fill="x",pady=(4,10))
        tk.Label(h,text="📊",font=("Segoe UI",18),fg=TH["ac"],bg=TH["bg"]).pack(side="left",padx=(0,8))
        tf=tk.Frame(h,bg=TH["bg"]); tf.pack(side="left")
        tk.Label(tf,text="BCI-IM Analysis Suite",font=("Segoe UI",14,"bold"),fg=TH["tx"],bg=TH["bg"]).pack(anchor="w")
        tk.Label(tf,text=f"v{VERSION} | PyRiemann:{'Y' if HAS_PYRIEMANN else 'N'} XGB:{'Y' if HAS_XGB else 'N'}",
                 font=("Segoe UI",8),fg=TH["tm"],bg=TH["bg"]).pack(anchor="w")
        # Data
        s=self._sec(ct,"INPUT DATA","📂")
        r=tk.Frame(s,bg=TH["sf"]); r.pack(fill="x",pady=(0,4))
        self._lbl(r,"Modo:").pack(side="left")
        for v,t in [("single","1 Session"),("multi","Multiple")]:
            tk.Radiobutton(r,text=t,variable=self.V.mode,value=v,font=("Segoe UI",8),fg=TH["tx"],bg=TH["sf"],selectcolor=TH["sf2"],activebackground=TH["sf"]).pack(side="left",padx=4)
        r2=tk.Frame(s,bg=TH["sf"]); r2.pack(fill="x",pady=(0,4))
        self._lbl(r2,"Protocolo:").pack(side="left")
        for v,t in [("graz_b","Graz B (MI)"),("movement","Movement (3 mov)")]:
            tk.Radiobutton(r2,text=t,variable=self.V.proto,value=v,font=("Segoe UI",8),fg=TH["tx"],bg=TH["sf"],selectcolor=TH["sf2"],activebackground=TH["sf"]).pack(side="left",padx=4)
        r3=tk.Frame(s,bg=TH["sf"]); r3.pack(fill="x",pady=(0,2))
        self._btn(r3,"Select File(s)",self._sel_files,"accent").pack(side="left")
        self.f_lbl=self._lbl(r3,"  None",fg=TH["tm"]); self.f_lbl.pack(side="left",padx=6)
        self.flist=tk.Listbox(s,height=2,font=("Consolas",7),bg=TH["sf2"],fg=TH["tx"],relief="flat",
                              highlightthickness=1,highlightbackground=TH["bd"])
        self.flist.pack(fill="x",pady=(2,0))
        # Preprocessing
        s=self._sec(ct,"PRE-PROCESSING","⚙")
        r=tk.Frame(s,bg=TH["sf"]); r.pack(fill="x",pady=1)
        self._chk(r,"Bandpass",self.V.bp).pack(side="left"); self._lbl(r," Lo:",font=("Segoe UI",8)).pack(side="left")
        self._ent(r,self.V.bp_lo,4).pack(side="left",padx=1); self._lbl(r,"Hi:",font=("Segoe UI",8)).pack(side="left")
        self._ent(r,self.V.bp_hi,4).pack(side="left",padx=1); self._lbl(r,"Ord:",font=("Segoe UI",8)).pack(side="left")
        self._ent(r,self.V.bp_ord,2).pack(side="left",padx=1)
        r=tk.Frame(s,bg=TH["sf"]); r.pack(fill="x",pady=1)
        self._chk(r,"Notch",self.V.notch).pack(side="left"); self._ent(r,self.V.notch_f,3).pack(side="left",padx=2)
        self._lbl(r,"Hz",font=("Segoe UI",8),fg=TH["tm"]).pack(side="left")
        self._chk(r,"  CAR",self.V.car).pack(side="left",padx=6); self._chk(r,"Z-Score",self.V.zs).pack(side="left")
        r=tk.Frame(s,bg=TH["sf"]); r.pack(fill="x",pady=1)
        self._chk(r,"Artifact Rejection",self.V.art).pack(side="left"); self._lbl(r," Percentil:",font=("Segoe UI",8)).pack(side="left")
        self._ent(r,self.V.art_p,3).pack(side="left",padx=2)
        # Segmentation
        s=self._sec(ct,"SEGMENTATION","🔲")
        r=tk.Frame(s,bg=TH["sf"]); r.pack(fill="x")
        self._lbl(r,"Window:").pack(side="left"); self._ent(r,self.V.win,4).pack(side="left",padx=2)
        self._lbl(r,"s  Overlap:").pack(side="left"); self._ent(r,self.V.ov,4).pack(side="left",padx=2)
        self._lbl(r," K-Folds:").pack(side="left"); self._ent(r,self.V.kf,3).pack(side="left",padx=2)
        # Features
        s=self._sec(ct,"FEATURE EXTRACTION","🧠")
        r=tk.Frame(s,bg=TH["sf"]); r.pack(fill="x",pady=(0,4))
        self._lbl(r,"Method:").pack(side="left")
        ttk.Combobox(r,textvariable=self.V.feat,values=["All"]+list(get_fmethods().keys()),
                     width=14,state="readonly",font=("Segoe UI",9)).pack(side="left",padx=6)
        # Classifiers
        s=self._sec(ct,"CLASSIFIERS","🎯")
        r=tk.Frame(s,bg=TH["sf"]); r.pack(fill="x")
        self._chk(r,"LDA",self.V.c_lda).pack(side="left",padx=(0,8))
        self._chk(r,"SVM",self.V.c_svm).pack(side="left",padx=(0,8))
        self._chk(r,"RF",self.V.c_rf).pack(side="left",padx=(0,8))
        cb=self._chk(r,"XGB",self.V.c_xgb); cb.pack(side="left")
        if not HAS_XGB: cb.config(state="disabled")
        r2=tk.Frame(s,bg=TH["sf"]); r2.pack(fill="x",pady=(4,0))
        self._chk(r2,"Multi-class (3 mov.)",self.V.c_3class).pack(side="left")
        self._lbl(r2,"  (OvO + 3-way)",font=("Segoe UI",7),fg=TH["tm"]).pack(side="left")
        # Plots
        s=self._sec(ct,"OUTPUT FIGURES","📈")
        r=tk.Frame(s,bg=TH["sf"]); r.pack(fill="x",pady=1)
        self._chk(r,"PSD",self.V.p_psd).pack(side="left",padx=(0,6))
        self._chk(r,"ERD/ERS",self.V.p_erd).pack(side="left",padx=(0,6))
        self._chk(r,"Topomap",self.V.p_topo).pack(side="left",padx=(0,6))
        self._chk(r,"Summary (Box+HM+CM)",self.V.p_summary).pack(side="left")
        # Output
        s=self._sec(ct,"OUTPUT","💾")
        r=tk.Frame(s,bg=TH["sf"]); r.pack(fill="x",pady=(0,4))
        self._lbl(r,"Folder:").pack(side="left"); self._ent(r,self.V.opath,28).pack(side="left",padx=4)
        self._btn(r,"...",self._br_out).pack(side="left")
        r=tk.Frame(s,bg=TH["sf"]); r.pack(fill="x")
        self._lbl(r,"Subfolder:").pack(side="left"); self._ent(r,self.V.oname,18).pack(side="left",padx=4)
        # Actions
        f=tk.Frame(ct,bg=TH["bg"]); f.pack(fill="x",pady=(4,8))
        self.btn_run=self._btn(f,"▶  RUN ANALYSIS",self._run,"green"); self.btn_run.pack(fill="x",ipady=6)
        r=tk.Frame(f,bg=TH["bg"]); r.pack(fill="x",pady=(6,0))
        self._btn(r,"📂 Open Output",self._open_out).pack(side="left",fill="x",expand=True,padx=(0,3))
        self._btn(r,"⬛ Stop",self._stop,"red").pack(side="left",fill="x",expand=True,padx=(3,0))
        # Log
        s=self._sec(ct,"LOG","📋")
        self.log_t=tk.Text(s,height=8,font=("Consolas",7),bg=TH["sf2"],fg=TH["tx"],relief="flat",
                           highlightthickness=1,highlightbackground=TH["bd"],wrap="word")
        self.log_t.pack(fill="both",expand=True)

    # Right panel: Tab
    def _bld_right_tabs(self, parent):
        style = ttk.Style()
        style.configure("Dark.TNotebook", background=TH["bg"])
        style.configure("Dark.TNotebook.Tab", background=TH["sf2"], foreground=TH["tx"],
                        padding=[12, 4], font=("Segoe UI", 9, "bold"))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", TH["sf"])], foreground=[("selected", TH["ac"])])
        self.nb = ttk.Notebook(parent, style="Dark.TNotebook")
        self.nb.pack(fill="both", expand=True)
        # Tab 1: Signal Preview
        tab1 = tk.Frame(self.nb, bg=TH["sf"]); self.nb.add(tab1, text="  📡 Signals  ")
        self._bld_signal_tab(tab1)
        # Tab 2: Functional Analysis
        tab2 = tk.Frame(self.nb, bg=TH["sf"]); self.nb.add(tab2, text="  🔬 Functional Analysis  ")
        self._bld_functional_tab(tab2)

    # Tab 1: Signal Preview 
    def _bld_signal_tab(self, parent):
        ctrl = tk.Frame(parent, bg=TH["sf"]); ctrl.pack(fill="x", padx=10, pady=(8, 4))
        # Individual preprocessing checkboxes
        tk.Label(ctrl, text="Pre-proc:", font=("Segoe UI", 8, "bold"), fg=TH["yl"], bg=TH["sf"]).pack(side="left")
        for txt, var in [("BP", self.V.pv_bp), ("Notch", self.V.pv_notch),
                         ("CAR", self.V.pv_car), ("Z-Score", self.V.pv_zs)]:
            tk.Checkbutton(ctrl, text=txt, variable=var, font=("Segoe UI", 8), fg=TH["tx"],
                           bg=TH["sf"], selectcolor=TH["sf2"], activebackground=TH["sf"],
                           command=self._upd_pv).pack(side="left", padx=2)
        tk.Label(ctrl, text="  Channel:", font=("Segoe UI", 8), fg=TH["tx"], bg=TH["sf"]).pack(side="left")
        cb = ttk.Combobox(ctrl, textvariable=self.V.pv_ch, values=["All"] + CH_ORDER,
                          width=8, state="readonly", font=("Segoe UI", 8))
        cb.pack(side="left", padx=4); cb.bind("<<ComboboxSelected>>", lambda e: self._upd_pv())
        # Navigation controls
        nav = tk.Frame(parent, bg=TH["sf"]); nav.pack(fill="x", padx=10, pady=(0, 4))
        self._btn(nav, "◀◀", lambda: self._pv_nav(-10), "accent").pack(side="left", padx=1)
        self._btn(nav, "◀", lambda: self._pv_nav(-1), "accent").pack(side="left", padx=1)
        self._pv_time_lbl = tk.Label(nav, text="0.0 – 10.0 s", font=("Segoe UI", 9, "bold"),
                                      fg=TH["tx"], bg=TH["sf"])
        self._pv_time_lbl.pack(side="left", padx=8)
        self._btn(nav, "▶", lambda: self._pv_nav(1), "accent").pack(side="left", padx=1)
        self._btn(nav, "▶▶", lambda: self._pv_nav(10), "accent").pack(side="left", padx=1)
        tk.Label(nav, text="  Trial:", font=("Segoe UI", 8), fg=TH["tm"], bg=TH["sf"]).pack(side="left", padx=(12, 2))
        self._pv_trial_var = tk.StringVar(value="1")
        self._pv_trial_spin = tk.Spinbox(nav, from_=1, to=1, textvariable=self._pv_trial_var,
                                          width=4, font=("Segoe UI", 8), bg=TH["sf2"], fg=TH["tx"],
                                          command=self._upd_pv)
        self._pv_trial_spin.pack(side="left")
        self._btn(nav, "🔄", self._upd_pv, "accent").pack(side="right")
        # Canvas placeholder
        self._pv_frame = tk.Frame(parent, bg=TH["sf"])
        self._pv_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.pfig, self.pax = plt.subplots(figsize=(6, 4))
        self.pfig.patch.set_facecolor(TH["sf"])
        self.pax.set_facecolor(TH["sf2"]); self.pax.tick_params(colors=TH["tm"])
        for sp in ['top', 'right']: self.pax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']: self.pax.spines[sp].set_color(TH["bd"])
        self.pax.set_title("Load data to visualize", color=TH["tm"], fontsize=10)
        self.pcv = FigureCanvasTkAgg(self.pfig, master=self._pv_frame)
        self.pcv.get_tk_widget().pack(fill="both", expand=True); self.pcv.draw()

    def _pv_nav(self, delta_s):
        """Navigate signal preview by delta_s seconds."""
        self._pv_offset = max(0, self._pv_offset + delta_s)
        self._upd_pv()

    def _get_pv_trials(self):
        """Get trials with individually selected preprocessing."""
        trials = [t.copy() for t in self.raw_trials]
        if not trials: return trials
        try:
            if self.V.pv_notch.get(): trials = pp_notch(trials, float(self.V.notch_f.get()))
            if self.V.pv_bp.get(): trials = pp_bp(trials, float(self.V.bp_lo.get()),
                                                   float(self.V.bp_hi.get()), SFREQ, int(self.V.bp_ord.get()))
            if self.V.pv_car.get(): trials = pp_car(trials)
            if self.V.pv_zs.get(): trials = pp_zs(trials)
        except: pass
        return trials

    def _upd_pv(self, *a):
        if self.pcv is None: return
        trials = self._get_pv_trials()
        if not trials:
            self.pfig.clear(); ax = self.pfig.add_subplot(111); ax.set_facecolor(TH["sf2"])
            ax.text(0.5, 0.5, "No data loaded\nSelect file(s)", ha='center', va='center',
                    transform=ax.transAxes, color=TH["tm"], fontsize=11)
            ax.set_xticks([]); ax.set_yticks([]); self.pcv.draw(); return
        # Trial selection
        try: ti = max(0, min(int(self._pv_trial_var.get()) - 1, len(trials) - 1))
        except: ti = 0
        self._pv_trial_spin.config(to=len(trials))
        trial = trials[ti]
        total_s = trial.shape[1] / SFREQ
        self._pv_offset = min(self._pv_offset, max(0, total_s - self._pv_win))
        s0 = int(self._pv_offset * SFREQ); s1 = min(int((self._pv_offset + self._pv_win) * SFREQ), trial.shape[1])
        t = np.arange(s1 - s0) / SFREQ + self._pv_offset
        self._pv_time_lbl.config(text=f"{self._pv_offset:.1f} – {self._pv_offset + self._pv_win:.1f} s  (total: {total_s:.1f} s)")
        sel = self.V.pv_ch.get()
        chs = [CH_ORDER.index(sel)] if sel in CH_ORDER else list(range(min(N_CH, trial.shape[0])))
        # Active filters label
        active = []
        if self.V.pv_bp.get(): active.append("BP")
        if self.V.pv_notch.get(): active.append("Notch")
        if self.V.pv_car.get(): active.append("CAR")
        if self.V.pv_zs.get(): active.append("ZS")
        filt_txt = "+".join(active) if active else "Raw"
        self.pfig.clear()
        if len(chs) == 1:
            ax = self.pfig.add_subplot(111); ax.set_facecolor(TH["sf2"]); ci = chs[0]
            ax.plot(t, trial[ci, s0:s1], lw=0.5, color=CH_COLORS[ci % 8], alpha=0.9)
            ax.set_title(f"{CH_ORDER[ci]} — {filt_txt} — Trial {ti+1}", color=TH["tx"], fontsize=10, fontweight='bold')
            ax.set_xlabel("Time (s)", color=TH["tm"], fontsize=8); ax.set_ylabel("µV", color=TH["tm"], fontsize=8)
            ax.tick_params(colors=TH["tm"], labelsize=7); ax.grid(True, alpha=0.15, color=TH["bd"])
            for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
            for sp in ['bottom', 'left']: ax.spines[sp].set_color(TH["bd"])
        else:
            axes = self.pfig.subplots(len(chs), 1, sharex=True)
            if len(chs) == 1: axes = [axes]
            for pi, ci in enumerate(chs):
                ax = axes[pi]; ax.set_facecolor(TH["sf2"]); c = CH_COLORS[ci % 8]
                ax.plot(t, trial[ci, s0:s1], lw=0.4, color=c, alpha=0.9)
                ax.set_ylabel(CH_ORDER[ci], color=c, fontsize=7, fontweight='bold', rotation=0, labelpad=20)
                ax.tick_params(colors=TH["tm"], labelsize=6)
                for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
                for sp in ['bottom', 'left']: ax.spines[sp].set_color(TH["bd"])
                ax.grid(True, alpha=0.1, color=TH["bd"])
                if pi < len(chs) - 1: ax.tick_params(labelbottom=False)
            axes[-1].set_xlabel("Time (s)", color=TH["tm"], fontsize=8)
            axes[0].set_title(f"{filt_txt} — Trial {ti+1}", color=TH["tx"], fontsize=10, fontweight='bold')
        self.pfig.tight_layout(); self.pcv.draw()

    # Tab 2: Functional Analysis
    def _bld_functional_tab(self, parent):
        ctrl = tk.Frame(parent, bg=TH["sf"]); ctrl.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(ctrl, text="Method:", font=("Segoe UI", 9, "bold"), fg=TH["ac"], bg=TH["sf"]).pack(side="left")
        techniques = ["ERD/ERS Interactive", "Classical Connectivity (Coherence)",
                      "RQA Connectivity (STR)", "Mean Recurrence Plot",
                      "Mean Smoothed Attractor", "CSP Spatial Filters",
                      "Nonlinear Metrics", "Phase-Amplitude Coupling"]
        ttk.Combobox(ctrl, textvariable=self.V.fa_technique, values=techniques,
                     width=30, state="readonly", font=("Segoe UI", 9)).pack(side="left", padx=6)
        # Params row
        p_row = tk.Frame(parent, bg=TH["sf"]); p_row.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(p_row, text="Band:", font=("Segoe UI", 8), fg=TH["tx"], bg=TH["sf"]).pack(side="left")
        ttk.Combobox(p_row, textvariable=self.V.fa_band,
                     values=["mu (8-13)", "beta (13-30)", "alpha (8-12)", "gamma (30-45)", "theta (4-8)", "broadband (1-40)"],
                     width=14, state="readonly", font=("Segoe UI", 8)).pack(side="left", padx=4)
        tk.Label(p_row, text="Channel:", font=("Segoe UI", 8), fg=TH["tx"], bg=TH["sf"]).pack(side="left", padx=(8, 2))
        ttk.Combobox(p_row, textvariable=self.V.fa_ch, values=CH_ORDER,
                     width=6, state="readonly", font=("Segoe UI", 8)).pack(side="left", padx=2)
        tk.Label(p_row, text="Win(s):", font=("Segoe UI", 8), fg=TH["tx"], bg=TH["sf"]).pack(side="left", padx=(8, 2))
        self._ent(p_row, self.V.fa_win, 4).pack(side="left", padx=2)
        # RQA params
        tk.Label(p_row, text="RQA dim:", font=("Segoe UI", 7), fg=TH["tm"], bg=TH["sf"]).pack(side="left", padx=(8, 1))
        self._ent(p_row, self.V.fa_rqa_dim, 3).pack(side="left")
        tk.Label(p_row, text="τ:", font=("Segoe UI", 7), fg=TH["tm"], bg=TH["sf"]).pack(side="left", padx=(4, 1))
        self._ent(p_row, self.V.fa_rqa_tau, 3).pack(side="left")
        # Buttons row
        btn_row = tk.Frame(parent, bg=TH["sf"]); btn_row.pack(fill="x", padx=10, pady=(0, 4))
        self._btn(btn_row, "▶ Compute", self._fa_compute, "green").pack(side="left", padx=(0, 4))
        # Navigation for ERD/ERS blocks
        self._fa_block_idx = 0
        self._btn(btn_row, "◀ Bloco", lambda: self._fa_nav(-1), "accent").pack(side="left", padx=2)
        self._fa_block_lbl = tk.Label(btn_row, text="Mean", font=("Segoe UI", 9), fg=TH["tx"], bg=TH["sf"])
        self._fa_block_lbl.pack(side="left", padx=4)
        self._btn(btn_row, "Bloco ▶", lambda: self._fa_nav(1), "accent").pack(side="left", padx=2)
        self._btn(btn_row, "Mean", lambda: self._fa_nav(0, reset=True), "accent").pack(side="left", padx=4)
        self._btn(btn_row, "💾 Save", self._fa_save, "normal").pack(side="right")
        # Info label
        self._fa_info = tk.Label(parent, text="Select a method and click Compute",
                                  font=("Segoe UI", 8), fg=TH["tm"], bg=TH["sf"], wraplength=600, justify="left")
        self._fa_info.pack(fill="x", padx=10, pady=(0, 4))
        # Canvas
        self._fa_frame = tk.Frame(parent, bg=TH["sf"])
        self._fa_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._fa_fig, self._fa_ax = plt.subplots(figsize=(7, 5))
        self._fa_fig.patch.set_facecolor(TH["sf"])
        self._fa_ax.set_facecolor(TH["sf2"]); self._fa_ax.set_title("Waiting...", color=TH["tm"])
        self._fa_cv = FigureCanvasTkAgg(self._fa_fig, master=self._fa_frame)
        self._fa_cv.get_tk_widget().pack(fill="both", expand=True); self._fa_cv.draw()

    def _fa_nav(self, delta, reset=False):
        if reset: self._fa_block_idx = 0
        else: self._fa_block_idx += delta
        if self.V.fa_technique.get() == "ERD/ERS Interactive": self._fa_compute()

    def _fa_save(self):
        """Save current functional analysis plot to output folder."""
        if self._fa_fig is None: return
        od = self._odir(); os.makedirs(od, exist_ok=True)
        tech = self.V.fa_technique.get().replace(" ", "_").replace("/", "_")[:30]
        ts = datetime.now().strftime("%H%M%S")
        p = os.path.join(od, f'fa_{tech}_{ts}.png')
        self._fa_fig.savefig(p, dpi=200, bbox_inches='tight', facecolor=self._fa_fig.get_facecolor())
        self._fa_info.config(text=f"✓ Salvo: {os.path.basename(p)}")
        self._log(f"  💾 FA salvo: {p}")

    def _parse_band(self):
        b = self.V.fa_band.get()
        bands_map = {"mu (8-13)": (8, 13), "beta (13-30)": (13, 30), "alpha (8-12)": (8, 12),
                     "gamma (30-45)": (30, 45), "theta (4-8)": (4, 8), "broadband (1-40)": (1, 40)}
        return bands_map.get(b, (8, 13))

    def _fa_compute(self):
        """Dispatch functional analysis computation in background thread."""
        if not self.raw_trials:
            self._fa_info.config(text="⚠ Load data first."); return
        tech = self.V.fa_technique.get()
        self._fa_info.config(text=f"Computing {tech}...")
        threading.Thread(target=self._fa_worker, args=(tech,), daemon=True).start()

    def _fa_worker(self, tech):
        try:
            trials = self._get_pv_trials()
            if not trials: return
            lo, hi = self._parse_band()
            ch_name = self.V.fa_ch.get()
            chi = CH_ORDER.index(ch_name) if ch_name in CH_ORDER else 0
            dim = int(self.V.fa_rqa_dim.get()); tau = int(self.V.fa_rqa_tau.get())

            self._fa_fig.clear()

            if tech == "ERD/ERS Interactive":
                self._fa_erd_interactive(trials, lo, hi, chi, ch_name)
            elif tech == "Classical Connectivity (Coherence)":
                self._fa_coherence(trials, lo, hi)
            elif tech == "RQA Connectivity (STR)":
                self._fa_str_connectivity(trials, lo, hi, dim, tau)
            elif tech == "Mean Recurrence Plot":
                self._fa_mean_rp(trials, chi, ch_name, dim, tau)
            elif tech == "Mean Smoothed Attractor":
                self._fa_mean_attractor(trials, chi, ch_name, dim, tau)
            elif tech == "CSP Spatial Filters":
                self._fa_csp_maps(trials)
            elif tech == "Nonlinear Metrics":
                self._fa_nonlinear_metrics(trials, lo, hi)
            elif tech == "Phase-Amplitude Coupling":
                self._fa_pac(trials, chi, ch_name)
            else:
                ax = self._fa_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Method '{tech}' not implemented", ha='center', va='center', transform=ax.transAxes)

            self._fa_fig.tight_layout()
            self.root.after(0, self._fa_cv.draw)
        except Exception as e:
            import traceback
            self.root.after(0, lambda: self._fa_info.config(text=f"✗ Error: {e}"))
            self._log(f"FA error: {traceback.format_exc()}")

    # FA: ERD/ERS Interactive 
    def _fa_erd_interactive(self, trials, lo, hi, chi, ch_name):
        bands = [('mu', 8, 13), ('beta', 13, 30)]
        tchs = ['C3', 'Cz', 'C4']
        rest = self.rest_segs
        labels = self.trial_labels
        ul = sorted(np.unique(labels)) if len(labels) else []
        n_trials = len(trials)
        if self._fa_block_idx == 0:
            # Mean across all trials
            title = "Mean of all trials"
            selected = list(range(n_trials))
        else:
            idx = max(1, min(abs(self._fa_block_idx), n_trials))
            self._fa_block_idx = idx if self._fa_block_idx > 0 else -idx
            selected = [idx - 1]
            lbl = labels[idx-1] if idx-1 < len(labels) else "?"
            title = f"Trial {idx}/{n_trials} (label={lbl})"
        self.root.after(0, lambda: self._fa_block_lbl.config(text=title))
        axes = self._fa_fig.subplots(len(bands), len(tchs), sharex=True)
        self._fa_fig.patch.set_facecolor('white')
        # Rest baseline
        ref = {}
        if rest:
            for bi, (bn, blo, bhi) in enumerate(bands):
                for ci, cn in enumerate(tchs):
                    chi_ = CH_ORDER.index(cn) if cn in CH_ORDER else 0; pws = []
                    for seg in rest:
                        if seg.shape[1] < int(SFREQ * 0.3): continue
                        sf = bandpass_filter(seg[chi_], blo, bhi); pws.append(np.mean(np.abs(hilbert(sf)) ** 2))
                    ref[(bi, ci)] = np.mean(pws) if pws else None
        min_len = min(t.shape[1] for t in [trials[i] for i in selected])
        tv = np.arange(min_len) / SFREQ
        for bi, (bn, blo, bhi) in enumerate(bands):
            for ci, cn in enumerate(tchs):
                chi_ = CH_ORDER.index(cn) if cn in CH_ORDER else 0
                ax = axes[bi, ci]; R = ref.get((bi, ci))
                erds = []
                for ti in selected:
                    trial = trials[ti]
                    sig = trial[chi_, :min_len]
                    sf = bandpass_filter(sig, blo, bhi); pw = np.abs(hilbert(sf)) ** 2
                    if R and R > 1e-12: erds.append(((pw - R) / R) * 100)
                    else:
                        bl = np.mean(pw[:max(int(0.5 * SFREQ), 1)])
                        erds.append(((pw - bl) / (bl + 1e-12)) * 100)
                erds = np.array(erds); me = np.mean(erds, 0); se = np.std(erds, 0) / np.sqrt(max(len(erds), 1))
                ax.plot(tv, me, color=CC[0], lw=1.2); ax.fill_between(tv, me - se, me + se, color=CC[0], alpha=0.2)
                ax.axhline(0, color='gray', ls='--', lw=0.8)
                if bi == 0: ax.set_title(cn, fontweight='bold')
                if ci == 0: ax.set_ylabel(f'{bn}\nERD/ERS (%)')
                if bi == len(bands) - 1: ax.set_xlabel('Time (s)')
        self._fa_fig.suptitle(f'ERD/ERS — {title}', fontsize=12, fontweight='bold')
        self.root.after(0, lambda: self._fa_info.config(
            text="◀▶ to browse individual trials | 'Mean' for the overall mean | Pfurtscheller & Lopes da Silva (1999)"))

    # FA: Classical Coherence Connectivity 
    def _fa_coherence(self, trials, lo, hi):
        n_ch = min(N_CH, trials[0].shape[0])
        coh_matrix = np.zeros((n_ch, n_ch))
        count = 0
        for trial in trials[:50]:  # limit for speed
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    f, Cxy = sp_coherence(trial[i], trial[j], fs=SFREQ, nperseg=min(256, trial.shape[1]))
                    mask = (f >= lo) & (f <= hi)
                    coh_matrix[i, j] += np.mean(Cxy[mask])
                    coh_matrix[j, i] = coh_matrix[i, j]
            count += 1
        coh_matrix /= max(count, 1)
        np.fill_diagonal(coh_matrix, 1.0)
        ax = self._fa_fig.add_subplot(111); self._fa_fig.patch.set_facecolor('white')
        im = ax.imshow(coh_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(n_ch)); ax.set_yticks(range(n_ch))
        ax.set_xticklabels(CH_ORDER[:n_ch], fontsize=8); ax.set_yticklabels(CH_ORDER[:n_ch], fontsize=8)
        self._fa_fig.colorbar(im, ax=ax, label='Coherence')
        ax.set_title(f'Coherence ({lo}-{hi} Hz) — {self.V.fa_band.get()}', fontweight='bold')
        self.root.after(0, lambda: self._fa_info.config(
            text="Spectral coherence matrix between channels. High values indicate phase/amplitude synchronization."))

    # FA: STR Connectivity (RQA-based) 
    def _fa_str_connectivity(self, trials, lo, hi, dim, tau):
        n_ch = min(N_CH, trials[0].shape[0])
        rr_matrix = np.zeros((n_ch, n_ch)); det_matrix = np.zeros((n_ch, n_ch))
        count = 0
        for trial in trials[:20]:  # limit
            filt = np.array([bandpass_filter(trial[ch], lo, hi) for ch in range(n_ch)])
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    crr, cdet = cross_recurrence_rqa(filt[i], filt[j], dim, tau)
                    rr_matrix[i, j] += crr; rr_matrix[j, i] += crr
                    det_matrix[i, j] += cdet; det_matrix[j, i] += cdet
            count += 1
        rr_matrix /= max(count, 1); det_matrix /= max(count, 1)
        axes = self._fa_fig.subplots(1, 2); self._fa_fig.patch.set_facecolor('white')
        for ax, mat, title in [(axes[0], rr_matrix, 'Cross-RR'), (axes[1], det_matrix, 'Cross-DET')]:
            im = ax.imshow(mat, cmap='YlOrRd', vmin=0)
            ax.set_xticks(range(n_ch)); ax.set_yticks(range(n_ch))
            ax.set_xticklabels(CH_ORDER[:n_ch], fontsize=7); ax.set_yticklabels(CH_ORDER[:n_ch], fontsize=7)
            self._fa_fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f'{title} ({lo}-{hi} Hz)', fontweight='bold', fontsize=10)
        self._fa_fig.suptitle('STR Connectivity — Rodrigues et al. (2019)', fontsize=12, fontweight='bold')
        self.root.after(0, lambda: self._fa_info.config(
            text="Cross-recurrence between channel pairs. RR=cross recurrence rate, DET=cross determinism. "
                 "High values indicate similar dynamics across regions (Rodrigues et al. 2019)."))

    # FA: Mean Recurrence Plot 
    def _fa_mean_rp(self, trials, chi, ch_name, dim, tau):
        labels = self.trial_labels; ul = sorted(np.unique(labels)) if len(labels) else [0]
        n_classes = min(len(ul), 3)
        axes = self._fa_fig.subplots(1, n_classes, squeeze=False)[0]; self._fa_fig.patch.set_facecolor('white')
        for ci, cl in enumerate(ul[:n_classes]):
            cl_trials = [trials[i] for i in range(len(trials)) if i < len(labels) and labels[i] == cl]
            if not cl_trials: continue
            # Average RP over first N trials
            rp_sum = None; count = 0
            for trial in cl_trials[:15]:
                sig = trial[chi][:int(3 * SFREQ)]  # 3s window
                traj = _embed(sig, dim, tau)
                if traj.shape[0] < 5: continue
                R, _ = _recurrence_matrix(traj)
                if rp_sum is None: rp_sum = R.astype(float)
                elif R.shape == rp_sum.shape: rp_sum += R
                count += 1
            if rp_sum is not None and count > 0:
                rp_mean = rp_sum / count
                axes[ci].imshow(rp_mean, cmap='binary', origin='lower', aspect='auto')
                # Get label name
                proto = self.V.proto.get()
                if proto == "graz_b": ln = GRAZ_N.get(cl, str(cl))
                else: ln = MOV_EN.get(cl - 19, MOV_EN.get(cl, str(cl)))
                axes[ci].set_title(f'{ln}\n{ch_name} (n={count})', fontweight='bold', fontsize=9)
            axes[ci].set_xlabel('Time index'); axes[ci].set_ylabel('Time index')
        self._fa_fig.suptitle(f'Mean Recurrence Plot — {ch_name} (dim={dim}, τ={tau})', fontsize=12, fontweight='bold')
        self.root.after(0, lambda: self._fa_info.config(
            text="Mean RP per class. Diagonal structures = determinism. Vertical = laminarity. "
                 "White = transitions between states (Webber & Zbilut 2005)."))

    # FA: Mean Attractor 
    def _fa_mean_attractor(self, trials, chi, ch_name, dim, tau):
        labels = self.trial_labels; ul = sorted(np.unique(labels)) if len(labels) else [0]
        n_classes = min(len(ul), 3)
        axes = self._fa_fig.subplots(1, n_classes, subplot_kw={'projection': '3d'}, squeeze=False)[0]
        self._fa_fig.patch.set_facecolor('white')
        for ci, cl in enumerate(ul[:n_classes]):
            cl_trials = [trials[i] for i in range(len(trials)) if i < len(labels) and labels[i] == cl]
            if not cl_trials: continue
            # Average smoothed attractor
            all_trajs = []
            for trial in cl_trials[:20]:
                sig = trial[chi][:int(3 * SFREQ)]
                traj = _embed(sig, min(3, dim), tau)
                if traj.shape[0] < 10 or traj.shape[1] < 3: continue
                # Smooth
                from scipy.ndimage import uniform_filter1d
                traj_s = np.column_stack([uniform_filter1d(traj[:, d], size=5) for d in range(3)])
                # Normalize to unit scale
                traj_s = (traj_s - traj_s.mean(0)) / (traj_s.std(0) + 1e-12)
                min_len = min(len(t) for t in all_trajs) if all_trajs else len(traj_s)
                min_len = min(min_len, len(traj_s))
                all_trajs.append(traj_s[:min_len])
            if all_trajs:
                min_len = min(len(t) for t in all_trajs)
                all_trajs = [t[:min_len] for t in all_trajs]
                mean_traj = np.mean(all_trajs, axis=0)
                ax = axes[ci]
                ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2], lw=0.6, color=CC[ci % len(CC)], alpha=0.8)
                proto = self.V.proto.get()
                if proto == "graz_b": ln = GRAZ_N.get(cl, str(cl))
                else: ln = MOV_EN.get(cl - 19, MOV_EN.get(cl, str(cl)))
                ax.set_title(f'{ln}', fontsize=9, fontweight='bold')
                ax.set_xlabel('v1', fontsize=7); ax.set_ylabel('v2', fontsize=7); ax.set_zlabel('v3', fontsize=7)
                ax.tick_params(labelsize=6)
        self._fa_fig.suptitle(f'Mean Smoothed Attractor — {ch_name} (dim=3, τ={tau})', fontsize=12, fontweight='bold')
        self.root.after(0, lambda: self._fa_info.config(
            text="Mean attractor reconstructed via time-delay embedding. Topological differences indicate "
                 "distinct dynamics across classes (Takens 1981)."))

    # FA: CSP Spatial Filters
    def _fa_csp_maps(self, trials):
        labels = self.trial_labels; ul = sorted(np.unique(labels))
        if len(ul) < 2:
            ax = self._fa_fig.add_subplot(111)
            ax.text(0.5, 0.5, "CSP requer ≥2 classes", ha='center', va='center', transform=ax.transAxes)
            return
        ca, cb = ul[0], ul[1]
        X_list = []; y_list = []
        for i, trial in enumerate(trials):
            if i >= len(labels): break
            if labels[i] in [ca, cb]:
                wn = int(SFREQ * float(self.V.win.get()))
                if trial.shape[1] >= wn:
                    X_list.append(trial[:, :wn]); y_list.append(0 if labels[i] == ca else 1)
        if len(X_list) < 4: return
        X = np.stack(X_list); y = np.array(y_list)
        csp = CSP_Reg(nf=2); csp.fit(X, y)
        if csp.W is None: return
        n_filters = csp.W.shape[1]
        axes = self._fa_fig.subplots(1, n_filters, squeeze=False)[0]; self._fa_fig.patch.set_facecolor('white')
        for fi in range(n_filters):
            w = csp.W[:, fi]
            _topo(w, axes[fi], cmap='RdBu_r', title=f'Filter {fi+1}')
        proto = self.V.proto.get()
        if proto == "graz_b": na, nb = GRAZ_N.get(ca, str(ca)), GRAZ_N.get(cb, str(cb))
        else: na, nb = MOV_EN.get(ca-19, str(ca)), MOV_EN.get(cb-19, str(cb))
        self._fa_fig.suptitle(f'CSP Spatial Filters — {na} vs {nb}', fontsize=12, fontweight='bold')
        self.root.after(0, lambda: self._fa_info.config(
            text="CSP spatial filters. First/last filters maximize the variance of each class. "
                 "Positive (red) and negative (blue) weights indicate each electrode's contribution."))

    # FA: Nonlinear Metrics 
    def _fa_nonlinear_metrics(self, trials, lo, hi):
        labels = self.trial_labels; ul = sorted(np.unique(labels))
        win_s = float(self.V.fa_win.get()); wn = int(SFREQ * win_s)
        n_ch = min(N_CH, trials[0].shape[0])
        proto = self.V.proto.get()
        # Compute metrics per class, per channel
        class_metrics = {}  # {class_label: {metric_name: [ch0_val, ch1_val, ...]}}
        for cl in ul:
            cl_trials = [trials[i] for i in range(min(len(trials), len(labels))) if labels[i] == cl]
            if not cl_trials: continue
            all_m = {nm: [] for nm in NL_METRIC_NAMES}
            for ch in range(n_ch):
                ch_vals = {nm: [] for nm in NL_METRIC_NAMES}
                for trial in cl_trials[:30]:
                    seg = trial[ch, :wn] if trial.shape[1] >= wn else trial[ch]
                    if lo > 0 and hi < SFREQ/2:
                        try: seg = bandpass_filter(seg, lo, hi)
                        except: pass
                    vals = compute_nl_metrics(seg)
                    for mi, nm in enumerate(NL_METRIC_NAMES): ch_vals[nm].append(vals[mi])
                for nm in NL_METRIC_NAMES: all_m[nm].append(np.mean(ch_vals[nm]))
            class_metrics[cl] = all_m
        # Plot: barplots per metric
        n_metrics = len(NL_METRIC_NAMES)
        rows = 2; cols = (n_metrics + 1) // 2
        axes = self._fa_fig.subplots(rows, cols, squeeze=False); self._fa_fig.patch.set_facecolor('white')
        x = np.arange(n_ch); width = 0.8 / max(len(ul), 1)
        for mi, nm in enumerate(NL_METRIC_NAMES):
            ax = axes[mi // cols, mi % cols]
            for ci, cl in enumerate(ul):
                if cl not in class_metrics: continue
                vals = class_metrics[cl][nm]
                if proto == "graz_b": ln = GRAZ_N.get(cl, str(cl))
                else: ln = MOV_EN.get(cl - 19, MOV_EN.get(cl, str(cl)))
                ax.bar(x + ci * width, vals, width, label=ln, color=CC[ci % len(CC)], alpha=0.8)
            ax.set_xticks(x + width * len(ul) / 2); ax.set_xticklabels(CH_ORDER[:n_ch], fontsize=6, rotation=45)
            ax.set_title(nm, fontsize=9, fontweight='bold')
            if mi == 0: ax.legend(fontsize=6)
        # Hide unused axes
        for mi in range(n_metrics, rows * cols):
            axes[mi // cols, mi % cols].axis('off')
        self._fa_fig.suptitle(f'Nonlinear Dynamics Metrics ({lo}-{hi} Hz, win={win_s}s)\nStam (2005), Abasolo et al. (2006)',
                               fontsize=11, fontweight='bold')
        # Show interpretation
        interp_text = " | ".join([f"{nm}: {NL_INTERPRETATIONS[nm][:60]}..." for nm in NL_METRIC_NAMES[:3]])
        self.root.after(0, lambda: self._fa_info.config(text=interp_text))

    # FA: Phase-Amplitude Coupling 
    def _fa_pac(self, trials, chi, ch_name):
        """Phase-Amplitude Coupling — Canolty et al. (2006)."""
        phase_bands = [(4, 8, 'θ'), (8, 13, 'α')]
        amp_bands = [(30, 45, 'γ_low'), (13, 30, 'β')]
        labels = self.trial_labels; ul = sorted(np.unique(labels))
        n_classes = min(len(ul), 3)
        axes = self._fa_fig.subplots(n_classes, len(phase_bands), squeeze=False); self._fa_fig.patch.set_facecolor('white')
        for ci, cl in enumerate(ul[:n_classes]):
            cl_trials = [trials[i] for i in range(min(len(trials), len(labels))) if labels[i] == cl]
            for pi, (plo, phi, pname) in enumerate(phase_bands):
                mi_vals = np.zeros(len(amp_bands))
                for ai, (alo, ahi, aname) in enumerate(amp_bands):
                    pac_sum = 0; count = 0
                    for trial in cl_trials[:20]:
                        sig = trial[chi]
                        try:
                            phase_sig = np.angle(hilbert(bandpass_filter(sig, plo, phi)))
                            amp_sig = np.abs(hilbert(bandpass_filter(sig, alo, ahi)))
                            # Modulation Index (Tort et al. 2010)
                            n_bins = 18; phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
                            amp_means = np.zeros(n_bins)
                            for b in range(n_bins):
                                mask = (phase_sig >= phase_bins[b]) & (phase_sig < phase_bins[b + 1])
                                amp_means[b] = np.mean(amp_sig[mask]) if np.any(mask) else 0
                            amp_means /= (np.sum(amp_means) + 1e-12)
                            uniform = np.ones(n_bins) / n_bins
                            kl = np.sum(amp_means * np.log((amp_means + 1e-12) / uniform))
                            mi = kl / np.log(n_bins)
                            pac_sum += mi; count += 1
                        except: pass
                    mi_vals[ai] = pac_sum / max(count, 1)
                ax = axes[ci, pi]
                ax.bar([a[2] for a in amp_bands], mi_vals, color=CC[ci % len(CC)], alpha=0.8)
                proto = self.V.proto.get()
                if proto == "graz_b": ln = GRAZ_N.get(cl, str(cl))
                else: ln = MOV_EN.get(cl - 19, MOV_EN.get(cl, str(cl)))
                ax.set_title(f'{ln} | Phase: {pname}', fontsize=8, fontweight='bold')
                ax.set_ylabel('MI' if pi == 0 else '', fontsize=7)
                ax.tick_params(labelsize=7)
        self._fa_fig.suptitle(f'Phase-Amplitude Coupling — {ch_name}\nTort et al. (2010), Canolty et al. (2006)',
                               fontsize=11, fontweight='bold')
        self.root.after(0, lambda: self._fa_info.config(
            text="Modulation Index (MI): coupling between low-frequency phase and high-frequency amplitude. "
                 "High MI indicates that gamma/beta amplitude is modulated by theta/alpha phase."))

    # File / data management
    def _sel_files(self):
        if self.V.mode.get() == "single":
            fp = filedialog.askopenfilename(title="CSV", filetypes=[("CSV", "*.csv")])
            if fp: self.loaded_files = [fp]
        else:
            fps = filedialog.askopenfilenames(title="CSVs", filetypes=[("CSV", "*.csv")])
            if fps: self.loaded_files = list(fps)
        n = len(self.loaded_files)
        self.f_lbl.config(text=f"  {n} file{'s' if n != 1 else ''}", fg=TH["gn"] if n else TH["tm"])
        self.flist.delete(0, tk.END)
        for f in self.loaded_files: self.flist.insert(tk.END, os.path.basename(f))
        if self.loaded_files: self._preload()

    def _preload(self):
        try:
            self.raw_trials = []; self.rest_segs = []; self.trial_labels = np.array([])
            proto = self.V.proto.get()
            for fp in self.loaded_files:
                df, _ = load_csv(fp)
                trials, labels = extract_graz(df) if proto == "graz_b" else extract_mov(df)
                self.raw_trials.extend(trials)
                self.trial_labels = np.concatenate([self.trial_labels, labels]) if len(self.trial_labels) else labels
                self.rest_segs.extend(extract_rest(df))
            self._pv_offset = 0
            self._pv_trial_spin.config(to=max(len(self.raw_trials), 1))
            self._apply_filt(); self._upd_pv()
            self._log(f"Preview: {len(self.raw_trials)} trials, {len(self.rest_segs)} rest segs")
        except Exception as e: self._log(f"Preview err: {e}")

    def _apply_filt(self):
        self.filt_trials = [t.copy() for t in self.raw_trials]
        try:
            if self.V.notch.get(): self.filt_trials = pp_notch(self.filt_trials, float(self.V.notch_f.get()))
            if self.V.bp.get(): self.filt_trials = pp_bp(self.filt_trials, float(self.V.bp_lo.get()),
                                                          float(self.V.bp_hi.get()), SFREQ, int(self.V.bp_ord.get()))
            if self.V.car.get(): self.filt_trials = pp_car(self.filt_trials)
            if self.V.zs.get(): self.filt_trials = pp_zs(self.filt_trials)
        except: pass

    def _br_out(self): p = filedialog.askdirectory(initialdir=self.V.opath.get()); p and self.V.opath.set(p)
    def _open_out(self):
        if self.is_running: messagebox.showwarning("", "Please wait for the analysis to finish."); return
        p = self._odir()
        if os.path.isdir(p): os.startfile(p)
        elif os.path.isdir(self.V.opath.get()): os.startfile(self.V.opath.get())
    def _odir(self):
        s = self.V.oname.get().strip() or datetime.now().strftime("analysis_%Y%m%d_%H%M%S")
        return os.path.join(self.V.opath.get(), s)

    def _log(self, m): self.log_q.put(m)
    def _poll(self):
        try:
            while not self.log_q.empty():
                self.log_t.insert(tk.END, self.log_q.get_nowait() + "\n"); self.log_t.see(tk.END)
            self.root.after(100, self._poll)
        except tk.TclError:
            pass  # window was destroyed
    def _stop(self): self.is_running = False; self._log("⬛ Stopped.")

    # Analysis pipeline
    def _run(self):
        if self.is_running: return
        if not self.loaded_files: messagebox.showwarning("", "Select file(s)."); return
        od = self._odir(); os.makedirs(od, exist_ok=True)
        self.is_running = True; self.btn_run.config(state="disabled"); self.log_t.delete("1.0", tk.END)
        self._apply_filt(); self._upd_pv()
        threading.Thread(target=self._worker, args=(od,), daemon=True).start()

    def _worker(self, od):
        try:
            import matplotlib as _m; _m.use("Agg", force=True); plt.switch_backend("Agg")
            self._log("=" * 50 + f"\n  BCI-IM Analysis Pipeline v{VERSION}\n" + "=" * 50)
            proto = self.V.proto.get()
            self._log("\n[1/6] Loading...")
            at = []; al = []; ar = []
            for fp in self.loaded_files:
                if not self.is_running: return
                self._log(f"  → {os.path.basename(fp)}")
                df, _ = load_csv(fp)
                trials, labels = extract_graz(df) if proto == "graz_b" else extract_mov(df)
                rest = extract_rest(df)
                if not trials: self._log("    ⚠ 0 trials"); continue
                self._log(f"    {len(trials)} trials, {len(rest)} rest segments")
                at.extend(trials); al.extend(labels.tolist()); ar.extend(rest)
            if not at: self._log("\n✗ No trials."); self._fin(); return
            labels = np.array(al)
            self._log(f"\n  Total: {len(at)} trials, {len(ar)} rest segments")
            self._log(f"  Classes: {dict(zip(*np.unique(labels, return_counts=True)))}")

            self._log("\n[2/6] Pre-processing...")
            rp = [r.copy() for r in ar]
            if self.V.notch.get():
                f = float(self.V.notch_f.get()); at = pp_notch(at, f); rp = pp_notch(rp, f); self._log(f"  ✓ Notch {f}Hz")
            if self.V.bp.get():
                lo = float(self.V.bp_lo.get()); hi = float(self.V.bp_hi.get()); o = int(self.V.bp_ord.get())
                at = pp_bp(at, lo, hi, SFREQ, o); rp = pp_bp(rp, lo, hi, SFREQ, o)
                self._log(f"  ✓ Bandpass {lo}-{hi}Hz (ord {o})")
            if self.V.car.get(): at = pp_car(at); rp = pp_car(rp); self._log("  ✓ CAR")
            if self.V.zs.get(): at = pp_zs(at); rp = pp_zs(rp); self._log("  ✓ Z-Score")

            ul = sorted(np.unique(labels))
            if proto == "graz_b":
                pairs = [(ul[0], ul[1])] if len(ul) >= 2 else []
                ln = {l: GRAZ_N.get(l, f"C{l}") for l in ul}
            else:
                pairs = list(itertools.combinations(ul, 2))
                ln = {l: MOV_EN.get(l - 19, MOV_EN.get(l, f"C{l}")) for l in ul}

            if len(ul) < 2:
                self._log(f"\n  ⚠ Only 1 class. Skipping classification.")
                self._log(f"\n{'=' * 50}\n  DONE! → {od}\n{'=' * 50}")
                self._fin(); return

            ws = float(self.V.win.get()); ov = float(self.V.ov.get()); nf = int(self.V.kf.get())
            fs = self.V.feat.get(); am = list(get_fmethods().keys())
            meth = am if fs == "All" else [fs]
            cn = []
            if self.V.c_lda.get(): cn.append('LDA')
            if self.V.c_svm.get(): cn.append('SVM')
            if self.V.c_rf.get(): cn.append('RF')
            if self.V.c_xgb.get() and HAS_XGB: cn.append('XGB')
            if not cn: self._log("✗ No classifier!"); self._fin(); return

            self._log(f"\n[3/6] Segmentation (win={ws}s, overlap={ov})")
            self._log(f"  Features: {meth}\n  Classificadores: {cn}")

            allr = []
            for ca, cb in pairs:
                if not self.is_running: return
                na = ln.get(ca, str(ca)); nb = ln.get(cb, str(cb))
                pt = f"{ca}v{cb}"
                self._log(f"\n  ── {na} vs {nb} ──")
                X, y, g = seg_trials(at, labels, ca, cb, ws, ov)
                if X is None: self._log("    ⚠ Insufficient data."); continue
                self._log(f"    Windows: {len(X)} (c0:{np.sum(y == 0)}, c1:{np.sum(y == 1)})")
                lm = {0: na, 1: nb}
                if self.V.art.get():
                    pc = float(self.V.art_p.get()); X, y, g, ri = rej_art(X, y, g, pc)
                    self._log(f"    Artifact rej: {ri['n_rej']}/{ri['n_total']} ({ri['pct']:.1f}%)")
                    if len(X) < 4: self._log("    ⚠ Insufficient."); continue
                self._log("\n[4/6] Generating figures...")
                if self.V.p_psd.get(): gen_psd(X, y, lm, od, pt); self._log("    ✓ PSD")
                if self.V.p_erd.get():
                    _, fb = gen_erd(X, y, lm, rp, od, pt)
                    self._log(f"    ✓ ERD/ERS {'(fallback)' if fb else '(rest baseline)'}")
                if self.V.p_topo.get(): gen_topo(X, y, lm, od, pt); self._log("    ✓ Topomap")
                self._log(f"\n[5/6] Classification ({pt})...")
                pr = run_clf(X, y, g, meth, cn, nf, self._log)
                for r in pr:
                    r['pair'] = pt; allr.append(r)
                    inv = f" [INV {r['folds_inverted']}/{r['folds_total']}]" if r.get('folds_inverted', 0) else ""
                    self._log(f"    {r['method']:12s}+{r['clf']:4s}: Acc={r['acc_mean']:.1%}±{r['acc_std']:.1%} κ={r['kappa_mean']:.3f}{inv}")
                if self.V.p_summary.get() and pr:
                    gen_summary_plot(pr, lm, od, pt); self._log("    ✓ Summary plot")
                self._close_worker_figs(); gc.collect()

            # 3-class classification for movement protocol
            if self.V.c_3class.get() and proto == "movement" and len(ul) >= 3:
                self._log(f"\n  ── Multi-class (3-way) ──")
                X3, y3, g3 = seg_trials_multi(at, labels, ul, ws, ov)
                if X3 is not None:
                    lm3 = {i: ln.get(c, str(c)) for i, c in enumerate(sorted(ul))}
                    self._log(f"    Windows: {len(X3)}, classes: {dict(zip(*np.unique(y3, return_counts=True)))}")
                    if self.V.art.get():
                        pc = float(self.V.art_p.get()); X3, y3, g3, ri = rej_art(X3, y3, g3, pc)
                        self._log(f"    Artifact rej: {ri['n_rej']}/{ri['n_total']}")
                    if len(X3) < 6:
                        self._log("    ⚠ Insufficient data for 3-class after rejection.")
                    else:
                        n_groups = len(np.unique(g3))
                        self._log(f"    Groups: {n_groups}, samples: {len(X3)}")
                        if self.V.p_psd.get(): gen_psd(X3, y3, lm3, od, "3class"); self._log("    ✓ PSD 3-class")
                        if self.V.p_topo.get(): gen_topo(X3, y3, lm3, od, "3class"); self._log("    ✓ Topo 3-class")
                        self._log(f"\n    3-way classification...")
                        try:
                            pr3 = run_clf(X3, y3, g3, meth, cn, nf, self._log)
                            self._log(f"    → {len(pr3)} results obtained")
                            for r in pr3:
                                r['pair'] = '3class'; allr.append(r)
                                self._log(f"    {r['method']:12s}+{r['clf']:4s}: Acc={r['acc_mean']:.1%}±{r['acc_std']:.1%} κ={r['kappa_mean']:.3f}")
                            if self.V.p_summary.get() and pr3:
                                gen_summary_plot(pr3, lm3, od, "3class"); self._log("    ✓ Summary 3-class")
                            elif not pr3:
                                self._log("    ⚠ No 3-way classification result (insufficient data per fold?)")
                        except Exception as e:
                            self._log(f"    ✗ 3-class error: {e}")
                            import traceback; self._log(traceback.format_exc())
                        self._close_worker_figs(); gc.collect()
                else:
                    self._log("    ⚠ Multi-class segmentation failed.")

            self._log(f"\n[6/6] Final summary...")
            if allr:
                save_csv(allr, od, "summary"); self._log("    ✓ CSV")
                self._log("\n  ═══ TOP 5 RESULTS ═══")
                for _, r in pd.DataFrame(allr).nlargest(5, 'acc_mean').iterrows():
                    self._log(f"    {r['pair']} | {r['method']}+{r['clf']} | Acc={r['acc_mean']:.1%} | κ={r['kappa_mean']:.3f}")
            self._log(f"\n{'=' * 50}\n  DONE! → {od}\n{'=' * 50}")
        except Exception as e:
            self._log(f"\n✗ ERROR: {e}"); import traceback; self._log(traceback.format_exc())
        finally:
            self._fin()

    def _close_worker_figs(self):
        """Close only Agg-backend figures, preserving TkAgg preview/FA figures."""
        for fig_num in list(plt.get_fignums()):
            try:
                fig = plt.figure(fig_num)
                if fig is not self.pfig and fig is not self._fa_fig:
                    plt.close(fig)
            except: pass

    def _fin(self):
        self.is_running = False
        try:
            for fig_num in list(plt.get_fignums()):
                fig = plt.figure(fig_num)
                if fig is not self.pfig and fig is not self._fa_fig:
                    plt.close(fig)
            gc.collect()
        except: pass
        try:
            import matplotlib as _m; _m.use("TkAgg", force=True); plt.switch_backend("TkAgg")
        except: pass
        try:
            self.root.after(0, lambda: self.btn_run.config(state="normal"))
            self.root.after(0, lambda: self.st_lbl.config(text="Done", fg=TH["gn"]))
        except tk.TclError: pass

    def run(self): self.root.mainloop()

if __name__ == "__main__":
    np.random.seed(RANDOM_STATE); AnalysisApp().run()
