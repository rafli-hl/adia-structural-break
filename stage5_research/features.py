"""Causal normal-score representation. No evaluator metadata enters state."""
import hashlib
import json
import math
import numpy as np
from scipy.special import ndtri
from sbrt.history import EmpiricalCDF, residuals
from sbrt.features import RollingMean
from sbrt.stage2_detectors import _encode, _decode, _array_bytes
from stage4_research.features import FeatureState as E14State
from .contract import BASE, MOMENTS, LIKELIHOOD, TRAJECTORY, CALIBRATED, IDS, WINDOWS


def hermite(z):
    z2=z*z
    return (z,(z2-1)/math.sqrt(2),(z2*z-3*z)/math.sqrt(6),(z2*z2-6*z2+3)/math.sqrt(24))


def historical_u(history, base):
    """Replay the frozen scale arithmetic, not another AR fit or residual procedure."""
    ar=base.ar
    e=residuals(history,(ar.mu,ar.sd,ar.beta))
    scale=base.scales["slow"]
    h=scale.b
    values=[]
    for j in range(5,len(e)):
        innovation=float(e[j])
        if j>=ar.cut: values.append(innovation/math.sqrt(max(h,scale.v_min)))
        h=(1.0-scale.alpha)*h+scale.alpha*(innovation*innovation)
    if h!=scale.h: raise AssertionError("Historical terminal scale differs from E14")
    result=np.asarray(values,dtype=np.float64)
    if float(result.mean())!=scale.mu or max(float(result.std(ddof=1)),1e-8)!=scale.sd:
        raise AssertionError("Historical C innovations differ from E14")
    return result


def bank(pair=False):
    return dict(windows=[RollingMean(w-int(pair)) for w in WINDOWS],n=0,mean=0.)


def accumulate(state,value):
    state["n"]+=1
    state["mean"]+=(value-state["mean"])/state["n"]
    result=[]
    for window in state["windows"]:
        avg=window.update(value)
        result.append((window.n,avg))
    result.append((state["n"],state["mean"]))
    return result


def aggregate(values):
    return (max(values[:3])+values[3])/2.0


def bounded(q):
    if not math.isfinite(q) or q<0: raise ValueError("Invalid nonnegative evidence")
    return q/(1.0+q)


def directional(n,mean,v0,eta=1.):
    """Returns finite bounded up/down plus numerical clamp/limit indicators."""
    clamped=0
    if mean<0:
        if mean < -1e-12*max(1.,v0): raise ValueError("Negative squared-energy rolling mean")
        mean=0.; clamped=1
    r=mean/v0
    if not math.isfinite(r) or eta<1 or not math.isfinite(eta): raise ValueError("Invalid likelihood scale")
    if r==0: return 0.,1.,clamped,1,0
    d=r-1.
    kernel=d-math.log1p(d) if abs(d)<=.5 else r-1.-math.log(r)
    kernel_clamp=int(kernel<0.)
    kernel=max(0.,kernel)
    value=bounded((n/2.0*kernel)/eta)
    return (value if r>1 else 0.),(value if r<1 else 0.),clamped,0,kernel_clamp


def hac(q,v0):
    q=np.asarray(q,dtype=np.float64)
    result=dict(eta=1.,supported=len(q)>=34,disabled=v0<=1e-16,
                gamma0=None,V_raw=None,V=None,covariances=[],roundoff_clamp=0)
    if not result["supported"] or result["disabled"]: return result
    centered=q-v0
    gamma=[float(centered@centered)/len(q)]
    gamma.extend(float(centered[lag:]@centered[:-lag])/len(q) for lag in range(1,17))
    value=gamma[0]+2*sum((1-lag/17)*gamma[lag] for lag in range(1,17))
    if value < -1e-12*max(1.,gamma[0]): raise ValueError("Invalid negative Bartlett covariance sum")
    result.update(gamma0=gamma[0],V_raw=value,V=max(value,0.),covariances=gamma,
        roundoff_clamp=int(value<0),eta=max(1.,max(value,0.)/(2*v0*v0)))
    return result


class Statistics:
    def __init__(self,z_reference,experiment):
        if experiment not in (*IDS,"export"): raise ValueError("Unregistered statistic set")
        self.experiment=experiment
        ref=np.asarray(z_reference,dtype=np.float64)
        if len(ref)<2 or not np.isfinite(ref).all(): raise ValueError("Invalid normal-score reference")
        self.mz=float(ref.mean())
        self.v0=float(np.mean((ref-self.mz)**2))
        self.calibration=hac((ref-self.mz)**2,self.v0) if experiment in ("E21","export") else None
        self.mu=[]; self.sd=[]; self.sd_raw=[]; self.moment_banks=[]
        if experiment in ("E18","export"):
            channels=list(hermite(ref))+[(ref[1:]-self.mz)*(ref[:-1]-self.mz)]
            for j,values in enumerate(channels):
                self.mu.append(float(values.mean()))
                self.sd_raw.append(float(values.std(ddof=1)))
                self.sd.append(max(self.sd_raw[-1],1e-8))
                self.moment_banks.append(bank(j==4))
        self.energy=bank() if experiment!="E18" else None
        self.previous=None
        self.age=0
        self.peak=self.trajectory=0.
        self.counts=dict(moment_sd_floor=sum(s<1e-8 for s in self.sd_raw),
            reference_disabled=int(self.v0<=1e-16),calibration_unsupported=int(self.calibration is not None and not self.calibration["supported"]),
            hac_clamp=self.calibration["roundoff_clamp"] if self.calibration else 0,
            moment_insufficient=[0]*4,pair_insufficient=[0]*4,likelihood_insufficient=[0]*4,
            zero_ratio=[0]*4,negative_mean_clamp=[0]*4,negative_kernel_clamp=[0]*4,
            likelihood_disabled=[0]*4)

    def update(self,z):
        if not math.isfinite(z): raise ValueError("Nonfinite normal score")
        self.age+=1
        row={}
        if self.moment_banks:
            values=list(hermite(z))+[None if self.previous is None else (z-self.mz)*(self.previous-self.mz)]
            for j,value in enumerate(values):
                out=[]
                stats=accumulate(self.moment_banks[j],(value-self.mu[j])/self.sd[j]) if value is not None else [(0,0.)]*4
                for w,(n,avg) in enumerate(stats):
                    if n<(8 if j==4 else 1):
                        self.counts["pair_insufficient" if j==4 else "moment_insufficient"][w]+=1
                        out.append(0.)
                    else: out.append(bounded(math.sqrt(n)*abs(avg)))
                row[MOMENTS[j]]=aggregate(out)
        if self.energy is not None:
            values=accumulate(self.energy,(z-self.mz)**2)
            up=[]; down=[]; cup=[]; cdown=[]
            for w,(n,avg) in enumerate(values):
                if self.v0<=1e-16:
                    self.counts["likelihood_disabled"][w]+=1
                    plus=minus=cplus=cminus=0.
                elif n<8:
                    self.counts["likelihood_insufficient"][w]+=1
                    plus=minus=cplus=cminus=0.
                else:
                    plus,minus,clamp,zero,kclamp=directional(n,avg,self.v0)
                    self.counts["negative_mean_clamp"][w]+=clamp
                    self.counts["zero_ratio"][w]+=zero
                    self.counts["negative_kernel_clamp"][w]+=kclamp
                    if self.calibration is not None:
                        cplus,cminus,*_=directional(n,avg,self.v0,self.calibration["eta"])
                    else: cplus=cminus=0.
                up.append(plus); down.append(minus); cup.append(cplus); cdown.append(cminus)
            row.update(zip(LIKELIHOOD,(aggregate(up),aggregate(down))))
            if self.calibration is not None: row.update(zip(CALIBRATED,(aggregate(cup),aggregate(cdown))))
            if self.experiment in ("E20","export"):
                current=max(row["L_plus"],row["L_minus"])
                self.peak=max(self.peak,current)
                self.trajectory=.99*self.trajectory+.01*current
                row.update(drawdown=self.peak-current,trajectory_ewma=self.trajectory)
        self.previous=z
        if not all(math.isfinite(v) and 0<=v<=1 for v in row.values()): raise ValueError("Invalid bounded feature")
        return row

    def reference_record(self):
        return dict(m_z=self.mz,v0=self.v0,channel_mean=self.mu,channel_sd=self.sd,
                    channel_sd_before_floor=self.sd_raw,calibration=self.calibration)


class FeatureState:
    def __init__(self,history,experiment,*,null_audit=False):
        self.experiment=experiment
        self.base=E14State(history,"E14")
        reference=historical_u(history,self.base)
        self.cdf=EmpiricalCDF(reference)
        z=ndtri(self.cdf.transform(reference))
        self.statistics=Statistics(z,experiment)
        self.counts=dict(below_reference=0,above_reference=0,lower_endpoint=0,upper_endpoint=0,observations=0)
        self.last={}
        self.null_rows=None
        if null_audit:
            audit=Statistics(z,"export")
            self.null_rows=[audit.update(float(v)) for v in z]
        self.historical=dict(M=len(z),u_min=float(reference.min()),u_max=float(reference.max()),
            z_mean=float(z.mean()),z_sd=float(z.std(ddof=1)),**self.statistics.reference_record())

    def observe(self,value):
        if self.base.pending: raise ValueError("Commit before consuming another point")
        innovation,_=self.base.ar.update(value)
        scale=self.base.scales["slow"].observe(innovation)
        u=scale["u"]
        p=float(self.cdf.transform(u)); z=float(ndtri(p))
        n=len(self.cdf.reference)
        self.counts["observations"]+=1
        self.counts["below_reference"]+=int(u<float(self.cdf.reference[0]))
        self.counts["above_reference"]+=int(u>float(self.cdf.reference[-1]))
        self.counts["lower_endpoint"]+=int(p==.5/(n+1))
        self.counts["upper_endpoint"]+=int(p==(n+.5)/(n+1))
        new=self.statistics.update(z)
        self.base.blocks.update(value)
        base={k:self.base.blocks.last_blocks[k] for k in BASE[:4]}
        base["J_s"]=scale["J"]
        self.base.last=base
        self.base.diagnostic=dict(e_squared=innovation*innovation,v_s=scale["variance"])
        self.base.pending=True
        self.last={**base,**new}
        return self.last

    def commit(self):
        self.base.commit()

    def dumps(self):
        fields={k:v for k,v in self.__dict__.items() if k not in ("base","statistics")}
        return json.dumps(dict(version=1,base=json.loads(self.base.dumps()),statistics=_encode(self.statistics.__dict__),
            fields=_encode(fields)),sort_keys=True,separators=(",",":"),allow_nan=False).encode()

    @classmethod
    def loads(cls,payload):
        v=json.loads(payload)
        if set(v)!={"version","base","statistics","fields"} or v["version"]!=1: raise ValueError("Invalid Stage-5 state")
        obj=cls.__new__(cls)
        obj.__dict__.update(_decode(v["fields"]))
        obj.base=E14State.loads(json.dumps(v["base"]))
        obj.statistics=Statistics.__new__(Statistics)
        obj.statistics.__dict__.update(_decode(v["statistics"]))
        if obj.experiment!=obj.statistics.experiment or obj.experiment not in (*IDS,"export"): raise ValueError("State mismatch")
        return obj

    def array_bytes(self):
        return _array_bytes(self,set())


class Detector:
    def __init__(self,history,model):
        self.features=FeatureState(history,model.pre.experiment)
        self.model=model

    def update(self,value):
        prediction=float(self.model.predict(self.features.observe(value)))
        self.features.commit()
        return prediction

    def dumps(self):
        return json.dumps(dict(version=1,model_sha256=hashlib.sha256(self.model.dumps()).hexdigest(),
            features=json.loads(self.features.dumps())),sort_keys=True,separators=(",",":"),allow_nan=False).encode()

    @classmethod
    def loads(cls,payload,model):
        v=json.loads(payload)
        if set(v)!={"version","model_sha256","features"} or v["version"]!=1 or v["model_sha256"]!=hashlib.sha256(model.dumps()).hexdigest():
            raise ValueError("State/model binding mismatch")
        obj=cls.__new__(cls); obj.model=model
        obj.features=FeatureState.loads(json.dumps(v["features"]))
        if obj.features.experiment!=model.pre.experiment: raise ValueError("Wrong feature state")
        return obj


def infer(datasets,model):
    yield
    for history,online in datasets:
        state=Detector(history,model)
        for value in online: yield state.update(value)
