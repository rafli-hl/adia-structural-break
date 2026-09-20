"""Unchanged logistic objective with strictly nested E14 preprocessing."""
import json
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits, threadpool_info
from sbrt.logistic import training_weights, objective, SOLVER_OPTIONS, InvalidFit
from sbrt.stage3_features import standardize_extra
from stage4_research.features import Preprocessing as E14Preprocessing
from .contract import IDS, INPUTS, EXTRA


def extra_values(experiment,values):
    columns=EXTRA[experiment]
    if isinstance(values,dict): return np.array([values[k] for k in columns],dtype=np.float64)
    return np.asfortranarray(values.loc[:,list(columns)].to_numpy(dtype=np.float64))


class Preprocessing:
    def __init__(self,experiment,base,mean,std):
        if experiment not in IDS or base.experiment!="E14": raise ValueError("Wrong parent or experiment")
        self.experiment,self.base=experiment,base
        for key,value in (("mean",mean),("std",std)):
            a=np.array(value,dtype=np.float64,copy=True)
            if a.shape!=(len(EXTRA[experiment]),) or not np.isfinite(a).all() or (key=="std" and (a<0).any()):
                raise ValueError("Invalid extension preprocessing")
            a.setflags(write=False); setattr(self,key,a)

    @classmethod
    def fit(cls,experiment,frame,omega):
        base,x14=E14Preprocessing.fit("E14",frame,omega)
        # Each column uses the trusted columnwise reduction, independently of count.
        mean,std,x=standardize_extra(extra_values(experiment,frame),omega)
        return cls(experiment,base,mean,std),np.column_stack((x14,x))

    def transform(self,values):
        x=extra_values(self.experiment,values)
        if not np.isfinite(x).all(): raise ValueError("Nonfinite extension feature")
        scaled=np.divide(x-self.mean,self.std,out=np.zeros_like(x),where=self.std>0)
        return np.concatenate((self.base.transform(values),scaled),axis=-1)

    def as_dict(self):
        return dict(experiment=self.experiment,inputs=INPUTS[self.experiment],base=self.base.as_dict(),
                    mean=self.mean.tolist(),std=self.std.tolist())

    @classmethod
    def from_dict(cls,value):
        v=dict(value)
        if v.pop("inputs")!=INPUTS[v["experiment"]]: raise ValueError("Input allowlist drift")
        v["base"]=E14Preprocessing.from_dict(v["base"])
        return cls(**v)


class Model:
    def __init__(self,pre,intercept,beta):
        self.pre,self.intercept=pre,float(intercept)
        self.beta=np.array(beta,dtype=np.float64,copy=True)
        if self.beta.shape!=(len(INPUTS[pre.experiment]),) or not np.isfinite(self.beta).all() or not np.isfinite(self.intercept):
            raise ValueError("Invalid fitted parameters")
        self.beta.setflags(write=False)

    def predict(self,values):
        x=self.pre.transform(values)
        logits=self.intercept+x[...,0]*self.beta[0]
        for j in range(1,len(self.beta)): logits=logits+x[...,j]*self.beta[j]
        result=expit(logits)
        if not np.isfinite(result).all(): raise ValueError("Nonfinite prediction")
        return result

    def dumps(self):
        return json.dumps(dict(version=1,preprocessing=self.pre.as_dict(),intercept=self.intercept,beta=self.beta.tolist()),
            sort_keys=True,separators=(",",":"),allow_nan=False).encode()

    @classmethod
    def loads(cls,payload):
        v=json.loads(payload)
        if set(v)!={"version","preprocessing","intercept","beta"} or v["version"]!=1: raise ValueError("Invalid model schema")
        return cls(Preprocessing.from_dict(v["preprocessing"]),v["intercept"],v["beta"])


def fit_fold(frame,heldout_fold,experiment):
    start=time.perf_counter()
    train=frame.loc[frame.fold!=heldout_fold]
    with threadpool_limits(limits=1):
        eligible,omega,audit=training_weights(train.time_online.to_numpy(),train.target.to_numpy())
        train=train.loc[eligible]
        pre,x=Preprocessing.fit(experiment,train,omega)
        y=train.target.to_numpy(dtype=np.float64)
        result=minimize(objective,np.zeros(x.shape[1]+1,dtype=np.float64),args=(x,y,omega),
            method="L-BFGS-B",jac=True,bounds=None,options=SOLVER_OPTIONS)
        loss,gradient=objective(result.x,x,y,omega)
        detail=dict(fold=int(heldout_fold),iterations=int(result.nit),function_evaluations=int(result.nfev),
            objective=loss,gradient_infinity_norm=float(np.max(np.abs(gradient))),final_gradient=gradient.tolist(),
            optimizer_success=bool(result.success),optimizer_status=int(result.status),optimizer_message=str(result.message),
            native_threads=threadpool_info(),preprocessing=pre.as_dict(),intercept=float(result.x[0]),beta=result.x[1:].tolist(),
            options=SOLVER_OPTIONS,training_seconds=time.perf_counter()-start)
    if (not result.success or not np.isfinite(result.x).all() or not np.isfinite(loss) or not np.isfinite(gradient).all()
            or detail["gradient_infinity_norm"]>1e-6 or result.nit>1000): raise InvalidFit(detail)
    model=Model(pre,result.x[0],result.x[1:])
    detail["serialized_model_bytes"]=len(model.dumps())
    return model,detail,audit
