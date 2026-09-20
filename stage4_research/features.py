"""Causal Stage-4 streams and fold-only contrast standardization."""
import hashlib
import json
import numpy as np
from sbrt.history import residuals
from sbrt.stage2_detectors import Stage2Detector, _encode, _decode, _array_bytes
from sbrt.logistic import standardize_fit
from sbrt.stage3_features import standardize_extra
from .contract import BASE, INPUTS, STREAMS
from .scale import ScaleEvidence, lag_one


class FeatureState:
    def __init__(self, history, experiment):
        if experiment not in (*INPUTS, "export"):
            raise ValueError("Only frozen Stage-4 feature sets")
        self.experiment = experiment
        self.blocks = Stage2Detector(history, "E07")
        # Clone already-fitted AR state; reuse its exact update, no refit or new AR math.
        self.ar = _decode(_encode(self.blocks.ar))
        e = residuals(history, (self.ar.mu,self.ar.sd,self.ar.beta))
        r = (e[self.ar.cut:]-self.ar.ref_mu)/self.ar.ref_sd
        self.original_squared_acf = lag_one(r*r)
        kinds = ("slow", "fast", "robust") if experiment == "export" else STREAMS[experiment]
        self.scales = {k: ScaleEvidence(e,self.ar.cut,k,rank=k=="slow" and experiment in ("export","E17")) for k in kinds}
        self.last = {}
        self.diagnostic = {}
        self.pending = False

    def observe(self, value):
        if self.pending:
            raise ValueError("Must commit before consuming another point")
        self.blocks.update(value)
        e, _ = self.ar.update(value)
        row = {k:self.blocks.last_blocks[k] for k in BASE}
        self.diagnostic = dict(e_squared=e*e)
        for kind, scale in self.scales.items():
            result = scale.observe(e)
            suffix = {"slow":"s","fast":"f","robust":"r"}[kind]
            row["J_"+suffix] = result["J"]
            if result["P"] is not None:
                row["P_s"] = result["P"]
            self.diagnostic["v_"+suffix] = result["variance"]
        self.last, self.pending = row, True
        return row

    def commit(self):
        if not self.pending:
            raise ValueError("No pending point")
        for scale in self.scales.values():
            scale.commit()
        self.pending = False

    def dumps(self):
        fields = {k:v for k,v in self.__dict__.items() if k != "scales"}
        return json.dumps(dict(version=1, fields=_encode(fields),
            scales={k:v.state() for k,v in self.scales.items()}),sort_keys=True,separators=(",",":"),allow_nan=False).encode()

    @classmethod
    def loads(cls, payload):
        v = json.loads(payload)
        if set(v) != {"version","fields","scales"} or v["version"] != 1:
            raise ValueError("Invalid feature state schema")
        obj = cls.__new__(cls)
        obj.__dict__.update(_decode(v["fields"]))
        obj.scales = {k:ScaleEvidence.restore(s) for k,s in v["scales"].items()}
        return obj

    def array_bytes(self):
        return _array_bytes(self,set())


def contrasts(experiment, values):
    suffix = {"E14":"s","E15":"f","E16":"r","E17":"s"}[experiment]
    energy = (np.asarray(values["J_"+suffix])-np.asarray(values["I"]))/np.sqrt(2.0)
    columns = [energy]
    if experiment == "E17":
        columns.append((np.asarray(values["P_s"])-np.asarray(values["P"]))/np.sqrt(2.0))
    return np.stack(columns,axis=-1)


class Preprocessing:
    def __init__(self, experiment, mean, std, extra_mean, extra_std):
        if experiment not in INPUTS:
            raise ValueError("Unregistered experiment")
        self.experiment = experiment
        for key, values, length in (("mean",mean,4),("std",std,4),
                ("extra_mean",extra_mean,len(INPUTS[experiment])-4),("extra_std",extra_std,len(INPUTS[experiment])-4)):
            a = np.array(values,dtype=np.float64,copy=True)
            if a.shape != (length,) or not np.isfinite(a).all() or ("std" in key and (a<0).any()):
                raise ValueError("Invalid preprocessing parameters")
            a.setflags(write=False)
            setattr(self,key,a)

    @classmethod
    def fit(cls, experiment, frame, omega):
        x = np.asfortranarray(frame.loc[:,list(BASE)].to_numpy(dtype=np.float64))
        mean,std,base = standardize_fit(x,omega)
        extra = np.asfortranarray(contrasts(experiment,frame))
        em,es,scaled = standardize_extra(extra,omega)
        return cls(experiment,mean,std,em,es), np.column_stack((base,scaled))

    def transform(self, values):
        if isinstance(values,dict):
            x = np.array([values[k] for k in BASE],dtype=np.float64)
        else:
            x = values.loc[:,list(BASE)].to_numpy(dtype=np.float64)
        extra = contrasts(self.experiment,values)
        if not np.isfinite(x).all() or not np.isfinite(extra).all():
            raise ValueError("Nonfinite input")
        base = np.divide(x-self.mean,self.std,out=np.zeros_like(x),where=self.std>0)
        scaled = np.divide(extra-self.extra_mean,self.extra_std,out=np.zeros_like(extra),where=self.extra_std>0)
        return np.concatenate((base,scaled),axis=-1)

    def as_dict(self):
        return dict(experiment=self.experiment,inputs=INPUTS[self.experiment],
            **{k:getattr(self,k).tolist() for k in ("mean","std","extra_mean","extra_std")})

    @classmethod
    def from_dict(cls,value):
        value = dict(value)
        if value.pop("inputs") != INPUTS[value["experiment"]]:
            raise ValueError("Input allowlist changed")
        return cls(**value)


class Detector:
    def __init__(self, history, model):
        self.features = FeatureState(history,model.pre.experiment)
        self.model = model

    def update(self, value):
        row = self.features.observe(value)
        score = float(self.model.predict(row))
        self.features.commit()
        return score

    def dumps(self):
        return json.dumps(dict(version=1,model_sha256=hashlib.sha256(self.model.dumps()).hexdigest(),
            features=json.loads(self.features.dumps())),sort_keys=True,separators=(",",":"),allow_nan=False).encode()

    @classmethod
    def loads(cls,payload,model):
        value = json.loads(payload)
        if set(value)!={"version","model_sha256","features"} or value["version"]!=1 or value["model_sha256"]!=hashlib.sha256(model.dumps()).hexdigest():
            raise ValueError("State/model mismatch")
        obj = cls.__new__(cls)
        obj.model = model
        obj.features = FeatureState.loads(json.dumps(value["features"]))
        if obj.features.experiment != model.pre.experiment:
            raise ValueError("Wrong feature state")
        return obj


def infer(datasets,model):
    yield
    for history,online in datasets:
        state = Detector(history,model)
        for value in online:
            yield state.update(value)
