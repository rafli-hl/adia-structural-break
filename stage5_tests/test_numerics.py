"""Preregistered arithmetic on explicitly synthetic data, seed 20260925."""
import math
import unittest
import numpy as np
from scipy.special import ndtri
from sbrt.history import EmpiricalCDF, ARChannel, residuals
from stage4_research.features import FeatureState as E14State
from stage5_research.features import FeatureState, Statistics, historical_u, hermite, directional, hac, bounded, aggregate
from stage5_research.contract import WINDOWS, BASE, RAW, MOMENTS


def fixture():
    rng=np.random.default_rng(20260925)
    h=rng.normal(size=601)
    for j in range(1,len(h)): h[j]+=.45*h[j-1]
    return h,rng.normal(size=630)


class Numerics(unittest.TestCase):
    def test_pit_ties_endpoints_inclusive_finite_and_frozen(self):
        a=np.array([-2.,0.,0.,3.]); c=EmpiricalCDF(a)
        np.testing.assert_array_equal(c.transform(a),[.2,.5,.5,.8])
        np.testing.assert_array_equal(c.transform([-1e200,1e200]),[.1,.9])
        z=ndtri(c.transform(a)); self.assertTrue(np.isfinite(z).all())
        self.assertEqual(z[1],z[2]); self.assertEqual(z[1],0)
        before=c.reference.tobytes(); c.transform(123.)
        self.assertEqual(before,c.reference.tobytes()); self.assertFalse(c.reference.flags.writeable)

    def test_exact_E14_units_history_split_scale_and_boundary(self):
        h,o=fixture(); s=FeatureState(h,"export"); e14=E14State(h,"E14"); ar=ARChannel(h)
        np.testing.assert_array_equal(s.base.ar.beta,ar.beta)
        self.assertEqual(s.base.ar.cut,int(.7*len(h)))
        e=residuals(h,(ar.mu,ar.sd,ar.beta)); b=max(float(np.mean(e[5:ar.cut]**2)),1e-16)
        vmin=max(1e-16,1e-4*b); state=b; reference=[]
        for j in range(5,len(h)):
            if j>=ar.cut: reference.append(float(e[j])/math.sqrt(max(state,vmin)))
            state=.99*state+.01*(float(e[j])*float(e[j]))
        np.testing.assert_array_equal(historical_u(h,s.base),reference)
        self.assertEqual(s.base.scales["slow"].h,state)
        coefficients=s.base.ar.beta.tobytes()
        for value in o[:35]:
            expected_e,_=ar.update(value); before=s.base.scales["slow"].h
            row=s.observe(value); expected=e14.observe(value)
            self.assertEqual({k:row[k] for k in BASE},expected)
            self.assertEqual(s.base.scales["slow"].last["u"],expected_e/math.sqrt(max(before,vmin)))
            self.assertEqual(s.base.scales["slow"].h,before)
            s.commit(); e14.commit()
            self.assertEqual(s.base.scales["slow"].h,e14.scales["slow"].h)
        self.assertEqual(coefficients,s.base.ar.beta.tobytes())

    def test_current_residual_denominator_isolation(self):
        h,_=fixture(); a=FeatureState(h,"E18"); b=FeatureState(h,"E18")
        a.observe(1.); b.observe(100.)
        self.assertEqual(a.base.scales["slow"].last["variance"],b.base.scales["slow"].last["variance"])
        self.assertNotEqual(a.base.scales["slow"].last["u"],b.base.scales["slow"].last["u"])
        with self.assertRaises(ValueError): a.observe(2.)
        a.commit()
        with self.assertRaises(ValueError): a.commit()

    def test_hermite_formulas_and_reference_normalization(self):
        z=np.array([-2.,-.5,0.,1.,3.])
        a=hermite(z)
        for got,want in zip(a,(z,(z*z-1)/np.sqrt(2),(z**3-3*z)/np.sqrt(6),(z**4-6*z*z+3)/np.sqrt(24))):
            np.testing.assert_allclose(got,want,rtol=1e-10,atol=1e-10)
        s=Statistics(z,"E18"); channels=list(a)+[(z[1:]-z.mean())*(z[:-1]-z.mean())]
        for j,ref in enumerate(channels):
            self.assertEqual(s.mu[j],float(ref.mean())); self.assertEqual(s.sd[j],max(float(ref.std(ddof=1)),1e-8))
        expected=aggregate([bounded(abs((float(a[0][0])-s.mu[0])/s.sd[0]))]*4)
        self.assertEqual(s.update(float(z[0]))["M1"],expected)

    def test_all_moment_windows_prefix_internal_pairs_direct(self):
        rng=np.random.default_rng(20260925); ref=rng.normal(size=303); z=rng.normal(size=630)
        s=Statistics(ref,"E18"); history=[]
        for t,value in enumerate(z,1):
            row=s.update(float(value)); history.append(float(value)); direct=[[],[],[],[],[]]
            for w in (*WINDOWS,t):
                a=np.array(history[-min(t,w):]); ch=list(hermite(a))+[(a[1:]-s.mz)*(a[:-1]-s.mz)]
                for j,values in enumerate(ch):
                    d=(values-s.mu[j])/s.sd[j]
                    direct[j].append(bounded(math.sqrt(len(d))*abs(float(d.mean()))) if len(d)>=(8 if j==4 else 1) else 0.)
            np.testing.assert_allclose([row[k] for k in MOMENTS],[aggregate(v) for v in direct],atol=1e-10,rtol=1e-10)
            self.assertEqual(s.moment_banks[4]["n"],t-1)
            for b,w in zip(s.moment_banks[4]["windows"],WINDOWS): self.assertEqual(b.n,min(t-1,w-1))
        self.assertEqual(s.counts["pair_insufficient"],[8]*4)
        self.assertEqual(s.counts["moment_insufficient"],[0]*4)

    def test_likelihood_formula_direction_near_one_and_zero(self):
        for r in (0.,1e-20,.49,.5,.9,1.,1.+1e-10,1.5,1.51,4.):
            up,down,*rest=directional(128,r,1.)
            if r==0: self.assertEqual((up,down),(0.,1.)); self.assertEqual(rest[1],1)
            else:
                d=r-1; k=d-math.log1p(d) if abs(d)<=.5 else r-1-math.log(r)
                expected=bounded(64*max(k,0.))
                self.assertEqual(up,expected if r>1 else 0.)
                self.assertEqual(down,expected if r<1 else 0.)
            self.assertTrue(np.isfinite([up,down]).all())
        self.assertEqual(directional(8,-1e-13,1.)[:4],(0.,1.,1,1))
        with self.assertRaises(ValueError): directional(8,-1e-5,1.)

    def test_likelihood_rolling_partial_support_and_direct(self):
        rng=np.random.default_rng(20260925); ref=rng.normal(size=300); z=rng.normal(size=630)
        s=Statistics(ref,"E19")
        for t,v in enumerate(z,1):
            row=s.update(float(v)); up=[]; down=[]
            for w in (*WINDOWS,t):
                a=z[max(0,t-w):t]
                if len(a)<8: p=m=0.
                else: p,m,*_=directional(len(a),float(np.mean((a-s.mz)**2)),s.v0)
                up.append(p); down.append(m)
            np.testing.assert_allclose([row["L_plus"],row["L_minus"]],[aggregate(up),aggregate(down)],atol=1e-10,rtol=1e-10)
        self.assertEqual(s.counts["likelihood_insufficient"],[7]*4)

    def test_online_only_trajectory_exact_recursion(self):
        rng=np.random.default_rng(20260925); ref=rng.normal(size=80); s=Statistics(ref,"E20")
        self.assertEqual((s.peak,s.trajectory),(0.,0.)); peak=ewma=0.
        for v in rng.normal(size=60):
            row=s.update(float(v)); current=max(row["L_plus"],row["L_minus"])
            peak=max(peak,current); ewma=.99*ewma+.01*current
            self.assertEqual(row["drawdown"],peak-current); self.assertEqual(row["trajectory_ewma"],ewma)
        h,_=fixture(); state=FeatureState(h,"E20")
        self.assertEqual((state.statistics.peak,state.statistics.trajectory),(0.,0.))

    def test_hac_population_denominator_taper_and_support(self):
        rng=np.random.default_rng(20260925); q=rng.normal(size=101)**2; v=float(q.mean()); c=q-v
        got=hac(q,v)
        gamma=[sum(c[j]*c[j-l] for j in range(l,len(c)))/len(c) for l in range(17)]
        V=gamma[0]+2*sum((1-l/17)*gamma[l] for l in range(1,17))
        np.testing.assert_allclose(got["covariances"],gamma,atol=1e-10,rtol=1e-10)
        self.assertAlmostEqual(got["V"],V,places=12)
        self.assertEqual(got["eta"],max(1.,got["V"]/(2*v*v)))
        self.assertEqual(hac(q[:33],v)["eta"],1.); self.assertFalse(hac(q[:33],v)["supported"])
        self.assertTrue(hac(q[:34],v)["supported"])

    def test_eta_one_reproduces_E19_and_deflation_precedes_map(self):
        ref=np.linspace(-1.,1.,32); a=Statistics(ref,"E19"); b=Statistics(ref,"E21")
        self.assertEqual(b.calibration["eta"],1.)
        for value in np.linspace(-3.,3.,75):
            x=a.update(float(value)); y=b.update(float(value))
            self.assertEqual(x["L_plus"],y["L_plus_calibrated"])
            self.assertEqual(x["L_minus"],y["L_minus_calibrated"])
        raw=directional(32,2.,1.)[0]; calibrated=directional(32,2.,1.,3.)[0]
        self.assertAlmostEqual(calibrated,bounded(16*(1-math.log(2))/3),places=15)
        self.assertNotEqual(calibrated,raw/3)
        self.assertEqual(directional(8,0,1,7)[:2],(0.,1.))

    def test_degeneracy_floors_and_no_infinite_stored_feature(self):
        s=Statistics(np.zeros(40),"export")
        self.assertEqual(s.counts["moment_sd_floor"],5)
        self.assertTrue(s.calibration["disabled"])
        for _ in range(15):
            row=s.update(0.); self.assertTrue(np.isfinite(list(row.values())).all())
            self.assertEqual(row["L_plus"],0.); self.assertEqual(row["L_minus_calibrated"],0.)
        self.assertEqual(s.counts["likelihood_disabled"],[15]*4)

    def test_export_candidate_feature_equivalence_and_cdf_no_mutation(self):
        h,o=fixture(); export=FeatureState(h,"export"); states={k:FeatureState(h,k) for k in ("E18","E19","E20","E21")}
        frozen=export.cdf.reference.tobytes()
        for value in o[:50]:
            row=export.observe(value); export.commit()
            for state in states.values():
                other=state.observe(value); state.commit()
                self.assertEqual(other,{k:row[k] for k in other})
        self.assertEqual(frozen,export.cdf.reference.tobytes())
        self.assertFalse(export.cdf.reference.flags.writeable)

    def test_exact_state_serialization_pending_and_resumption(self):
        h,o=fixture()
        for name in ("E18","E19","E20","E21","export"):
            a=FeatureState(h,name)
            for x in o[:37]: a.observe(x); a.commit()
            before=a.dumps(); b=FeatureState.loads(before); self.assertEqual(before,b.dumps())
            for x in o[37:80]:
                self.assertEqual(a.observe(x),b.observe(x)); a.commit(); b.commit()
            self.assertEqual(a.dumps(),b.dumps())
            a.observe(o[81]); b=FeatureState.loads(a.dumps()); a.commit(); b.commit()
            self.assertEqual(a.dumps(),b.dumps())

    def test_null_replay_is_reset_without_warming_online_state(self):
        h,_=fixture(); s=FeatureState(h,"export",null_audit=True)
        self.assertEqual(s.statistics.age,0); self.assertIsNone(s.statistics.previous)
        self.assertEqual((s.statistics.peak,s.statistics.trajectory),(0.,0.))
        self.assertEqual(len(s.null_rows),len(h)-int(.7*len(h)))
        self.assertEqual(s.null_rows[0]["M5"],0.)
        self.assertEqual(s.null_rows[0]["L_minus"],0.)
