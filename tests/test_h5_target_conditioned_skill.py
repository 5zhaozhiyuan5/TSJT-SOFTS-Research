"""CPU-only H5 math tests; no GeomLoss/data/model/checkpoint required.

python -m unittest discover -s tests -p test_h5_target_conditioned_skill.py
"""
import unittest
import torch
from model.Meta_Models.h5_target_conditioned_skill import (
    source_band_capability,target_fit_demand,band_head_cosine,
    stratified_shuffle,h5_candidate_alphas
)


class H5ScoreTests(unittest.TestCase):
    def setup_data(self):
        n=40
        truth=torch.full((n,36),50.0)
        pred=truth.clone()
        history=torch.zeros(n,12,2)
        history[:,-1,0]=0.0
        for i in range(n):
            for band in range(3):
                pred[i,band*12:(band+1)*12] += (i%9+1)*(band+1)/2
        return pred,truth,history

    def test_city_scale_and_relative_skill(self):
        p,y,h=self.setup_data()
        c1,meta1=source_band_capability(p,y,h,[50.0],[10.0])
        c2,meta2=source_band_capability(p*2+10,y*2+10,h,[110.0],[20.0])
        self.assertTrue(torch.allclose(c1,c2,atol=1e-6))
        self.assertEqual(c1.shape,(40,3))
        self.assertTrue((c1>=0).all() and (c1<=1).all())
        for a,b in zip(meta1["normalized_model_band_mae"],
                       meta2["normalized_model_band_mae"]):
            self.assertAlmostEqual(a,b,places=6)

    def test_target_demand_and_strict_shapes(self):
        p=torch.zeros(2,3,36)
        y=torch.zeros_like(p)
        y[...,24:]=9
        y[...,12:24]=3
        d,meta=target_fit_demand(p,y,[3.0])
        self.assertEqual(d.shape,(3,))
        self.assertAlmostEqual(float(d.sum()),1.0,places=5)
        self.assertGreater(float(d[2]),float(d[1]))
        self.assertGreater(float(d[1]),float(d[0]))
        with self.assertRaises(ValueError):
            target_fit_demand(p[:0],y[:0],[3.0])

    def test_head_cosine_band_specificity_and_zero(self):
        sf=torch.tensor([[1.0,0.0],[-1.0,0.0]])
        tf=torch.tensor([[1.0,0.0]])
        sd=torch.ones(2,36)
        td=torch.ones(1,36)
        c=band_head_cosine(sf,sd,tf,td)
        self.assertTrue(torch.allclose(c,torch.tensor([[1.,1.,1.],[-1.,-1.,-1.]]),
                                       atol=1e-5))
        z=band_head_cosine(sf,sd,tf,torch.zeros_like(td))
        self.assertTrue(torch.equal(z,torch.zeros_like(z)))

    def test_permutes_within_f2_quartiles(self):
        n=40
        cap=torch.arange(n,dtype=torch.float32).unsqueeze(1).repeat(1,3)/n
        alpha=torch.softmax(torch.arange(n,dtype=torch.float32)/8,dim=0)
        z1=stratified_shuffle(cap,alpha,7)
        z2=stratified_shuffle(cap,alpha,7)
        self.assertTrue(torch.equal(z1,z2))
        self.assertFalse(torch.equal(z1,cap))
        for q in range(4):
            a=q*10;b=(q+1)*10
            self.assertTrue(torch.equal(torch.sort(z1[a:b,0]).values,
                                        torch.sort(cap[a:b,0]).values))

    def test_fixed_ablations_no_nans_mass_and_strength(self):
        n=40
        alpha=torch.softmax(torch.arange(n,dtype=torch.float32)/12,dim=0)
        cap=torch.zeros(n,3)
        cap[:,0]=torch.linspace(0.15,0.95,n)
        cap[:,1]=torch.linspace(0.95,0.15,n)
        cap[:,2]=0.4
        cosine=2*cap-1
        demand=torch.tensor([.7,.2,.1])
        prc=torch.linspace(0.1,0.9,n)
        c,meta=h5_candidate_alphas(alpha,prc,cap,cosine,demand,
                                   seed=52,strength=.5)
        self.assertEqual(len(c),6)
        for name,weight in c.items():
            self.assertEqual(weight.shape,(n,))
            self.assertTrue(torch.isfinite(weight).all(),name)
            self.assertTrue((weight>0).all(),name)
            self.assertAlmostEqual(float(weight.sum()),1.,places=5)
            ratio=(weight/alpha)
            self.assertLess(float(ratio.max()/ratio.min()),3.01)
            self.assertIn("ess_after",meta[name])
        self.assertFalse(torch.allclose(c["joint"],c["rotated_demand"]))
        self.assertFalse(torch.allclose(c["joint"],c["shuffled_skill"]))
        zero,_=h5_candidate_alphas(alpha,prc,cap,cosine,demand,
                                  seed=52,strength=0)
        for weight in zero.values():
            self.assertTrue(torch.equal(weight,alpha))

    def test_invalid_candidate_rejected(self):
        alpha=torch.ones(20)/20
        prc=torch.ones(20)*.5
        cap=torch.ones(20,3)*.5
        cos=torch.zeros(20,3)
        with self.assertRaises(ValueError):
            h5_candidate_alphas(alpha,prc,cap,cos,torch.tensor([1.,0.,1.]),1)
        with self.assertRaises(ValueError):
            h5_candidate_alphas(alpha,prc,cap,cos,torch.tensor([.3,.3,.4]),1,
                                strength=1.5)


if __name__=="__main__":
    unittest.main()
