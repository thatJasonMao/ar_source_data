"""Explicit spatial reward bases. Fit knots/scales on training observations only."""
import itertools
import numpy as np


class SpatialRewardBasis:
    def __init__(self, spec):
        self.spec=spec
        self.input_names=spec['input_names']
        self.names=spec['names']

    @classmethod
    def fit(cls, x, input_names, kind='linear'):
        if kind not in ['linear','spline','interaction','rbf32','rbf64']:raise ValueError(kind)
        # First five non-bias columns are the agreed geometric metrics.
        numeric=[input_names.index('scaled_'+n) for n in ['concavity','distance_variation','elongation','horizontal_size','next_decision_distance']]
        knots={str(i):np.unique(np.quantile(x[:,i],[.1,.3,.5,.7,.9])).tolist() for i in numeric} if kind!='linear' else {}
        spec=dict(version=1,kind=kind,input_names=list(input_names),numeric=numeric,knots=knots,
                  names=[],means=[],scales=[])
        if kind.startswith('rbf'):
            from sklearn.cluster import MiniBatchKMeans
            count=int(kind[3:]);z=np.asarray(x[:,numeric],np.float32)
            centers=MiniBatchKMeans(n_clusters=count,random_state=20260926,n_init=3,batch_size=2048).fit(z).cluster_centers_
            d2=((centers[:,None,:]-centers[None,:,:])**2).sum(2)
            np.fill_diagonal(d2,np.inf)
            width=max(.25,float(np.median(np.sqrt(np.sort(d2,axis=1)[:,:3]))))
            spec.update(rbf_centers=centers.tolist(),rbf_width=width)
        obj=cls(spec);raw,names=obj.raw(x)
        means=raw.mean(0);scales=raw.std(0);scales[scales<.05]=1.
        # Preserve the intercept; all other columns are centered/scaled.
        means[0]=0.;scales[0]=1.
        spec.update(names=names,means=means.tolist(),scales=scales.tolist())
        return cls(spec)

    def raw(self,x):
        x=np.asarray(x,np.float32)
        if x.ndim!=2 or x.shape[1]!=len(self.input_names):raise ValueError('Spatial basis schema mismatch')
        cols=[x[:,i] for i in range(x.shape[1])];names=list(self.input_names)
        for key,knots in self.spec['knots'].items():
            i=int(key)
            for knot in knots:
                cols.append(np.maximum(x[:,i]-knot,0));names.append(f'hinge({self.input_names[i]}>{knot:.6g})')
        if self.spec['kind']=='interaction':
            numeric=self.spec['numeric']
            for i,j in itertools.combinations(numeric,2):
                cols.append(x[:,i]*x[:,j]);names.append(self.input_names[i]+' * '+self.input_names[j])
            contexts=[i for i,n in enumerate(self.input_names) if n=='high_smoke' or n.startswith(('floor_','stage_'))]
            for i,j in itertools.product(numeric,contexts):
                cols.append(x[:,i]*x[:,j]);names.append(self.input_names[i]+' * '+self.input_names[j])
        if self.spec['kind'].startswith('rbf'):
            z=x[:,self.spec['numeric']];centers=np.asarray(self.spec['rbf_centers'],np.float32)
            for j,center in enumerate(centers):
                cols.append(np.exp(-((z-center)**2).sum(1)/(2*self.spec['rbf_width']**2)))
                names.append(f'spatial_rbf_{j:02d}')
        return np.column_stack(cols).astype(np.float32),names

    def transform(self,x):
        raw,names=self.raw(x)
        if names!=self.names:raise ValueError('Reward basis names differ')
        return ((raw-np.asarray(self.spec['means'],np.float32))/np.asarray(self.spec['scales'],np.float32)).astype(np.float32)
