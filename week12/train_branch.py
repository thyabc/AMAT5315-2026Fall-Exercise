#!/usr/bin/env python3
"""JAX x64 imitation learning; the validation/test graphs never enter training."""
import os
os.environ["JAX_PLATFORMS"]="cpu"
import argparse
from collections import Counter
import json
import hashlib
import math
from pathlib import Path
import platform
import time
import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import numpy as np

FEATURES=["x_val","frac_dist","field_h","abs_field_h","deg_norm","bond_bias",
          "eff_field","abs_eff_field","mean_bond_dual","mean_bond_gap","mean_nbr_frac","gap_ratio"]

def independent_features(graph,node):
    """Independent Python recomputation from the recorded parent LP state."""
    x=np.clip(np.asarray(node["lp_x"],dtype=np.float64),0,1)
    s=1-2*x
    z=np.asarray(node["lp_z"],dtype=np.float64)
    dual=np.asarray(node["row_duals"],dtype=np.float64)
    fixed=node["fixed"]
    edges=graph["edges"]
    assert len(dual)==4*len(edges)
    incident=[[] for _ in range(graph["n"])]
    for k,e in enumerate(edges):
        incident[e["u"]].append((k,e["v"]))
        incident[e["v"]].append((k,e["u"]))
    dmax=max(1,max(map(len,incident)))
    ddual=max(1e-9,float(np.max(np.abs(dual),initial=0)))
    best=node["incumbent"]
    ratio=1.0 if best is None else min(2,max(0,(best-node["lp_bound"])/max(abs(best),1)))
    result=[]
    for i in node["candidates"]:
        degree=len(incident[i]); denominator=max(1,degree)
        bias=0.0; effective=float(graph["h"][i]); dual_sum=0.0; gap=0.0; frac=0.0
        for k,j in incident[i]:
            coupling=edges[k]["j"]
            bias+=float(coupling>0)-float(coupling<0)
            effective+=coupling*s[j]
            dual_sum+=sum(abs(float(v)) for v in dual[4*k:4*k+4])/ddual
            gap+=abs(z[k]-0.5*(1-s[i]*s[j]))
            if fixed[j]<0:
                frac+=2*min(x[j],1-x[j])
        effective/=degree+1
        result.append([x[i],2*min(x[i],1-x[i]),graph["h"][i],abs(graph["h"][i]),
            degree/dmax,bias/denominator,effective,abs(effective),dual_sum/denominator,
            gap/denominator,frac/denominator,ratio])
    return np.asarray(result,dtype=np.float64)

def checked_dataset(path):
    data=json.loads(Path(path).read_text())
    assert data["schema"]=="branchopt.labels.v1"
    assert data["feature_names"]==FEATURES
    assert data["source_validation_status"]=="passed"
    graphs={r["id"]:r["graph"] for r in data["instances"]}
    assert len(graphs)==len(data["instances"])
    counts=Counter(); seen=set(); maximum=0.0
    train=[]; validation=[]
    for node in data["nodes"]:
        instance=node["instance_id"]
        assert 0<=instance<48 and instance in graphs
        assert node["split"]==("train" if instance<36 else "validation")
        key=(instance,node["node"])
        assert key not in seen
        seen.add(key); counts[instance]+=1
        assert 2<=len(node["candidates"])<=graphs[instance]["n"]
        assert len(set(node["candidates"]))==len(node["candidates"])
        assert node["chosen_spin"] in node["candidates"]
        actual=np.asarray(node["features"],dtype=np.float64)
        expected=independent_features(graphs[instance],node)
        assert actual.shape==(len(node["candidates"]),12) and np.isfinite(actual).all()
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
        maximum=max(maximum,float(np.max(np.abs(actual-expected))))
        gains=np.asarray(node["gains"],dtype=np.float64)
        scores=np.asarray(node["scores"],dtype=np.float64)
        assert gains.shape==(len(actual),2) and scores.shape==(len(actual),)
        np.testing.assert_allclose(scores,np.maximum(gains[:,0],1e-6)*np.maximum(gains[:,1],1e-6),rtol=1e-12,atol=1e-12)
        (train if instance<36 else validation).append(node)
    assert train and len(validation)>=6,"Need training nodes and at least six validation nodes"
    assert max(counts.values())<=80
    train_ids=sorted({n["instance_id"] for n in train})
    val_ids=sorted({n["instance_id"] for n in validation})
    assert not set(train_ids)&set(val_ids)
    provenance={"parameters":data["parameters"],
        "graph_hashes":[hashlib.sha256(json.dumps(g,sort_keys=True,separators=(",",":")).encode()).hexdigest() for g in graphs.values()]}
    return train,validation,train_ids,val_ids,maximum,provenance

def padded_nodes(nodes,mean,std,width):
    x=np.zeros((len(nodes),width,12),dtype=np.float64)
    mask=np.zeros((len(nodes),width),dtype=bool)
    target=np.zeros((len(nodes),width),dtype=np.float64)
    for i,node in enumerate(nodes):
        raw=np.asarray(node["features"],dtype=np.float64)
        k=len(raw)
        x[i,:k]=(raw-mean)/np.maximum(std,1e-6)
        mask[i,:k]=True
        score=np.asarray(node["scores"],dtype=np.float64)
        best=float(score.max())
        tied=np.abs(score-best)<=1e-9*max(1,abs(best))
        target[i,:k]=tied/tied.sum()
    return jnp.asarray(x),jnp.asarray(mask),jnp.asarray(target)

def logits(params,x):
    return jnp.tanh(x@params["w1"]+params["b1"])@params["w2"]+params["b2"]

def cross_entropy(params,x,mask,target):
    # A finite mask value avoids 0*(-infinity) in padded targets and gradients.
    masked=jnp.where(mask,logits(params,x),-1e30)
    log_probability=jax.nn.log_softmax(masked,axis=-1)
    return -jnp.mean(jnp.sum(target*log_probability,axis=-1))

def objective(params,x,mask,target):
    return cross_entropy(params,x,mask,target)+2e-4*(jnp.sum(params["w1"]**2)+jnp.sum(params["w2"]**2))

@jax.jit
def train_step(params,m,v,count,x,mask,target):
    value,grad=jax.value_and_grad(objective)(params,x,mask,target)
    m=jax.tree_util.tree_map(lambda a,g:0.9*a+0.1*g,m,grad)
    v=jax.tree_util.tree_map(lambda a,g:0.999*a+0.001*g*g,v,grad)
    params=jax.tree_util.tree_map(lambda p,a,b:p-4e-3*(a/(1-0.9**count))/(jnp.sqrt(b/(1-0.999**count))+1e-8),params,m,v)
    return params,m,v,value

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",default="artifacts/bnb_strong/dataset.json")
    parser.add_argument("--out",default="artifacts/model.json")
    parser.add_argument("--epochs",type=int,default=120)
    args=parser.parse_args()
    assert args.epochs>0
    train,val,train_ids,val_ids,feature_error,provenance=checked_dataset(args.dataset)
    raw_train=np.concatenate([np.asarray(n["features"],dtype=np.float64) for n in train])
    mean=raw_train.mean(axis=0); std=raw_train.std(axis=0)
    width=max(len(n["candidates"]) for n in train+val)
    train_batch=padded_nodes(train,mean,std,width)
    val_batch=padded_nodes(val,mean,std,width)
    key1,key2=jax.random.split(jax.random.PRNGKey(2026))
    params={"w1":jax.random.normal(key1,(12,48),dtype=jnp.float64)*math.sqrt(2/60),
        "b1":jnp.zeros(48,dtype=jnp.float64),
        "w2":jax.random.normal(key2,(48,),dtype=jnp.float64)*math.sqrt(2/49),
        "b2":jnp.asarray(0.0,dtype=jnp.float64)}
    assert all(a.dtype==jnp.float64 for a in jax.tree_util.tree_leaves(params))
    assert jax.default_backend()=="cpu" and jax.config.x64_enabled
    m=jax.tree_util.tree_map(jnp.zeros_like,params)
    v=jax.tree_util.tree_map(jnp.zeros_like,params)
    ce=jax.jit(cross_entropy)
    rng=np.random.default_rng(2026)
    history=[]; best_params=None; best_ce=math.inf; best_epoch=None; step=0
    started=time.perf_counter()
    print(f"JAX {jax.__version__}; CPU float64; train nodes={len(train)}, val nodes={len(val)}",flush=True)
    print(f"Train instances={len(train_ids)}, val instances={len(val_ids)}, overlap=0; feature error={feature_error:.3e}",flush=True)
    for epoch in range(args.epochs+1):
        if epoch:
            permutation=rng.permutation(len(train))
            for start in range(0,len(train),64):
                indices=permutation[start:start+64]
                batch=tuple(a[indices] for a in train_batch)
                step+=1
                params,m,v,_=train_step(params,m,v,jnp.asarray(step,dtype=jnp.int32),*batch)
        train_ce=float(ce(params,*train_batch)); val_ce=float(ce(params,*val_batch))
        assert math.isfinite(train_ce) and math.isfinite(val_ce),"Non-finite training loss"
        history.append({"epoch":epoch,"train_ce":train_ce,"validation_ce":val_ce})
        if epoch==1 or (epoch>0 and epoch%5==0):
            if val_ce<best_ce:
                best_ce=val_ce; best_epoch=epoch
                best_params=jax.tree_util.tree_map(lambda a:np.asarray(a).copy(),params)
        if epoch in [0,1] or epoch%10==0:
            print(f"epoch {epoch:3d}: train CE={train_ce:.6f}; validation CE={val_ce:.6f}",flush=True)
    assert best_params is not None
    selected={k:jnp.asarray(v,dtype=jnp.float64) for k,v in best_params.items()}
    probes=[]
    for index in np.linspace(0,len(val)-1,6,dtype=int):
        node=val[int(index)]
        raw=np.asarray(node["features"],dtype=np.float64)
        scores=logits(selected,jnp.asarray((raw-mean)/np.maximum(std,1e-6)))
        probabilities=jax.nn.softmax(scores)
        probes.append({"instance_id":node["instance_id"],"node":node["node"],
            "candidates":node["candidates"],"features":raw.tolist(),
            "logits":np.asarray(scores).tolist(),"probabilities":np.asarray(probabilities).tolist()})
    masked=jnp.where(train_batch[1],logits(selected,train_batch[0]),-1e30)
    p=np.asarray(jax.nn.softmax(masked,axis=-1))
    pad_probability=float(np.max(p[~np.asarray(train_batch[1])],initial=0))
    assert pad_probability==0.0
    summary={"train_nodes":len(train),"validation_nodes":len(val),
        "train_instances":len(train_ids),"validation_instances":len(val_ids),
        "train_instance_ids":train_ids,"validation_instance_ids":val_ids,"instance_overlap":[],
        "best_epoch":best_epoch,"best_validation_ce":best_ce,"epochs":args.epochs,
        "batch_size":64,"learning_rate":4e-3,"weight_l2":2e-4,"biases_regularized":False,
        "seed":2026,"feature_check_max_abs_error":feature_error,
        "maximum_padded_probability":pad_probability,"training_wall_seconds":time.perf_counter()-started,
        "initialization":"Xavier-normal weights; zero biases; JAX PRNGKey(2026)",
        "normalization":"population mean/std over real candidate rows from training instances only",
        "checkpoint_rule":"lowest validation CE among epoch 1 and every fifth epoch"}
    output={"schema":"branchopt.mlp.v1","feature_names":FEATURES,
        "stats":{"mean":mean.tolist(),"std":std.tolist()},
        "weights":{k:np.asarray(v).tolist() for k,v in best_params.items()},
        "history":history,"training_summary":summary,"probes":probes,"data_provenance":provenance,
        "environment":{"jax":jax.__version__,"numpy":np.__version__,"platform":platform.platform(),
            "backend":jax.default_backend(),"x64":jax.config.x64_enabled}}
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2,allow_nan=False))
    print("Saved",path.resolve(),flush=True)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
