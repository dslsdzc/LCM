"""Seven-axis cognitive evaluation for the V3 causal student."""
import argparse, json, pickle
import jax, jax.numpy as jnp
import numpy as np
import optax
from tokenizers import Tokenizer
from train.config import LCMConfig
from train.causal_student_train import z_state, local_data, PARAPHRASES
from train.fusion import gen_head_forward

COUNTERFACTUAL = [
 ("如果植物一直得到充足水分，它还会因为缺水萎蔫吗？", "不会"),
 ("如果水龙头没有关闭，水流会停止吗？", "不会"),
 ("如果气球停止充气，它还会因为继续膨胀而爆裂吗？", "不会"),
 ("如果手机重新接通电源，它会继续因为没电关机吗？", "不会"),
]
MULTIHOP = [
 ("雨水落到地面，地面变湿；随后气温降到零度附近，地面的水可能怎样？", "结冰"),
 ("手机电量耗尽会关机；给手机恢复电能后，它接下来可能怎样？", "开机"),
 ("纸靠近火焰会燃烧；燃烧会释放热量，所以周围温度会怎样？", "升高"),
]

def encode_questions(texts,tok,g2l,pad,n=32):
 out=[]
 for text in texts:
  ids=[g2l.get(i,pad) for i in tok.encode(text).ids[:n]]
  out.append(ids+[pad]*(n-len(ids)))
 return jnp.asarray(out,dtype=jnp.int32)

def generate(p,z,bos,eos,pad,gids,tok,n=24):
 cur=jnp.full((len(z),1),bos,dtype=jnp.int32); seq=[]
 for _ in range(n):
  x=jnp.pad(cur,((0,0),(0,n-cur.shape[1])),constant_values=pad)
  logits=gen_head_forward(p['gen_head'],z,x)
  nxt=jnp.argmax(logits[:,cur.shape[1]-1],-1); seq.append(np.asarray(nxt)); cur=jnp.concatenate([cur,nxt[:,None]],1)
 out=np.stack(seq,1); texts=[]
 for row in out:
  ids=row.tolist(); ids=ids[:ids.index(eos)] if eos in ids else ids
  texts.append(tok.decode([int(gids[i]) for i in ids if i!=pad]))
 return texts

def ce_for(p,z,x,y,m):
 logits=gen_head_forward(p['gen_head'],z,jnp.asarray(x)); ce=optax.softmax_cross_entropy_with_integer_labels(logits,jnp.asarray(y))
 return float((ce*jnp.asarray(m)).sum()/jnp.asarray(m).sum()), np.asarray(jnp.argmax(logits[:,0],-1))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True)
 ap.add_argument('--data',default='data/causal_dialogue.json'); ap.add_argument('--tokenizer',default='checkpoints/Qwen2.5-0.5B-Instruct/tokenizer.json'); args=ap.parse_args()
 tok=Tokenizer.from_file(args.tokenizer); gids=np.load(args.checkpoint+'/global_token_ids.npy'); g2l={int(g):i for i,g in enumerate(gids)}
 with open(args.checkpoint+'/student.pkl','rb') as f: ck=pickle.load(f)
 p=jax.tree_util.tree_map(jnp.asarray,ck['params']); cfg=LCMConfig(**ck['config'])
 rows,q,x,y,m,_,pad,bos,eos=local_data(args.data,tok,augment=True); z=z_state(p,jnp.asarray(q),cfg)
 normal,first=ce_for(p,z,x,y,m); zero,zfirst=ce_for(p,jnp.zeros_like(z),x,y,m); perm,pfirst=ce_for(p,jnp.roll(z,1,axis=0),x,y,m)
 target=np.asarray(y)[:,0]; result={'intervention':{'normal_ce':normal,'zero_ce':zero,'permuted_ce':perm,'normal_first_acc':float((first==target).mean()),'zero_first_acc':float((zfirst==target).mean()),'permuted_first_acc':float((pfirst==target).mean())}}
 # Linear readability probe: predict the answer's first token from z.
 Z=np.asarray(z); split=30; classes=sorted(set(target.tolist())); c2i={c:i for i,c in enumerate(classes)}; Y=np.eye(len(classes))[np.array([c2i[v] for v in target])]
 W=np.linalg.solve(Z[:split].T@Z[:split]+.1*np.eye(Z.shape[1]),Z[:split].T@Y[:split]); pred=np.argmax(Z[split:]@W,1)
 result['passive_probe']={'heldout_first_token_acc':float((pred==np.array([c2i[v] for v in target[split:]])).mean()),'note':'linear readout of latent state; checkpoint has no trained W_out'}
 cfq=[q for q,_ in COUNTERFACTUAL]; cfz=z_state(p,encode_questions(cfq,tok,g2l,pad),cfg); cfa=generate(p,cfz,bos,eos,pad,gids,tok)
 result['counterfactual']=[{'q':q,'answer':a,'expected':e,'pass':e in a} for (q,e),a in zip(COUNTERFACTUAL,cfa)]
 mhq=[q for q,_ in MULTIHOP]; mhz=z_state(p,encode_questions(mhq,tok,g2l,pad),cfg); mha=generate(p,mhz,bos,eos,pad,gids,tok)
 result['multihop']=[{'q':q,'answer':a,'expected':e,'pass':e in a} for (q,e),a in zip(MULTIHOP,mha)]
 os_path=args.checkpoint+'/cognitive_suite.json'
 with open(os_path,'w',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
