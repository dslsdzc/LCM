"""Fine-tune V3 for counterfactual/composition and memory-update tests."""
import argparse, dataclasses, json, os, pickle
import jax, jax.numpy as jnp
import numpy as np
import optax
from tokenizers import Tokenizer
from train.config import LCMConfig
from train.causal_student_train import z_state
from train.fusion import gen_head_forward

CF_TRAIN=[
 ("植物得到充足水分时，还会因缺水萎蔫吗？","不会，因为植物已经获得了充足水分。"),
 ("水龙头没有关闭时，水流会停止吗？","不会，水会继续流出。"),
 ("停止给气球充气后，它还会因持续充气而爆裂吗？","不会，因为充气已经停止。"),
 ("手机重新接通电源后，还会因没有电能而关机吗？","不会，因为手机已经重新获得电能。"),
 ("纸没有达到燃点时会燃烧吗？","不会，因为纸还没有达到燃点。"),
 ("没有雨水落到地面时，地面会因下雨变湿吗？","不会，因为没有雨水落到地面。"),
]
CF_TEST=[
 ("如果植物一直得到充足水分，它还会因为缺水萎蔫吗？","不会"),
 ("如果水龙头没有关闭，水流会停止吗？","不会"),
 ("如果气球停止充气，它还会因为继续膨胀而爆裂吗？","不会"),
 ("如果手机重新接通电源，它会继续因为没电关机吗？","不会"),
]
EDGE_TRAIN=[
 ("雨水落到地面后会怎样？","地面会变湿。"), ("地面的水降到零度附近会怎样？","地面的水会结冰。"),
 ("纸达到燃点后会怎样？","纸会燃烧。"), ("燃烧释放热量后周围温度会怎样？","周围温度会升高。"),
 ("剧烈运动后身体需要什么？","身体需要更多氧气。"), ("身体需要更多氧气时呼吸会怎样？","呼吸会加快。"),
]
CHAIN_TEST=[
 ("雨水落到地面，随后气温降到零度附近，地面的水最终怎样？","结冰"),
 ("纸达到燃点并燃烧释放热量后，周围温度最终怎样？","升高"),
 ("剧烈运动使身体需要更多氧气，人的呼吸随后怎样？","加快"),
]
MULTI_TRAIN=[
 ("雨水使地面变湿，接着气温降到零度附近，地面的水最终会怎样？","地面的水最终会结冰。"),
 ("纸达到燃点后燃烧并释放热量，周围温度最终会怎样？","周围温度最终会升高。"),
 ("剧烈运动使身体需要更多氧气，因此人的呼吸最终会怎样？","人的呼吸最终会加快。"),
]
MULTI_TEST=[
 ("先下雨让路面湿润，再突然降温到冰点，路面的水最后怎样？","结冰"),
 ("纸先被点燃，燃烧又释放热量，附近温度接下来如何？","升高"),
 ("跑步令身体需氧量上升，这又会让呼吸发生什么变化？","加快"),
]

def load_expand(path,tok,text_rows):
 with open(path+'/student.pkl','rb') as f: ck=pickle.load(f)
 p=jax.tree_util.tree_map(jnp.asarray,ck['params']); old=np.load(path+'/global_token_ids.npy').tolist()
 ids=set(old)
 for q,a in text_rows: ids.update(tok.encode(q).ids); ids.update(tok.encode(a).ids)
 ids.update([tok.token_to_id('<|endoftext|>'),tok.token_to_id('<|im_start|>'),tok.token_to_id('<|im_end|>')])
 gids=np.asarray(sorted(ids),dtype=np.int32); oldmap={g:i for i,g in enumerate(old)}; newmap={int(g):i for i,g in enumerate(gids)}
 key=jax.random.PRNGKey(81); d=p['gen_head']['w_embed'].shape[1]
 ee=jax.random.normal(key,(len(gids),d))*d**-0.5
 ge=jax.random.normal(jax.random.PRNGKey(82),(len(gids),d))*d**-0.5
 w3=jax.random.normal(jax.random.PRNGKey(83),(p['gen_head']['w_3'].shape[0],len(gids)))*p['gen_head']['w_3'].shape[0]**-0.5
 oi=np.asarray([oldmap[g] for g in old]); ni=np.asarray([newmap[g] for g in old])
 ee=ee.at[ni].set(p['encoder']['embed'][oi]); ge=ge.at[ni].set(p['gen_head']['w_embed'][oi]); w3=w3.at[:,ni].set(p['gen_head']['w_3'][:,oi])
 p['encoder']['embed']=ee; p['gen_head']['w_embed']=ge; p['gen_head']['w_3']=w3
 cfg=LCMConfig(**{**ck['config'],'vocab_size':len(gids)}); return p,cfg,gids,newmap

def encode(rows,tok,g2l,nq=32,na=24):
 pad=g2l[tok.token_to_id('<|endoftext|>')]; bos=g2l[tok.token_to_id('<|im_start|>')]; eos=g2l[tok.token_to_id('<|im_end|>')]
 q=[];x=[];y=[];m=[]
 for a,b in rows:
  qi=[g2l[i] for i in tok.encode(a).ids[:nq]]; ai=[g2l[i] for i in tok.encode(b).ids[:na-1]]+[eos]
  q.append(qi+[pad]*(nq-len(qi))); xx=[bos]+ai[:-1]; x.append(xx+[pad]*(na-len(xx))); y.append(ai+[pad]*(na-len(ai))); m.append([1.]*len(ai)+[0.]*(na-len(ai)))
 return tuple(jnp.asarray(v) for v in (q,x,y,m)),pad,bos,eos

def generate(p,cfg,rows,tok,gids,g2l,pad,bos,eos):
 (q,_,_,_),_,_,_=encode([(x,"。") for x,_ in rows],tok,g2l); z=z_state(p,q,cfg); cur=jnp.full((len(rows),1),bos,dtype=jnp.int32); out=[]
 for _ in range(24):
  xx=jnp.pad(cur,((0,0),(0,24-cur.shape[1])),constant_values=pad); nxt=jnp.argmax(gen_head_forward(p['gen_head'],z,xx)[:,cur.shape[1]-1],-1); out.append(np.asarray(nxt)); cur=jnp.concatenate([cur,nxt[:,None]],1)
 texts=[]
 for s in np.stack(out,1):
  a=s.tolist(); a=a[:a.index(eos)] if eos in a else a; texts.append(tok.decode([int(gids[i]) for i in a if i!=pad]))
 return texts

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['counterchain','multihop','memory'],required=True); ap.add_argument('--steps',type=int,default=400)
 ap.add_argument('--base',default='checkpoints/causal_student_v3_aug_s800'); ap.add_argument('--data',default='data/causal_dialogue.json'); ap.add_argument('--tokenizer',default='checkpoints/Qwen2.5-0.5B-Instruct/tokenizer.json'); ap.add_argument('--output',required=True)
 ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--replay-ratio',type=float,default=.25)
 ap.add_argument('--distill-z',type=float,default=.1); ap.add_argument('--distill-logits',type=float,default=.2)
 ap.add_argument('--temperature',type=float,default=2.); ap.add_argument('--encoder-scale',type=float,default=.1)
 ap.add_argument('--lattice-scale',type=float,default=.1); args=ap.parse_args()
 tok=Tokenizer.from_file(args.tokenizer); base=json.load(open(args.data,encoding='utf-8')); replay=[(r['user'],r['answer']) for r in base]
 if args.mode=='counterchain': special=CF_TRAIN+EDGE_TRAIN; tests=CF_TEST+CHAIN_TEST
 elif args.mode=='multihop': special=MULTI_TRAIN; tests=MULTI_TEST
 else: special=[(base[0]['user'],base[11]['answer'])]; tests=[(base[0]['user'],'手机')]+[(r['user'],r['answer'][:2]) for r in base[1:]]
 p,cfg,gids,g2l=load_expand(args.base,tok,replay+special+tests)
 # The teacher is the pre-specialization student, not Qwen.  It is kept only
 # during training and is deliberately absent from the saved checkpoint.
 teacher=jax.tree_util.tree_map(jax.lax.stop_gradient,p)
 (NQ,NX,NY,NM),pad,bos,eos=encode(special,tok,g2l)
 (RQ,RX,RY,RM),_,_,_=encode(replay,tok,g2l)
 baseline=generate(p,cfg,replay,tok,gids,g2l,pad,bos,eos)
 opt=optax.adamw(5e-4); state=opt.init(p)

 def scale_grads(g):
  """Slow shared cognition while leaving the active language head plastic."""
  lattice={'hrq','sparse','lowrank','manifold','binding','contrast'}
  out={}
  for name,value in g.items():
   scale=args.encoder_scale if name=='encoder' else (args.lattice_scale if name in lattice else 1.)
   out[name]=jax.tree_util.tree_map(lambda x:x*scale,value)
  return out

 @jax.jit
 def step(p,state,nq,nx,ny,nm,rq,rx,ry,rm):
  def loss(pp):
   nz=z_state(pp,nq,cfg); nl=gen_head_forward(pp['gen_head'],nz,nx)
   nce=optax.softmax_cross_entropy_with_integer_labels(nl,ny); nce=(nce*nm).sum()/nm.sum()
   rz=z_state(pp,rq,cfg); rl=gen_head_forward(pp['gen_head'],rz,rx)
   rce=optax.softmax_cross_entropy_with_integer_labels(rl,ry); rce=(rce*rm).sum()/rm.sum()
   old_z=z_state(teacher,rq,cfg); old_l=gen_head_forward(teacher['gen_head'],old_z,rx)
   zloss=jnp.mean((rz-old_z)**2)
   temp=args.temperature; old_prob=jax.nn.softmax(old_l/temp,-1)
   kd=-(old_prob*jax.nn.log_softmax(rl/temp,-1)*rm[...,None]).sum()/rm.sum()*(temp**2)
   total=nce+args.replay_ratio*rce+args.distill_z*zloss+args.distill_logits*kd
   return total,(nce,rce,zloss,kd)
  (loss,parts),g=jax.value_and_grad(loss,has_aux=True)(p); g=scale_grads(g)
  up,state=opt.update(g,state,p); return optax.apply_updates(p,up),state,loss,parts
 rng=np.random.default_rng(9)
 for i in range(args.steps):
  ni=rng.integers(0,len(special),args.batch_size); ri=rng.integers(0,len(replay),max(1,round(args.batch_size*args.replay_ratio)))
  p,state,loss,parts=step(p,state,NQ[ni],NX[ni],NY[ni],NM[ni],RQ[ri],RX[ri],RY[ri],RM[ri])
  if i%100==0 or i+1==args.steps: print(i+1,float(loss),'new/replay/z/kd',*[float(x) for x in parts],flush=True)
 answers=generate(p,cfg,tests,tok,gids,g2l,pad,bos,eos); result=[{'prompt':q,'expected':e,'answer':a,'pass':e in a} for (q,e),a in zip(tests,answers)]
 retained=generate(p,cfg,replay,tok,gids,g2l,pad,bos,eos)
 retention=[{'prompt':q,'before':before,'after':after,'exact_preserved':before==after} for (q,_),before,after in zip(replay,baseline,retained)]
 os.makedirs(args.output,exist_ok=True); np.save(args.output+'/global_token_ids.npy',gids)
 with open(args.output+'/student.pkl','wb') as f: pickle.dump({'params':jax.tree_util.tree_map(np.asarray,p),'config':dataclasses.asdict(cfg),'step':args.steps},f)
 with open(args.output+'/results.json','w',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
 with open(args.output+'/retention.json','w',encoding='utf-8') as f: json.dump(retention,f,ensure_ascii=False,indent=2)
 print('exact retention',sum(r['exact_preserved'] for r in retention),'/',len(retention))
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
