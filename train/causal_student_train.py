"""Small causal-dialogue experiment with a removable Qwen teacher.

Qwen is used only to build soft language targets.  Gradients update the LCM
encoder/lattices and its active generation head.  The saved checkpoint contains
no Qwen tensors, so evaluation is a genuine student-only inference path.
"""
import argparse, dataclasses, json, os, pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tokenizers import Tokenizer

from train.config import LCMConfig
from train.encoder import init_encoder_params, encoder_forward
from train.lattices import (init_hrq_params, init_sparse_params,
    init_lowrank_params, init_manifold_params, init_binding_params,
    init_contrast_params)
from train.cog_loop import cog_loop_scan
from train.cog_train import pack_codebooks_for_c
from train.fusion import init_gen_head_params, gen_head_forward
from train.qwen_lm import load_qwen_params, qwen_forward

PARAPHRASES = [
 "小狗一般会发出什么声音？", "气温降到冰点时水会有什么变化？",
 "一场雨过后，路面为什么湿了？", "植物长期缺水会有什么结果？",
 "接通台灯开关以后会发生什么？", "冰块放到暖和的屋里为何变成水？",
 "玻璃杯摔到水泥地上可能有什么后果？", "剧烈跑步后人为什么呼吸急促？",
 "把水龙头拧紧以后为什么不流水？", "盐放进水里充分搅拌有什么变化？",
 "乌云越来越多通常预示什么？", "手机没有剩余电量会发生什么？",
 "光源移动时影子为什么也会移动？", "纸张接触火苗可能造成什么？",
 "没睡够为什么容易走神？", "排水口被堵住还一直放水会怎样？",
 "不断往气球里打气最终可能怎样？", "金属汤匙泡在热汤里为何发烫？",
 "开窗后屋里的烟通常会如何变化？", "反复复习为何能让记忆更牢？"
]


def local_data(path, tok, max_q=32, max_a=24, augment=False):
    rows=json.load(open(path, encoding='utf-8'))
    if augment:
        rows=rows+[{"user":q,"answer":r["answer"]} for q,r in zip(PARAPHRASES,rows)]
    eos_id=tok.token_to_id('<|im_end|>') or 151645
    encoded=[]; vocab={tok.token_to_id('<|endoftext|>') or 0,
                       tok.token_to_id('<|im_start|>') or 151644, eos_id}
    for r in rows:
        q=tok.encode(r['user']).ids[:max_q]
        a=tok.encode(r['answer']).ids[:max_a-1]+[eos_id]
        vocab.update(q); vocab.update(a); encoded.append((r,q,a))
    gids=sorted(vocab); g2l={g:i for i,g in enumerate(gids)}
    pad=g2l[tok.token_to_id('<|endoftext|>') or 0]
    bos=g2l[tok.token_to_id('<|im_start|>') or 151644]
    def arr(ids,n): return [g2l[x] for x in ids]+[pad]*(n-len(ids))
    qs=[]; xs=[]; ys=[]; masks=[]
    for _,q,a in encoded:
        qs.append(arr(q,max_q)); inp=[bos]+[g2l[x] for x in a[:-1]]
        xs.append(inp+[pad]*(max_a-len(inp)))
        yy=[g2l[x] for x in a]; ys.append(yy+[pad]*(max_a-len(yy)))
        masks.append([1.]*len(yy)+[0.]*(max_a-len(yy)))
    return rows, np.int32(qs),np.int32(xs),np.int32(ys),np.float32(masks),np.int32(gids),pad,bos,g2l[eos_id]


def init_student(cfg, key, vocab):
    k=jax.random.split(key,9); d=cfg.d_model
    return {
      'encoder':init_encoder_params(k[0],d,cfg.d_ff,cfg.n_heads,cfg.n_encoder_layers,vocab,cfg.max_seq_len),
      'hrq':init_hrq_params(k[1],d,cfg.M_top,cfg.M_fine,cfg.n_hrq_layers),
      'sparse':init_sparse_params(k[2],d,cfg.M_sparse),
      'lowrank':init_lowrank_params(k[3],d,cfg.M_lr,cfg.ranks),
      'manifold':init_manifold_params(k[4],d,cfg.M_man,cfg.t_dim),
      'binding':init_binding_params(k[5],d,cfg.M_bind,cfg.n_bind_layers,cfg.r_max),
      'contrast':init_contrast_params(k[6],d,cfg.M_contrast,cfg.n_contrast_layers),
      'gen_head':init_gen_head_params(k[7],d,vocab)}


def z_state(p,q,cfg,mode='residual'):
    z=encoder_forward(p['encoder'],q,cfg.n_heads)
    cbs=pack_codebooks_for_c(p)
    cn=jnp.sqrt(sum(jnp.mean(jnp.sum(c*c,axis=-1)) for c in cbs)/len(cbs))
    z=z*(cn/(jnp.sqrt(jnp.mean(jnp.sum(z*z,axis=-1)))+1e-8))
    fn=lambda x: cog_loop_scan(x,cbs,max_steps=cfg.max_inference_steps,
                               thresholds=None,tau=0.5)[0][-1]
    if mode == 'encoder_only':
        return z/(jnp.sqrt(jnp.mean(z*z,axis=-1,keepdims=True))+1e-6)
    z_loop=jax.vmap(fn)(z)
    if mode == 'lattice_only':
        return z_loop/(jnp.sqrt(jnp.mean(z_loop*z_loop,axis=-1,keepdims=True))+1e-6)
    # Preserve prompt identity while retaining the deliberated lattice state.
    out=z+z_loop
    return out/(jnp.sqrt(jnp.mean(out*out,axis=-1,keepdims=True))+1e-6)


def teacher_targets(rows,tok,gids,qwen_path,cache,max_a=24):
    if os.path.exists(cache): return np.load(cache)['probs']
    qp=load_qwen_params(qwen_path); outs=[]
    for i,r in enumerate(rows):
        prefix=f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{r['user']}<|im_end|>\n<|im_start|>assistant\n"
        p=tok.encode(prefix).ids
        a=tok.encode(r['answer']).ids[:max_a-1]+[tok.token_to_id('<|im_end|>') or 151645]
        ids=(p+a)[:64]; ids=ids+[0]*(64-len(ids))
        logits=qwen_forward(qp,jnp.array([ids],dtype=jnp.int32),n_layers=4)
        start=len(p)-1
        sel=np.array(logits[0,start:start+max_a,:][:,gids],dtype=np.float32,copy=True)
        sel-=sel.max(-1,keepdims=True); prob=np.exp(sel); prob/=prob.sum(-1,keepdims=True)
        outs.append(prob); print(f"teacher {i+1}/{len(rows)}")
    probs=np.stack(outs); os.makedirs(os.path.dirname(cache),exist_ok=True)
    np.savez(cache,probs=probs); return probs


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--data',default='data/causal_dialogue.json')
    ap.add_argument('--tokenizer',default='checkpoints/Qwen2.5-0.5B-Instruct/tokenizer.json')
    ap.add_argument('--qwen',default='checkpoints/qwen_instruct/qwen_params.npz')
    ap.add_argument('--output',default='checkpoints/causal_student_v3')
    ap.add_argument('--teacher-cache',default='data/causal_teacher/probs.npz')
    ap.add_argument('--augment-paraphrases',action='store_true')
    ap.add_argument('--state-mode',choices=['residual','encoder_only','lattice_only'],default='residual')
    ap.add_argument('--seed',type=int,default=7)
    args=ap.parse_args(); tok=Tokenizer.from_file(args.tokenizer)
    rows,q,x,y,mask,gids,pad,bos,eos=local_data(args.data,tok,augment=args.augment_paraphrases)
    teacher=teacher_targets(rows,tok,gids,args.qwen,args.teacher_cache,y.shape[1])
    cfg=dataclasses.replace(LCMConfig(),d_model=64,d_ff=96,n_heads=4,d_head=16,
        vocab_size=len(gids),max_seq_len=q.shape[1],M_top=64,M_fine=32,
        M_sparse=64,M_lr=32,M_man=64,M_bind=64,M_contrast=64,
        n_self_codes=16,max_inference_steps=4,use_bf16=False)
    p=init_student(cfg,jax.random.PRNGKey(args.seed),len(gids)); opt=optax.adamw(2e-3)
    state=opt.init(p); Q=jnp.array(q); X=jnp.array(x); Y=jnp.array(y)
    M=jnp.array(mask); T=jnp.array(teacher)
    @jax.jit
    def step(p,state,Q,X,Y,M,T):
      def loss(pp):
        z=z_state(pp,Q,cfg,args.state_mode); logits=gen_head_forward(pp['gen_head'],z,X)
        ce=optax.softmax_cross_entropy_with_integer_labels(logits,Y)
        ce=(ce*M).sum()/M.sum()
        logp=jax.nn.log_softmax(logits,-1)
        kd=-(T*logp*M[...,None]).sum()/M.sum()
        z0=jax.lax.stop_gradient(jnp.zeros_like(z))
        l0=gen_head_forward(pp['gen_head'],z0,X)[:,0]
        true=logits[:,0]; idx=Y[:,0,None]
        margin=jnp.maximum(0.,.5-jnp.take_along_axis(true,idx,1).squeeze()+
                              jnp.take_along_axis(l0,idx,1).squeeze()).mean()
        return ce+.02*kd+margin,(ce,kd,margin)
      (loss,aux),g=jax.value_and_grad(loss,has_aux=True)(p)
      up,state=opt.update(g,state,p); return optax.apply_updates(p,up),state,loss,aux
    rng=np.random.default_rng(args.seed+35)
    for i in range(args.steps):
      ids=rng.choice(len(rows),args.batch_size,replace=False)
      p,state,loss,aux=step(p,state,Q[ids],X[ids],Y[ids],M[ids],T[ids])
      if i%10==0 or i+1==args.steps: print('step',i+1,'loss',float(loss),'ce/kd/margin',*[float(v) for v in aux])
    os.makedirs(args.output,exist_ok=True)
    np.save(os.path.join(args.output,'global_token_ids.npy'),gids)
    with open(os.path.join(args.output,'student.pkl'),'wb') as f:
      pickle.dump({'params':jax.tree.map(np.asarray,p),'config':dataclasses.asdict(cfg),'step':args.steps,'state_mode':args.state_mode},f)
    z=z_state(p,Q,cfg,args.state_mode); cur=jnp.full((len(rows),1),bos,dtype=jnp.int32); out=[]
    for _ in range(y.shape[1]):
      padded=jnp.pad(cur,((0,0),(0,y.shape[1]-cur.shape[1])),constant_values=pad)
      logits=gen_head_forward(p['gen_head'],z,padded)
      nxt=jnp.argmax(logits[:,cur.shape[1]-1],-1); out.append(np.asarray(nxt)); cur=jnp.concatenate([cur,nxt[:,None]],1)
    out=np.stack(out,1); report=[]
    for i,r in enumerate(rows):
      seq=out[i].tolist(); seq=seq[:seq.index(eos)] if eos in seq else seq
      text=tok.decode([int(gids[t]) for t in seq if t != pad])
      report.append({'prompt':r['user'],'target':r['answer'],'generated':text})
      print(r['user'],'=>',text)
    json.dump(report,open(os.path.join(args.output,'rollout.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print('saved student-only checkpoint:',args.output)

if __name__=='__main__': main()
