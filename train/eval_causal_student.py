"""Evaluate a saved student without loading Qwen."""
import argparse, dataclasses, json, pickle
import jax
import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer
from train.config import LCMConfig
from train.causal_student_train import z_state
from train.fusion import gen_head_forward

TESTS = [
 "狗叫起来是什么样的？", "接近零度以后液态水通常如何？",
 "雨停了，地面为何留下水迹？", "忘记给花浇水很久会如何？",
 "有电时按动台灯按钮，结果是什么？", "温暖环境为什么能让冰块消失？",
 "杯子坠落并撞击坚硬地面会如何？", "跑完一千米为何大口呼吸？",
 "关掉龙头为何能截断水流？", "搅拌盐水时盐粒去了哪里？",
 "大片黑云聚集后更可能出现什么天气？", "电池彻底没电时手机还能开机吗？",
 "手电筒换个方向，影子为何跟着变？", "纸放到明火旁边安全吗，会怎样？",
 "熬夜以后为何难以集中精神？", "下水道口堵塞而水持续流入，结果呢？",
 "气球越吹越大会不会一直增大？", "汤中的铁勺为何很快烫手？",
 "屋里有烟时开窗有什么作用？", "学过的内容多复习几遍有何效果？"
]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True)
 ap.add_argument('--tokenizer',default='checkpoints/Qwen2.5-0.5B-Instruct/tokenizer.json')
 args=ap.parse_args(); tok=Tokenizer.from_file(args.tokenizer)
 with open(args.checkpoint+'/student.pkl','rb') as f: ck=pickle.load(f)
 p=jax.tree_util.tree_map(jnp.asarray,ck['params']); cfg=LCMConfig(**ck['config']); gids=np.load(args.checkpoint+'/global_token_ids.npy')
 g2l={int(g):i for i,g in enumerate(gids)}; pad=g2l.get(tok.token_to_id('<|endoftext|>'),0)
 bos=g2l[tok.token_to_id('<|im_start|>')]; eos=g2l[tok.token_to_id('<|im_end|>')]
 q=[]
 for s in TESTS:
  ids=[g2l.get(i,pad) for i in tok.encode(s).ids[:32]]; q.append(ids+[pad]*(32-len(ids)))
 Q=jnp.array(q,dtype=jnp.int32); z=z_state(p,Q,cfg,ck.get('state_mode','residual')); cur=jnp.full((len(q),1),bos,dtype=jnp.int32)
 out=[]
 for _ in range(24):
  x=jnp.pad(cur,((0,0),(0,24-cur.shape[1])),constant_values=pad)
  logits=gen_head_forward(p['gen_head'],z,x); nxt=jnp.argmax(logits[:,cur.shape[1]-1],-1)
  out.append(np.asarray(nxt)); cur=jnp.concatenate([cur,nxt[:,None]],1)
 out=np.stack(out,1); results=[]
 for s,seq in zip(TESTS,out):
  a=seq.tolist(); a=a[:a.index(eos)] if eos in a else a
  text=tok.decode([int(gids[i]) for i in a if i!=pad]); results.append({'question':s,'answer':text}); print(s,'=>',text)
 with open(args.checkpoint+'/heldout_eval.json','w',encoding='utf-8') as f: json.dump(results,f,ensure_ascii=False,indent=2)

if __name__=='__main__': main()
