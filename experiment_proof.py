#!/usr/bin/env python3
"""300-step proof-of-concept: self-supervised Per-Voxel refinement with pruning."""
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch, torch.nn as nn, torch.nn.functional as F, sys, json, math, random, time
import numpy as np
from PIL import Image
PROJECT_ROOT = '/home-ldap/sunchang/3dProjects/GuassDiff'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'anysplat'))
from safetensors.torch import load_file
import dataclasses

device = 'cuda:0'; H, W = 224, 448; OUT = '/data/sunchang/exp_train/proof'
os.makedirs(OUT, exist_ok=True)

from src.model.model.anysplat import AnySplat
from src.model.encoder.anysplat import EncoderAnySplatCfg, OpacityMappingCfg
from src.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg, DecoderSplattingCUDA
from src.model.types import Gaussians
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

print('Loading AnySplat...')
with open(os.path.join(PROJECT_ROOT, 'anysplat/pretrained_model/config.json')) as f: cfg = json.load(f)
enc_dict = cfg['encoder_cfg']
valid = {f.name for f in dataclasses.fields(EncoderAnySplatCfg)}
for k,v in {'scale_align':False,'n_offsets':2,'color_attr':'3D','mlp_type':'unified','scaffold':True,
    'intermediate_layer_idx':None,'voxelize':False,'freeze_backbone':False,'freeze_module':'None',
    'distill':False,'num_surfaces':1,'gaussians_per_pixel':1}.items():
    enc_dict.setdefault(k,v)
ef = {k:v for k,v in enc_dict.items() if k in valid}
ef['backbone'] = BackboneCrocoCfg(**ef['backbone'])
ef['gaussian_adapter'] = GaussianAdapterCfg(**enc_dict.get('gaussian_adapter', {}))
ef['visualizer'] = EncoderVisualizerEpipolarCfg(**enc_dict.get('visualizer', {}))
ef['opacity_mapping'] = OpacityMappingCfg(**enc_dict.get('opacity_mapping', {}))
model = AnySplat(EncoderAnySplatCfg(**ef), DecoderSplattingCUDACfg(**cfg['decoder_cfg'])).to(device)
model.load_state_dict(load_file(os.path.join(PROJECT_ROOT, 'anysplat/pretrained_model/model.safetensors')), strict=False)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
decoder = DecoderSplattingCUDA(DecoderSplattingCUDACfg(name='splatting_cuda',background_color=[1,1,1],make_scale_invariant=False)).to(device)

print('Loading Wan VAE...')
from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained('Wan-AI/Wan2.2-TI2V-5B-Diffusers',subfolder='vae',torch_dtype=torch.float32).eval().to(device)
for p in vae.parameters(): p.requires_grad_(False)
z_dim=48; s_t=4

SCENE='/data/sunchang/dl3dv_benchmark/032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7/nerfstudio'
with open(os.path.join(SCENE,'transforms.json')) as f: meta = json.load(f)

try:
    from lpips import LPIPS
    lpips_fn = LPIPS(net='vgg').to(device).eval()
    for p in lpips_fn.parameters(): p.requires_grad_(False)
    print('LPIPS: loaded')
except: lpips_fn = None

class RefinerV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(131,256),nn.LayerNorm(256),nn.ReLU(),
            nn.Linear(256,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,83))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x):
        d = self.net(x)  # [B, N, 83]
        return d[:,:,:82], torch.sigmoid(d[:,:,82:])  # delta [B,N,82], keep_prob [B,N,1]

refiner = RefinerV2().to(device)
opt = torch.optim.AdamW(refiner.parameters(), lr=3e-4)
print(f'Refiner: {sum(p.numel() for p in refiner.parameters())/1e3:.0f}K params')

def load_ctx(bi):
    imgs = [Image.open(os.path.join(SCENE,f'images_4/frame_{bi+i:05d}.png')).resize((W,H),Image.LANCZOS)
            for i in range(2)]
    return torch.stack([torch.tensor(np.array(i),dtype=torch.float32).permute(2,0,1)/255.0 for i in imgs]).unsqueeze(0).to(device)

def project_video(wan_feats, pts, ext, intr_m):
    V = ext.shape[1]; N = pts.shape[1]; B = pts.shape[0]
    vf = []
    for vi in range(V):
        cp = ext[0,vi,:3,:3]@pts[0].T + ext[0,vi,:3,3:4]
        uv = intr_m[0,vi]@cp
        grid = torch.stack([uv[0]/(uv[2]+1e-8)/(W-1)*2-1, uv[1]/(uv[2]+1e-8)/(H-1)*2-1],dim=1).unsqueeze(0).unsqueeze(0)
        vf.append(F.grid_sample(wan_feats[0,vi:vi+1],grid,mode='bilinear',align_corners=False)[:,:,0])
    return torch.stack(vf).mean(0).permute(0,2,1)  # [B, N, feat_dim]

def save_render(g, name, step_=0):
    sd = os.path.join(OUT, f'step_{step_}')
    os.makedirs(sd, exist_ok=True)
    out = decoder.forward(g, ext_c2w, intr_n,
        torch.tensor([[0.1]],device=device),torch.tensor([[100.0]],device=device),(H,W))
    img = out.color[0,0].detach().cpu().permute(1,2,0).numpy()
    Image.fromarray((img.clip(0,1)*255).astype(np.uint8)).save(os.path.join(sd,f'{name}.png'))

print('Training 300 steps...')
t_start=time.time()

for step in range(301):
    bi = random.randint(50, 200)
    ctx = load_ctx(bi); ctx_n = ctx*2-1

    with torch.no_grad():
        enc_out,_ = model(ctx_n, 100000); g=enc_out.gaussians; N=g.means.shape[1]
        with torch.amp.autocast('cuda',enabled=True,dtype=torch.bfloat16):
            tokens,ps = model.encoder.aggregator(ctx_n.to(torch.bfloat16),
                intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
        with torch.amp.autocast('cuda',enabled=False):
            pe = model.encoder.camera_head(tokens)
            ext, intr = pose_encoding_to_extri_intri(pe[-1], (H,W))
            intr_n_ = intr.clone(); intr_n_[:,:,0]/=W; intr_n_[:,:,1]/=H
        # Wan features
        wan = torch.zeros(1,2,z_dim,H,W,device=device)
        for vi in range(2):
            img=ctx[0,vi].float().unsqueeze(0).unsqueeze(2).repeat(1,1,s_t,1,1)
            h=vae._encode(img); wan[0,vi]=F.interpolate(h[:,:z_dim,:,0].float(),size=(H,W),mode='bilinear',align_corners=False)[0]

    # GT camera for view 1
    tgt_T = torch.tensor(meta['frames'][bi+1]['transform_matrix'],dtype=torch.float32).to(device)
    # Also for view 0
    ctx_T = torch.tensor(meta['frames'][bi]['transform_matrix'],dtype=torch.float32).to(device)

    # Render at BOTH views using GT cameras
    # First: project video features using model's predicted cameras (ext)
    # Then: render using GT cameras (tgt_T, ctx_T)
    extrinsics = torch.stack([ctx_T, tgt_T])  # [2, 4, 4]
    ext_gt = extrinsics.unsqueeze(0)  # [1, 2, 4, 4]
    intr_n = intr_n_[:,0:1].expand(-1,2,-1,-1)  # same intrinsics for both views

    # Voxel features
    vf = torch.cat([g.opacities.unsqueeze(-1), g.scales, g.rotations, g.harmonics.view(1,N,-1)], dim=-1)
    video_vox = project_video(wan, g.means, ext, intr)

    # Refine
    cat_in = torch.cat([vf, video_vox], dim=-1)
    delta, keep = refiner(cat_in)

    # Build refined Gaussians
    s = 0.001*F.softplus((vf[:,:,1:4]+delta[:,:,:3]).clamp(-10,10)).clamp(max=0.3)
    rn_ = (vf[:,:,4:8]+delta[:,:,3:7]).norm(dim=-1,keepdim=True)
    r = (vf[:,:,4:8]+delta[:,:,3:7])/(rn_+1e-8)
    w_,x_,y_,z_ = r.unbind(-1)
    R = torch.stack([1-2*y_*y_-2*z_*z_,2*x_*y_-2*w_*z_,2*x_*z_+2*w_*y_,
        2*x_*y_+2*w_*z_,1-2*x_*x_-2*z_*z_,2*y_*z_-2*w_*x_,
        2*x_*z_-2*w_*y_,2*y_*z_+2*w_*x_,1-2*x_*x_-2*y_*y_],dim=-1).view(1,N,3,3)
    I = torch.eye(3,device=device).unsqueeze(0).unsqueeze(0)
    cov = R@(I*s.unsqueeze(-1)).pow(2)@R.transpose(-1,-2)
    sh = (vf[:,:,8:]+delta[:,:,7:]).view(1,N,3,-1)
    op = torch.sigmoid(vf[:,:,0:1]+delta[:,:,0:1]).squeeze(-1)
    refined_g = Gaussians(means=g.means, covariances=cov, harmonics=sh, opacities=op, scales=s, rotations=r)

    # Render at BOTH views
    out = decoder.forward(refined_g, ext_gt, intr_n,
        torch.tensor([[0.1]],device=device),torch.tensor([[100.0]],device=device),(H,W))

    # Loss over both views
    rc0, rc1 = out.color[0,0], out.color[0,1]
    gt0, gt1 = ctx[0,0], ctx[0,1]
    lmse = F.mse_loss(rc0, gt0) + F.mse_loss(rc1, gt1)
    llp = 0
    if lpips_fn:
        llp = lpips_fn(rc0.unsqueeze(0),gt0.unsqueeze(0)).mean() + lpips_fn(rc1.unsqueeze(0),gt1.unsqueeze(0)).mean()
    lreg = 1e-6*delta.pow(2).mean() + 1e-3*(1-keep).mean()
    loss = lmse + 0.05*llp + lreg

    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(refiner.parameters(),1.0); opt.step()

    if step%30==0:
        psnr0 = -10*math.log10(max(F.mse_loss(rc0,gt0).item(),1e-10))
        psnr1 = -10*math.log10(max(F.mse_loss(rc1,gt1).item(),1e-10))
        print(f'Step{step:3d}: PSNR=[{psnr0:.1f},{psnr1:.1f}] |Δ|={delta.norm().item():.1f} keep={keep.mean().item():.3f} loss={loss.item():.3f}')

    if step%100==0 or step==300:
        os.makedirs(os.path.join(OUT,f'step_{step}'), exist_ok=True)
        with torch.no_grad():
            ext_c2w = torch.cat([ext,torch.tensor([0,0,0,1],device=device).reshape(1,1,1,4).expand(1,2,-1,-1)],dim=2).inverse()
            intr_n_fixed = intr_n_[:,0:1]
            # Save initial
            g_init_renders = decoder.forward(g, ext_c2w, intr_n_fixed.expand(-1,2,-1,-1),
                torch.tensor([[0.1]],device=device),torch.tensor([[100.0]],device=device),(H,W))
            for vi in range(2):
                img = g_init_renders.color[0,vi].detach().cpu().permute(1,2,0).numpy()
                Image.fromarray((img.clip(0,1)*255).astype(np.uint8)).save(os.path.join(OUT,f'step_{step}',f'init_v{vi}.png'))
            # Save refined
            ref_renders = decoder.forward(refined_g, ext_gt, intr_n,
                torch.tensor([[0.1]],device=device),torch.tensor([[100.0]],device=device),(H,W))
            for vi in range(2):
                img = ref_renders.color[0,vi].detach().cpu().permute(1,2,0).numpy()
                Image.fromarray((img.clip(0,1)*255).astype(np.uint8)).save(os.path.join(OUT,f'step_{step}',f'ref_v{vi}.png'))
            print(f'  Saved step_{step}')

print(f'Done! {time.time()-t_start:.0f}s')
torch.save(refiner.state_dict(), os.path.join(OUT, 'refiner_final.pt'))
