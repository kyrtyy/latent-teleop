"""Re-render every logged state at 224 px and encode it with frozen DINOv2-S.
Writes <data>/emb_p{P}.npy (fp16, N x tokens x 384: CLS + PxP pooled patches) and <data>/meta.npz."""
import argparse
import glob

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from sim import make_env, render

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Frames(torch.utils.data.Dataset):
    def __init__(self, states):
        self.states, self.env = states, None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        if self.env is None:
            self.env = make_env()
        return torch.from_numpy(render(self.env, self.states[i]))


def build_meta(data, max_episodes=None):
    files = sorted(glob.glob(f"{data}/episodes/ep_*.npz"))[:max_episodes]
    states, contact, action, ep, offsets, profiles, success = [], [], [], [], [0], [], []
    for i, f in enumerate(files):
        L = np.load(f)
        states.append(L["state"])
        contact.append(L["contact"])
        action.append(np.concatenate([L["exec"], L["exec"][-1:]]))  # action at frame t = command executed during tick t
        ep.append(np.full(len(L["state"]), i))
        offsets.append(offsets[-1] + len(L["state"]))
        profiles.append(str(L["profile"]))
        success.append(bool(L["success"]))
    meta = dict(states=np.concatenate(states), contact=np.concatenate(contact), action=np.concatenate(action),
                ep=np.concatenate(ep), offsets=np.array(offsets), profiles=np.array(profiles),
                success=np.array(success), files=np.array(files))
    np.savez(f"{data}/meta.npz", **meta)
    return meta


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--pool", type=int, default=8, help="patch grid side after pooling; 0 = CLS only")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    meta = build_meta(args.data, args.max_episodes)
    n_tok = 1 + args.pool**2
    out = np.lib.format.open_memmap(f"{args.data}/emb_p{args.pool}.npy", "w+", np.float16, (len(meta["states"]), n_tok, 384))
    dev = "cuda"
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval().half()
    loader = torch.utils.data.DataLoader(Frames(meta["states"]), batch_size=args.batch, num_workers=args.workers)
    i = 0
    for b, img in enumerate(loader):
        # no HF image processor: its resize-256 + center-crop-224 would cut off the workspace edges
        x = ((img.permute(0, 3, 1, 2).float() / 255 - MEAN) / STD).to(dev).half()
        h = model(pixel_values=x).last_hidden_state  # (B, 1 + 16*16, 384)
        tok = h[:, :1]
        if args.pool:
            g = h[:, 1:].transpose(1, 2).reshape(len(h), 384, 16, 16)
            tok = torch.cat([tok, F.adaptive_avg_pool2d(g, args.pool).flatten(2).transpose(1, 2)], 1)
        out[i:i + len(h)] = tok.cpu().numpy()
        i += len(h)
        if b % 200 == 0:
            print(f"{i}/{len(out)} frames", flush=True)
    out.flush()
    print(f"wrote {args.data}/emb_p{args.pool}.npy {out.shape}")


if __name__ == "__main__":
    main()
