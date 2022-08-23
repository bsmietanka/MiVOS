"""
Heart of most evaluation scripts (DAVIS semi-sup/interactive, GUI)
Handles propagation and fusion
See eval_semi_davis.py / eval_interactive_davis.py for examples
"""

import os
import shutil
from typing import List, Tuple, Union
import torch
import numpy as np
import cv2
from interact.interactive_utils import images_to_torch

from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
from model.aggregate import aggregate_sbg, aggregate_wbg

from util.tensor_util import pad_divide_by, unpad



# TODO: add fixed length buffers for faster access times
class TorchImageCache:

    def __init__(self, image_paths: List[str], divide_by: int = 16,
                 dev: str = "cpu"):
        self.image_paths = image_paths
        self.divide_by = divide_by
        self.dev = dev
        assert len(self) > 0
        self.pad_array = None
        self.channels, self.height, self.width = self[0].shape[-3:]
        assert self.pad_array is not None
        self.orig_height, self.orig_width = cv2.imread(self.image_paths[0]).shape[:2]

    def __getitem__(self, idx: int) -> torch.Tensor:
        # load and add batch dimension
        frame = cv2.cvtColor(cv2.imread(self.image_paths[idx]),
                             cv2.COLOR_BGR2RGB)[None]
        torch_imgs = images_to_torch(frame, self.dev)[0]
        torch_imgs, pad_array = pad_divide_by(torch_imgs,
                                              self.divide_by)
        if self.pad_array is None:
            self.pad_array = pad_array
        return torch_imgs

    def __len__(self) -> int:
        return len(self.image_paths)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (len(self.image_paths), self.channels, self.height, self.width)

    @property
    def original_shape(self) -> Tuple[int, int, int, int]:
        return (len(self.image_paths), self.channels, self.orig_height, self.orig_width)


class MaskCache:

    def __init__(self, root: str, num_masks: int, height: int, width: int,
                 divide_by: int = 16, dev: str = "cpu", pad = None):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.num_masks = num_masks
        self.height = height
        self.width = width
        self.divide_by = divide_by
        self.dev = dev
        if pad is None:
            self.pad = (0, 0, 0, 0)
        else:
            self.pad = pad

    def __mask_path(self, idx: int) -> str:
        return os.path.join(self.root, f"{idx}.png")

    def __getitem__(self, idx: int) -> np.ndarray:
        assert idx < self.num_masks
        mask_path = self.__mask_path(idx)
        if not os.path.exists(mask_path):
            return np.zeros((self.height, self.width), np.uint8)
        return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    @torch.no_grad()
    def get_torch(self, idx: int) -> torch.ByteTensor:
        mask = torch.ByteTensor(self[idx])
        return pad_divide_by(mask, self.divide_by)[0].unsqueeze(0)

    @torch.no_grad()
    def update(self, idx: int, mask: Union[np.ndarray, torch.Tensor]) -> None:
        assert idx < self.num_masks
        mask_path = self.__mask_path(idx)
        if not isinstance(mask, (np.ndarray, torch.Tensor)):
            raise RuntimeError(f"Mask is an unsupported type: {type(mask)}")
        if isinstance(mask, torch.Tensor):
            mask = mask.byte().cpu().numpy()
            mask = unpad(mask, self.pad)
        mask = mask.reshape((self.height, self.width))
        success = cv2.imwrite(mask_path, mask)
        if not success:
            raise RuntimeError(f"Couldn't save mask: {mask_path}")

    def __len__(self) -> int:
        return self.num_masks

    def reset(self, idx: int):
        zero_mask = np.zeros((self.height, self.width), np.uint8)
        self.update(idx, zero_mask)


class TorchProbsCache:

    def __init__(self, root: str, num_objects: int, num_masks: int,
                 height: int, width: int, dev: str = "cpu"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.num_objects = num_objects
        # actual number of objects including background
        self._num_obj = num_objects + 1
        self.num_masks = num_masks
        self.height = height
        self.width = width
        self.dev = dev

    def __prob_path(self, idx: int) -> str:
        return os.path.join(self.root, f"{idx}.pt")

    @torch.no_grad()
    def __getitem__(self, idx: int) -> torch.Tensor:
        assert idx < self.num_masks
        prob_path = self.__prob_path(idx)
        if not os.path.exists(prob_path):
            # singular dimension for compatibility with previous shapes
            probs = torch.zeros((self._num_obj, 1, self.height, self.width),
                               device=self.dev)
            probs[0] = 1e-7
            return probs
        return torch.load(prob_path)

    def __len__(self) -> int:
        return self.num_masks

    @torch.no_grad()
    def update(self, idx: int, probs: torch.Tensor) -> None:
        assert idx < self.num_masks
        prob_path = self.__prob_path(idx)
        probs = probs.reshape((self._num_obj, 1, self.height, self.width))
        torch.save(probs.cpu(), prob_path)


class InferenceCore:
    """
    images - leave them in original dimension (unpadded), but do normalize them. 
            Should be CPU tensors of shape B*T*3*H*W
            
    mem_profile - How extravagant I can use the GPU memory. 
                Usually more memory -> faster speed but I have not drawn the exact relation
                0 - Use the most memory
                1 - Intermediate, larger buffer 
                2 - Intermediate, small buffer 
                3 - Use the minimal amount of GPU memory
                Note that *none* of the above options will affect the accuracy
                This is a space-time tradeoff, not a space-performance one

    mem_freq - Period at which new memory are put in the bank
                Higher number -> less memory usage
                Unlike the last option, this *is* a space-performance tradeoff
    """
    def __init__(self, prop_net:PropagationNetwork, fuse_net:FusionNet, images, num_objects, 
                    mem_profile=0, mem_freq=5, device='cuda:0'):
        self.prop_net = prop_net.to(device, non_blocking=True)
        if fuse_net is not None:
            self.fuse_net = fuse_net.to(device, non_blocking=True)
        self.mem_profile = mem_profile
        self.mem_freq = mem_freq
        self.device = device

        if mem_profile == 0:
            self.data_dev = device
            self.result_dev = device
            self.q_buf_size = 105
            self.i_buf_size = -1 # no need to buffer image
        elif mem_profile == 1:
            self.data_dev = 'cpu'
            self.result_dev = device
            self.q_buf_size = 105
            self.i_buf_size = 105
        elif mem_profile == 2:
            self.data_dev = 'cpu'
            self.result_dev = 'cpu'
            self.q_buf_size = 3
            self.i_buf_size = 3
        else:
            self.data_dev = 'cpu'
            self.result_dev = 'cpu'
            self.q_buf_size = 1
            self.i_buf_size = 1

        # True dimensions
        self.k = num_objects

        # Pad each side to multiples of 16
        self.images = TorchImageCache(images, 16, self.data_dev)
        # Padded dimensions
        nh, nw = self.images.shape[-2:]
        h, w = self.images.original_shape[-2:]
        t = len(self.images)
        self.pad = self.images.pad_array

        # These two store the same information in different formats
        # TODO?: configuration of segmentation "project" directory structure
        shutil.rmtree("temp")
        self.masks = MaskCache("temp/masks", t, h, w, 16, "cpu", self.pad)

        # Object probabilities, background included
        self.prob = TorchProbsCache("temp/probs", self.k, t, nh, nw, "cpu")

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh // 16
        self.kw = self.nw // 16

        # TODO: replace with deque?
        self.query_buf = {}
        self.image_buf = {}
        self.interacted = set()

        self.certain_mem_k = None
        self.certain_mem_v = None

    def get_image_buffered(self, idx):
        # buffer the .cuda() calls
        if idx not in self.image_buf:
            # Flush buffer
            if len(self.image_buf) > self.i_buf_size:
                self.image_buf = {}
            self.image_buf[idx] = self.images[idx].to(self.device)

        return self.image_buf[idx]

    def get_query_kv_buffered(self, idx):
        # Queries' key/value never change, so we can buffer them here
        if idx not in self.query_buf:
            # Flush buffer
            if len(self.query_buf) > self.q_buf_size:
                self.query_buf = {}
            self.query_buf[idx] = self.prop_net.get_query_values(
                self.get_image_buffered(idx))

        return self.query_buf[idx]

    def do_pass(self, key_k, key_v, idx, forward=True, step_cb=None):
        """
        Do a complete pass that includes propagation and fusion
        key_k/key_v -  memory feature of the starting frame
        idx - Frame index of the starting frame
        forward - forward/backward propagation
        step_cb - Callback function used for GUI (progress bar) only
        """

        # Pointer in the memory bank
        num_certain_keys = self.certain_mem_k.shape[2]
        m_front = num_certain_keys

        # Determine the required size of the memory bank
        if forward:
            closest_ti = min([ti for ti in self.interacted if ti > idx] + [self.t])
            total_m = (closest_ti - idx - 1)//self.mem_freq + 1 + num_certain_keys
        else:
            closest_ti = max([ti for ti in self.interacted if ti < idx] + [-1])
            total_m = (idx - closest_ti - 1)//self.mem_freq + 1 + num_certain_keys
        K, CK, _, H, W = key_k.shape
        _, CV, _, _, _ = key_v.shape

        # Pre-allocate keys/values memory
        keys = torch.empty((K, CK, total_m, H, W), dtype=torch.float32, device=self.device)
        values = torch.empty((K, CV, total_m, H, W), dtype=torch.float32, device=self.device)

        # Initial key/value passed in
        keys[:,:,0:num_certain_keys] = self.certain_mem_k
        values[:,:,0:num_certain_keys] = self.certain_mem_v
        prev_in_mem = True
        last_ti = idx

        # Note that we never reach closest_ti, just the frame before it
        if forward:
            this_range = range(idx+1, closest_ti)
            step = +1
            end = closest_ti - 1
        else:
            this_range = range(idx-1, closest_ti, -1)
            step = -1
            end = closest_ti + 1

        for ti in this_range:
            if prev_in_mem:
                this_k = keys[:,:,:m_front]
                this_v = values[:,:,:m_front]
            else:
                this_k = keys[:,:,:m_front+1]
                this_v = values[:,:,:m_front+1]
            query = self.get_query_kv_buffered(ti)
            out_mask = self.prop_net.segment_with_query(this_k, this_v, *query)

            out_mask = aggregate_wbg(out_mask, keep_bg=True)

            if ti != end:
                keys[:,:,m_front:m_front+1], values[:,:,m_front:m_front+1] = self.prop_net.memorize(
                        self.get_image_buffered(ti), out_mask[1:])
                if abs(ti-last_ti) >= self.mem_freq:
                    # Memorize the frame
                    m_front += 1
                    last_ti = ti
                    prev_in_mem = True
                else:
                    prev_in_mem = False

            # In-place fusion, maximizes the use of queried buffer
            # esp. for long sequence where the buffer will be flushed
            if (closest_ti != self.t) and (closest_ti != -1):
                self.prob.update(ti,
                                 self.fuse_one_frame(closest_ti, idx, ti,
                                                     self.prob[ti], out_mask,
                                                     key_k, query[3])
                                 )
            else:
                self.prob.update(ti, out_mask)

            # Callback function for the GUI
            if step_cb is not None:
                step_cb()

        return closest_ti

    def fuse_one_frame(self, tc, tr, ti, prev_mask, curr_mask, mk16, qk16):
        assert(tc<ti<tr or tr<ti<tc)

        prob = torch.zeros((self.k, 1, self.nh, self.nw), dtype=torch.float32, device=self.device)

        # Compute linear coefficients
        nc = abs(tc-ti) / abs(tc-tr)
        nr = abs(tr-ti) / abs(tc-tr)
        dist = torch.FloatTensor([nc, nr]).to(self.device).unsqueeze(0)
        for k in range(1, self.k+1):
            attn_map = self.prop_net.get_attention(mk16[k-1:k], self.pos_mask_diff[k:k+1], self.neg_mask_diff[k:k+1], qk16)

            w = torch.sigmoid(self.fuse_net(self.get_image_buffered(ti), 
                    prev_mask[k:k+1].to(self.device), curr_mask[k:k+1].to(self.device), attn_map, dist))
            prob[k-1] = w 
        return aggregate_wbg(prob, keep_bg=True)

    def interact(self, mask, idx, total_cb=None, step_cb=None):
        """
        Interact -> Propagate -> Fuse

        mask - One-hot mask of the interacted frame, background included
        idx - Frame index of the interacted frame
        total_cb, step_cb - Callback functions for the GUI

        Return: all mask results in np format for DAVIS evaluation
        """
        self.interacted.add(idx)

        mask = mask.to(self.device)
        mask, _ = pad_divide_by(mask, 16, mask.shape[-2:])
        self.mask_diff = mask - self.prob[idx].to(self.device)
        self.pos_mask_diff = self.mask_diff.clamp(0, 1)
        self.neg_mask_diff = (-self.mask_diff).clamp(0, 1)

        self.prob.update(idx, mask)
        key_k, key_v = self.prop_net.memorize(self.get_image_buffered(idx), mask[1:])

        if self.certain_mem_k is None:
            self.certain_mem_k = key_k
            self.certain_mem_v = key_v
        else:
            self.certain_mem_k = torch.cat([self.certain_mem_k, key_k], 2)
            self.certain_mem_v = torch.cat([self.certain_mem_v, key_v], 2)

        if total_cb is not None:
            # Finds the total num. frames to process
            front_limit = min([ti for ti in self.interacted if ti > idx] + [self.t])
            back_limit = max([ti for ti in self.interacted if ti < idx] + [-1])
            total_num = front_limit - back_limit - 2 # -1 for shift, -1 for center frame
            if total_num > 0:
                total_cb(total_num)

        self.do_pass(key_k, key_v, idx, True, step_cb=step_cb)
        self.do_pass(key_k, key_v, idx, False, step_cb=step_cb)
        
        # This is a more memory-efficient argmax
        for ti in range(self.t):
            self.masks.update(ti, torch.argmax(self.prob[ti], dim=0))

    def update_mask_only(self, prob_mask, idx):
        """
        Interaction only, no propagation/fusion
        prob_mask - mask of the interacted frame, background included
        idx - Frame index of the interacted frame

        Return: all mask results in np format for DAVIS evaluation
        """
        mask = torch.argmax(prob_mask, 0)
        self.masks.update(idx, mask)
