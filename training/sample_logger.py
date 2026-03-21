"""
Sample logger for debugging training data.
Logs detailed information about input sequences to a file.
"""
import os
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class SampleLogger:
    """Logger for training samples - writes detailed sequence info to a log file."""
    
    def __init__(
        self,
        log_dir: str,
        text_tokenizer,
        log_every_n_steps: int = 100,
        max_samples_per_step: int = 2,
        enabled: bool = True,
    ):
        """
        Args:
            log_dir: Directory to save log files
            text_tokenizer: Tokenizer for decoding text tokens
            log_every_n_steps: Log samples every N global steps
            max_samples_per_step: Maximum number of samples to log per step (per task type)
            enabled: Whether logging is enabled
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.text_tokenizer = text_tokenizer
        self.log_every_n_steps = log_every_n_steps
        self.max_samples_per_step = max_samples_per_step
        self.enabled = enabled
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"samples_{timestamp}.log"
        self.json_log_file = self.log_dir / f"samples_{timestamp}.jsonl"
        
        # Special token IDs (will be set later)
        self.special_tokens = {}
        
        if self.enabled:
            with open(self.log_file, "w") as f:
                f.write(f"# Sample Log Started at {datetime.now().isoformat()}\n")
                f.write(f"# Log every {log_every_n_steps} steps, max {max_samples_per_step} samples per task\n")
                f.write("=" * 80 + "\n\n")
    
    def set_special_tokens(self, sptids_dict: Dict[str, torch.Tensor]):
        """Set special token IDs for decoding."""
        self.special_tokens = {
            int(v.item()) if torch.is_tensor(v) else int(v): k 
            for k, v in sptids_dict.items()
        }
    
    def should_log(self, global_step: int) -> bool:
        """Check if we should log at this step."""
        return self.enabled and (global_step % self.log_every_n_steps == 0)
    
    def _decode_sequence(
        self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        text_vocab_size: int = 50305,
    ) -> Dict[str, Any]:
        """Decode a sequence into human-readable format."""
        input_ids_list = input_ids.cpu().tolist()
        
        # Categorize tokens
        text_tokens = []
        image_tokens = []
        special_tokens_found = []
        masked_positions = []
        pad_positions = []
        
        # Find pad token id
        pad_id = self.special_tokens.get('<|pad|>', None)
        if pad_id is None:
            for tid, name in self.special_tokens.items():
                if name == '<|pad|>':
                    pad_id = tid
                    break
        
        for i, token_id in enumerate(input_ids_list):
            if token_id in self.special_tokens:
                special_tokens_found.append((i, self.special_tokens[token_id]))
                if self.special_tokens[token_id] == '<|pad|>':
                    pad_positions.append(i)
            elif token_id >= text_vocab_size and token_id < 58497:  # image token range
                image_tokens.append(i)
            elif token_id == 58497:  # mask token
                masked_positions.append(i)
            else:
                text_tokens.append(i)
        
        # Decode text portion
        text_token_ids = [input_ids_list[i] for i in text_tokens if input_ids_list[i] < text_vocab_size]
        decoded_text = ""
        if text_token_ids:
            try:
                decoded_text = self.text_tokenizer.decode(text_token_ids, skip_special_tokens=False)
            except:
                decoded_text = f"[Failed to decode {len(text_token_ids)} tokens]"
        
        # Count labels
        label_info = {}
        if labels is not None:
            labels_cpu = labels.cpu().tolist()
            non_ignore = [l for l in labels_cpu if l != -100]
            label_info = {
                "total_labels": len(labels_cpu),
                "non_ignore_labels": len(non_ignore),
                "ignore_labels": len(labels_cpu) - len(non_ignore),
            }
        
        # Find sequence structure (positions of key tokens)
        soi_pos = None
        eoi_pos = None
        t2i_pos = None
        mmu_pos = None
        eos_positions = []
        bos_positions = []
        
        for pos, name in special_tokens_found:
            if name == '<|soi|>':
                soi_pos = pos
            elif name == '<|eoi|>':
                eoi_pos = pos
            elif name == '<|t2i|>':
                t2i_pos = pos
            elif name == '<|mmu|>':
                mmu_pos = pos
            elif name == '<|eot|>' or 'eos' in name.lower():
                eos_positions.append(pos)
            elif name == '<|sot|>' or 'bos' in name.lower():
                bos_positions.append(pos)
        
        # Calculate structure info
        structure = {
            "pad_count": len(pad_positions),
            "pad_range": f"{min(pad_positions)}-{max(pad_positions)}" if pad_positions else "none",
            "soi_position": soi_pos,
            "eoi_position": eoi_pos,
            "t2i_position": t2i_pos,
            "mmu_position": mmu_pos,
            "eos_positions": eos_positions,
            "bos_positions": bos_positions,
        }
        
        if soi_pos is not None and eoi_pos is not None:
            structure["image_region"] = f"{soi_pos+1}-{eoi_pos-1} ({eoi_pos - soi_pos - 1} tokens)"
        
        return {
            "sequence_length": len(input_ids_list),
            "text_tokens_count": len(text_tokens),
            "image_tokens_count": len(image_tokens),
            "masked_tokens_count": len(masked_positions),
            "pad_tokens_count": len(pad_positions),
            "special_tokens": special_tokens_found,
            "structure": structure,
            "decoded_text": decoded_text[:500],  # truncate long texts
            "label_info": label_info,
            "first_10_tokens": input_ids_list[:10],
            "last_10_tokens": input_ids_list[-10:],
            "all_tokens": input_ids_list,  # Full sequence for debugging
        }
    
    def log_t2i_sample(
        self,
        global_step: int,
        batch_idx: int,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        original_text: str,
        mask_prob: float,
        image_tokens_ori: Optional[torch.Tensor] = None,
    ):
        """Log a T2I (text-to-image) sample."""
        if not self.should_log(global_step) or batch_idx >= self.max_samples_per_step:
            return
        
        info = self._decode_sequence(input_ids, labels)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "global_step": global_step,
            "task_type": "t2i",
            "batch_idx": batch_idx,
            "original_text": original_text[:200] if original_text else "",
            "mask_probability": float(mask_prob) if torch.is_tensor(mask_prob) else mask_prob,
            **info,
        }
        
        self._write_log(log_entry)
    
    def log_mmu_sample(
        self,
        global_step: int,
        batch_idx: int,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        domain: Optional[str] = None,
        original_text: Optional[str] = None,
    ):
        """Log an MMU (multimodal understanding) sample."""
        if not self.should_log(global_step) or batch_idx >= self.max_samples_per_step:
            return
        
        info = self._decode_sequence(input_ids, labels)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "global_step": global_step,
            "task_type": "mmu",
            "batch_idx": batch_idx,
            "domain": domain,
            "original_text": original_text[:200] if original_text else "",
            **info,
        }
        
        self._write_log(log_entry)
    
    def log_lm_sample(
        self,
        global_step: int,
        batch_idx: int,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        original_text: Optional[str] = None,
    ):
        """Log an LM (language modeling) sample."""
        if not self.should_log(global_step) or batch_idx >= self.max_samples_per_step:
            return
        
        info = self._decode_sequence(input_ids, labels)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "global_step": global_step,
            "task_type": "lm",
            "batch_idx": batch_idx,
            "original_text": original_text[:200] if original_text else "",
            **info,
        }
        
        self._write_log(log_entry)
    
    def log_batch_summary(
        self,
        global_step: int,
        batch_size_t2i: int,
        batch_size_lm: int,
        batch_size_mmu: int,
        total_seq_length: int,
        mask_prob_mean: float,
    ):
        """Log a summary of the batch."""
        if not self.should_log(global_step):
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "global_step": global_step,
            "entry_type": "batch_summary",
            "batch_size_t2i": batch_size_t2i,
            "batch_size_lm": batch_size_lm,
            "batch_size_mmu": batch_size_mmu,
            "total_batch_size": batch_size_t2i + batch_size_lm + batch_size_mmu,
            "sequence_length": total_seq_length,
            "mean_mask_probability": float(mask_prob_mean) if torch.is_tensor(mask_prob_mean) else mask_prob_mean,
        }
        
        self._write_log(log_entry, is_summary=True)
    
    def _write_log(self, entry: Dict[str, Any], is_summary: bool = False):
        """Write log entry to files."""
        # Write to JSONL file (without all_tokens to save space in jsonl)
        entry_for_json = {k: v for k, v in entry.items() if k != 'all_tokens'}
        with open(self.json_log_file, "a") as f:
            f.write(json.dumps(entry_for_json, ensure_ascii=False) + "\n")
        
        # Write human-readable format to log file
        with open(self.log_file, "a") as f:
            if is_summary:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"STEP {entry['global_step']} BATCH SUMMARY\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"  T2I samples: {entry['batch_size_t2i']}\n")
                f.write(f"  LM samples: {entry['batch_size_lm']}\n")
                f.write(f"  MMU samples: {entry['batch_size_mmu']}\n")
                f.write(f"  Sequence length: {entry['sequence_length']}\n")
                f.write(f"  Mean mask prob: {entry['mean_mask_probability']:.4f}\n")
                f.write(f"{'=' * 80}\n\n")
            else:
                f.write(f"\n{'-' * 60}\n")
                f.write(f"Step {entry['global_step']} | {entry['task_type'].upper()} | Sample {entry['batch_idx']}\n")
                f.write(f"{'-' * 60}\n")
                
                if entry.get('domain'):
                    f.write(f"  Domain: {entry['domain']}\n")
                if entry.get('mask_probability'):
                    f.write(f"  Mask probability: {entry['mask_probability']:.4f}\n")
                
                f.write(f"  Sequence length: {entry['sequence_length']}\n")
                f.write(f"  Text tokens: {entry['text_tokens_count']}\n")
                f.write(f"  Image tokens: {entry['image_tokens_count']} (non-masked)\n")
                f.write(f"  Masked tokens: {entry['masked_tokens_count']}\n")
                f.write(f"  Pad tokens: {entry.get('pad_tokens_count', 0)}\n")
                
                if entry.get('label_info'):
                    li = entry['label_info']
                    f.write(f"  Labels: {li['non_ignore_labels']}/{li['total_labels']} non-ignore\n")
                
                # Structure info
                if entry.get('structure'):
                    st = entry['structure']
                    f.write(f"\n  === SEQUENCE STRUCTURE ===\n")
                    f.write(f"  Pad count: {st['pad_count']}, range: {st['pad_range']}\n")
                    f.write(f"  T2I token position: {st['t2i_position']}\n")
                    f.write(f"  MMU token position: {st['mmu_position']}\n")
                    f.write(f"  SOI position: {st['soi_position']}\n")
                    f.write(f"  EOI position: {st['eoi_position']}\n")
                    if st.get('image_region'):
                        f.write(f"  Image region: {st['image_region']}\n")
                    f.write(f"  EOS positions: {st['eos_positions']}\n")
                    f.write(f"  BOS positions: {st['bos_positions']}\n")
                
                f.write(f"\n  First 10 tokens: {entry['first_10_tokens']}\n")
                f.write(f"  Last 10 tokens: {entry['last_10_tokens']}\n")
                
                # Full token sequence (grouped for readability)
                if entry.get('all_tokens'):
                    f.write(f"\n  === FULL TOKEN SEQUENCE ({len(entry['all_tokens'])} tokens) ===\n")
                    tokens = entry['all_tokens']
                    # Print in chunks of 50 for readability
                    for i in range(0, len(tokens), 50):
                        chunk = tokens[i:i+50]
                        f.write(f"  [{i:4d}-{min(i+49, len(tokens)-1):4d}]: {chunk}\n")
                
                f.write(f"\n  Original text: {entry.get('original_text', 'N/A')}\n")
                f.write(f"  Decoded text: {entry['decoded_text']}\n")

