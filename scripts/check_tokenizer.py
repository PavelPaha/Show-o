#!/usr/bin/env python3
"""
Скрипт для проверки special tokens в phi-1.5 tokenizer.
"""

from transformers import AutoTokenizer

def main():
    print("=" * 60)
    print("Checking phi-1.5 tokenizer special tokens")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    
    print(f"\n📌 Token IDs:")
    print(f"   bos_token_id = {tokenizer.bos_token_id}")
    print(f"   eos_token_id = {tokenizer.eos_token_id}")
    print(f"   pad_token_id = {tokenizer.pad_token_id}")
    print(f"   unk_token_id = {tokenizer.unk_token_id}")
    
    print(f"\n📌 Token strings:")
    print(f"   bos_token = '{tokenizer.bos_token}'")
    print(f"   eos_token = '{tokenizer.eos_token}'")
    print(f"   pad_token = '{tokenizer.pad_token}'")
    print(f"   unk_token = '{tokenizer.unk_token}'")
    
    print(f"\n📌 Vocab size: {len(tokenizer)}")
    
    # Check if BOS == EOS
    if tokenizer.bos_token_id == tokenizer.eos_token_id:
        print(f"\n✅ CONFIRMED: bos_token_id == eos_token_id == {tokenizer.bos_token_id}")
        print("   This is normal for GPT-2 based tokenizers!")
    else:
        print(f"\n❌ bos_token_id ({tokenizer.bos_token_id}) != eos_token_id ({tokenizer.eos_token_id})")
    
    # Show some special tokens from Show-o
    print("\n📌 Show-o special token IDs (if added):")
    special_tokens = ["<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"]
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            print(f"   {token} = {token_id}")
        else:
            print(f"   {token} = NOT IN VOCAB (would be {tokenizer.unk_token_id})")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()







