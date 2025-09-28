#!/usr/bin/env python3
"""
FLORES Beam Search Implementation using Hugging Face Transformers

This script implements beam search for machine translation evaluation
on the FLORES dataset using only the Hugging Face library.
"""

import json
import argparse
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    GenerationConfig,
    DataCollatorForSeq2Seq
)
from tqdm import tqdm
import sacrebleu
from comet import load_from_checkpoint
import numpy as np


class FloresBeamSearch:
    """Beam search implementation for FLORES dataset evaluation."""
    
    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 8,
        comet_model_path: Optional[str] = None
    ):
        """
        Initialize the FLORES beam search evaluator.
        
        Args:
            model_name: Hugging Face model name or path
            tokenizer_name: Tokenizer name (if different from model)
            device: Device to use ('auto', 'cuda', 'cpu')
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            comet_model_path: Path to COMET model for evaluation
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.comet_model_path = comet_model_path
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Load COMET model if provided
        self.comet_model = None
        if self.comet_model_path and os.path.exists(self.comet_model_path):
            print(f"Loading COMET model from {self.comet_model_path}")
            self.comet_model = load_from_checkpoint(self.comet_model_path)
    
    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        print(f"Loading tokenizer: {self.tokenizer_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load model
        try:
            # Try seq2seq model first (for translation models)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model_type = "seq2seq"
            print("Loaded as seq2seq model")
        except Exception as e:
            print(f"Failed to load as seq2seq model: {e}")
            try:
                # Fallback to causal LM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.model_type = "causal"
                print("Loaded as causal LM model")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model as both seq2seq and causal LM: {e2}")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model type: {self.model_type}")
    
    def generate_with_beam_search(
        self,
        input_texts: List[str],
        num_beams: int = 4,
        num_return_sequences: int = 1,
        max_new_tokens: int = 256,
        early_stopping: bool = True,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50
    ) -> List[str]:
        """
        Generate translations using beam search.
        
        Args:
            input_texts: List of source texts to translate
            num_beams: Number of beams for beam search
            num_return_sequences: Number of sequences to return per input
            max_new_tokens: Maximum number of new tokens to generate
            early_stopping: Whether to stop when EOS is generated
            length_penalty: Length penalty for beam search
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            do_sample: Whether to use sampling
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            top_k: Top-k for sampling
            
        Returns:
            List of generated translations
        """
        all_generations = []
        
        # Process in batches
        for i in tqdm(range(0, len(input_texts), self.batch_size), desc="Generating"):
            batch_texts = input_texts[i:i + self.batch_size]
            
            # Tokenize inputs
            if self.model_type == "seq2seq":
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
            else:
                # For causal LM, we might need to format the input differently
                formatted_texts = [f"Translate to English: {text}" for text in batch_texts]
                inputs = self.tokenizer(
                    formatted_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with beam search
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    early_stopping=early_stopping,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode outputs
            if self.model_type == "seq2seq":
                # For seq2seq, the outputs are already the generated sequences
                generated_ids = outputs
            else:
                # For causal LM, we need to remove the input part
                input_length = inputs['input_ids'].shape[1]
                generated_ids = outputs[:, input_length:]
            
            # Decode the generated sequences
            batch_generations = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Reshape if we have multiple sequences per input
            if num_return_sequences > 1:
                batch_generations = [
                    batch_generations[i:i + num_return_sequences]
                    for i in range(0, len(batch_generations), num_return_sequences)
                ]
            else:
                batch_generations = [[gen] for gen in batch_generations]
            
            all_generations.extend(batch_generations)
        
        # Flatten if single sequence per input
        if num_return_sequences == 1:
            all_generations = [gen[0] for gen in all_generations]
        
        return all_generations
    
    def evaluate_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BLEU score using sacrebleu."""
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        # Clean texts
        predictions = [pred.strip() for pred in predictions]
        references = [ref.strip() for ref in references]
        
        # Calculate BLEU score
        bleu_score = sacrebleu.corpus_bleu(
            predictions, 
            [references], 
            tokenize="flores101", 
            force=True
        ).score
        
        return {"bleu": bleu_score}
    
    def evaluate_comet(self, sources: List[str], predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate COMET score."""
        if not self.comet_model:
            return {"comet": None}
        
        if len(sources) != len(predictions) or len(predictions) != len(references):
            raise ValueError("Number of sources, predictions, and references must match")
        
        # Prepare inputs for COMET
        inputs = [
            {"src": src.strip(), "mt": pred.strip(), "ref": ref.strip()} 
            for src, pred, ref in zip(sources, predictions, references)
        ]
        
        # Calculate COMET score
        output = self.comet_model.predict(inputs, batch_size=16, gpus=1 if self.device == "cuda" else 0)
        
        return {"comet": output.system_score}
    
    def load_flores_data(self, data_path: str) -> Dict[str, List[str]]:
        """Load FLORES dataset from JSONL file."""
        data = {"sources": [], "references": []}
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data["sources"].append(item.get("src", ""))
                    data["references"].append(item.get("ref", ""))
                except json.JSONDecodeError:
                    continue
        
        return data
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    
    def test_beam_search(self, test_text: str = "Hello world", num_beams: int = 4) -> str:
        """Test beam search with a simple example."""
        print(f"Testing beam search with: '{test_text}'")
        print(f"Model type: {self.model_type}")
        print(f"Number of beams: {num_beams}")
        
        try:
            result = self.generate_with_beam_search(
                [test_text],
                num_beams=num_beams,
                max_new_tokens=50
            )
            print(f"Generated: '{result[0]}'")
            return result[0]
        except Exception as e:
            print(f"Error in beam search: {e}")
            raise


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="FLORES Beam Search Evaluation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Hugging Face model name or path")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                       help="Tokenizer name (if different from model)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to FLORES dataset JSONL file")
    parser.add_argument("--output_path", type=str, default="results.json",
                       help="Path to save results")
    
    # Generation arguments
    parser.add_argument("--num_beams", type=int, default=4,
                       help="Number of beams for beam search")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                       help="Length penalty for beam search")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                       help="Size of n-grams to avoid repeating")
    
    # Evaluation arguments
    parser.add_argument("--comet_model_path", type=str, default=None,
                       help="Path to COMET model for evaluation")
    parser.add_argument("--skip_bleu", action="store_true",
                       help="Skip BLEU evaluation")
    parser.add_argument("--skip_comet", action="store_true",
                       help="Skip COMET evaluation")
    
    # Test arguments
    parser.add_argument("--test_mode", action="store_true",
                       help="Run in test mode with a simple example")
    parser.add_argument("--test_text", type=str, default="Hello world",
                       help="Text to use for testing")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = FloresBeamSearch(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        batch_size=args.batch_size,
        comet_model_path=args.comet_model_path
    )
    
    # Test mode
    if args.test_mode:
        print("Running in test mode...")
        evaluator.test_beam_search(args.test_text, args.num_beams)
        return
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = evaluator.load_flores_data(args.data_path)
    print(f"Loaded {len(data['sources'])} examples")
    
    # Generate translations
    print("Generating translations with beam search...")
    predictions = evaluator.generate_with_beam_search(
        input_texts=data["sources"],
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size
    )
    
    # Evaluate
    results = {
        "model_name": args.model_name,
        "num_examples": len(data["sources"]),
        "generation_config": {
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size
        },
        "predictions": predictions,
        "sources": data["sources"],
        "references": data["references"]
    }
    
    # BLEU evaluation
    if not args.skip_bleu:
        print("Calculating BLEU score...")
        bleu_results = evaluator.evaluate_bleu(predictions, data["references"])
        results.update(bleu_results)
        print(f"BLEU Score: {bleu_results['bleu']:.4f}")
    
    # COMET evaluation
    if not args.skip_comet and evaluator.comet_model:
        print("Calculating COMET score...")
        comet_results = evaluator.evaluate_comet(
            data["sources"], predictions, data["references"]
        )
        results.update(comet_results)
        print(f"COMET Score: {comet_results['comet']:.4f}")
    
    # Save results
    evaluator.save_results(results, args.output_path)
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
