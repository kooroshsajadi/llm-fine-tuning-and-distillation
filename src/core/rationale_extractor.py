#!/usr/bin/env python3
import logging
import json
import re
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
from src.core.model_loader import ModelLoader
from src.data.data_preparation import TextPromptDataset
from src.utils.utils import batch_iterable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class RationaleExtractor:
    def __init__(
        self,
        teacher_model_name: str = "openai-community/gpt2-medium",
        adapter_path: str = None,
        cot_examples: List[Tuple[str, str, str]] = None,
        max_length: int = 128,
        temperature: float = 0.7
    ):
        """
        Extract rationales and labels from a teacher model using CoT prompting.
        Based on "Distilling Step-by-Step" (Hsieh et al., 2023).

        Args:
            teacher_model_name (str): HuggingFace or local model path (e.g., "data/gpt2_small_kd").
            adapter_path (str, optional): Path to LoRA adapter (e.g., "data/adapters/final_adapters/gpt2_lora").
            cot_examples (List[Tuple[str, str, str]]): List of (input, rationale, label) for few-shot CoT.
            max_length (int): Maximum sequence length for generation.
            temperature (float): Decoding temperature for generation.
        """
        self.max_length = max_length
        self.temperature = temperature
        self.cot_examples = cot_examples or []
        if not self.cot_examples:
            logger.warning("No CoT examples provided; using zero-shot prompting")

        # Load teacher model
        self.loader = ModelLoader(
            model_name=teacher_model_name,
            adapter_path=adapter_path,
            max_length=max_length,
            train_mode=False,
            use_qlora=False
        )
        self.teacher = self.loader.model
        self.tokenizer = self.loader.tokenizer

    def build_prompt(self, input_text: str) -> str:
        """
        Construct a CoT prompt with few-shot examples.

        Args:
            input_text (str): Input prompt to generate rationale for.

        Returns:
            str: Formatted CoT prompt.
        """
        prompt = ""
        for inp, rationale, label in self.cot_examples:
            prompt += f"Question: {inp}\nAnswer: {rationale} The answer is {label}.\n\n"
        prompt += f"Question: {input_text}\nAnswer:"
        return prompt

    def parse_output(self, output_text: str) -> Tuple[str, str]:
        """
        Parse LLM output to extract rationale and label.

        Args:
            output_text (str): Generated text from teacher model.

        Returns:
            Tuple[str, str]: (rationale, label).
        """
        try:
            # Match "The answer is <label>." with regex
            match = re.search(r"The answer is (.+?)\.", output_text, re.DOTALL)
            if match:
                label = match.group(1).strip()
                rationale = output_text[:match.start()].strip()
                return rationale, label
            else:
                logger.warning(f"Could not parse output: {output_text[:50]}...")
                return output_text.strip(), "UNKNOWN"
        except Exception as e:
            logger.error(f"Parsing failed: {str(e)}")
            return output_text.strip(), "UNKNOWN"

    def generate(self, prompt: str) -> str:
        """
        Generate text using the teacher model.

        Args:
            prompt (str): Input prompt for generation.

        Returns:
            str: Generated text.
        """
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.teacher.device)
            outputs = self.teacher.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Generation failed for prompt '{prompt[:50]}...': {str(e)}")
            return ""

    def extract_for_batch(self, inputs: List[str]) -> List[Dict]:
        """
        Generate rationales and labels for a batch of inputs.

        Args:
            inputs (List[str]): List of input prompts.

        Returns:
            List[Dict]: List of {'input': ..., 'rationale': ..., 'label': ...}.
        """
        results = []
        for input_text in inputs:
            try:
                prompt = self.build_prompt(input_text)
                output = self.generate(prompt)
                rationale, label = self.parse_output(output)
                results.append({
                    "input": input_text,
                    "rationale": rationale,
                    "label": label
                })
            except Exception as e:
                logger.error(f"Failed to process input '{input_text[:50]}...': {str(e)}")
                results.append({
                    "input": input_text,
                    "rationale": "",
                    "label": "UNKNOWN"
                })
        return results

    def extract_for_dataset(
        self,
        data_path: str = "data/synthetic/prompts_v1.txt",
        output_dir: str = "data/synthetic/rationale_v1",
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Extract rationales and labels for a dataset, saving to JSONL.

        Args:
            data_path (str): Path to prompt file (e.g., "data/synthetic/prompts_v1.txt").
            output_dir (str): Directory to save rationale JSONL.
            batch_size (int): Batch size for processing.

        Returns:
            List[Dict]: List of {'input': ..., 'rationale': ..., 'label': ...}.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        output_file = output_dir / "rationale.jsonl"

        dataset = TextPromptDataset(data_path)
        if len(dataset) == 0:
            raise ValueError(f"No prompts found in {data_path}")

        all_results = []
        with open(output_file, 'w', encoding='utf-8') as f:
            for batch in tqdm(
                batch_iterable(dataset.prompts, batch_size),
                desc="Extracting rationales",
                total=(len(dataset.prompts) + batch_size - 1) // batch_size
            ):
                batch_results = self.extract_for_batch(batch)
                for result in batch_results:
                    all_results.append(result)
                    json.dump(result, f)
                    f.write('\n')
                logger.debug(f"Processed batch, saved to {output_file}")

        logger.info(f"Rationales saved to {output_file}")
        return all_results

def main():
    """
    Extract rationales and labels from prompts using a teacher model.
    """
    try:
        # Example CoT prompts (replace with actual examples)
        cot_examples = [
            (
                "What is 2 + 2?",
                "To solve this, we need to add the numbers 2 and 2. Adding them gives 2 + 2 = 4.",
                "4"
            ),
            (
                "Is the sky blue?",
                "The sky appears blue due to Rayleigh scattering, which scatters shorter blue light more than other colors.",
                "Yes"
            )
        ]

        extractor = RationaleExtractor(
            teacher_model_name="openai-community/gpt2-medium",
            adapter_path="data/adapters/final_adapters/gpt2_lora",
            cot_examples=cot_examples,
            max_length=128,
            temperature=0.7
        )
        results = extractor.extract_for_dataset(
            data_path="data/synthetic/prompts_v1.txt",
            output_dir="data/synthetic/rationale_v1",
            batch_size=8
        )
        logger.info(f"Extracted {len(results)} rationales")
        return results

    except Exception as e:
        logger.error(f"Rationale extraction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()