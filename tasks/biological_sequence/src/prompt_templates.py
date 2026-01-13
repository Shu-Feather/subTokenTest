"""
Prompt templates for biological sequence manipulation tasks.
"""

from typing import Dict, Any

class PromptTemplates:
    """Templates for generating prompts for different biological sequence tasks."""
    
    @staticmethod
    def get_dna_complement_prompt(input_sequence: str, restricted_reasoning: bool = False) -> str:
        """Generate prompt for DNA complement task."""
        prompt = f"""You are a molecular biology expert. I need you to find the complementary DNA sequence for a given DNA sequence.

**Biological Knowledge Required:**
In DNA, the four nucleotide bases pair specifically through hydrogen bonding:
- Adenine (A) always pairs with Thymine (T)
- Thymine (T) always pairs with Adenine (A)
- Guanine (G) always pairs with Cytosine (C)
- Cytosine (C) always pairs with Guanine (G)

This complementary base pairing is fundamental to DNA structure and replication.

**Task:**
Given the following DNA sequence, provide its complementary sequence by applying the base pairing rules above.

**Example:**
Input: ATCG
Output answer: TAGC (A→T, T→A, C→G, G→C)

**Input DNA sequence:** {input_sequence}

**Instructions:**
1. For each base in the input sequence, determine its complement using the pairing rules
2. Write out the complete complementary sequence
3. Provide your final answer between the tags <answer> and </answer>

Please provide the complementary DNA sequence for the given input."""
        if restricted_reasoning:
            prompt += "\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
        return prompt

    @staticmethod
    def get_rna_complement_prompt(input_sequence: str, restricted_reasoning: bool = False) -> str:
        """Generate prompt for RNA complement task."""
        prompt = f"""You are a molecular biology expert. I need you to find the complementary RNA sequence for a given RNA sequence.

**Biological Knowledge Required:**
In RNA, the four nucleotide bases pair specifically through hydrogen bonding:
- Adenine (A) always pairs with Uracil (U)
- Uracil (U) always pairs with Adenine (A)  
- Guanine (G) always pairs with Cytosine (C)
- Cytosine (C) always pairs with Guanine (G)

Note: RNA uses Uracil (U) instead of Thymine (T) found in DNA. This complementary base pairing is crucial for RNA secondary structure and function.

**Task:**
Given the following RNA sequence, provide its complementary sequence by applying the base pairing rules above.

**Example:**
Input: AUCG
Output answer: UAGC (A→U, U→A, C→G, G→C)

**Input RNA sequence:** {input_sequence}

**Instructions:**
1. For each base in the input sequence, determine its complement using the RNA pairing rules
2. Write out the complete complementary sequence
3. Provide your final answer between the tags <answer> and </answer>

Please provide the complementary RNA sequence for the given input."""
        if restricted_reasoning:
            prompt += "\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
        return prompt

    @staticmethod
    def get_protein_three_to_one_prompt(input_sequence: str, restricted_reasoning: bool = False) -> str:
        """Generate prompt for protein three-letter to one-letter conversion."""
        prompt = f"""You are a biochemistry expert. I need you to convert a protein sequence from three-letter amino acid codes to one-letter codes.

**Biological Knowledge Required:**
Each of the 20 standard amino acids has both a three-letter and one-letter abbreviation:

ALA = A (Alanine)      ARG = R (Arginine)     ASN = N (Asparagine)   ASP = D (Aspartic acid)
CYS = C (Cysteine)     GLN = Q (Glutamine)    GLU = E (Glutamic acid) GLY = G (Glycine)
HIS = H (Histidine)    ILE = I (Isoleucine)   LEU = L (Leucine)      LYS = K (Lysine)
MET = M (Methionine)   PHE = F (Phenylalanine) PRO = P (Proline)     SER = S (Serine)
THR = T (Threonine)    TRP = W (Tryptophan)   TYR = Y (Tyrosine)     VAL = V (Valine)

**Task:**
Convert the following protein sequence from three-letter amino acid codes to one-letter codes.

**Example:**
Input: GLY-ARG-PHE
Output answer: GRF

**Input protein sequence (three-letter codes):** {input_sequence}

**Instructions:**
1. Split the input sequence by hyphens to get individual amino acids
2. Convert each three-letter code to its corresponding one-letter code using the table above
3. Join the one-letter codes together WITHOUT any separators (no hyphens, no spaces)
4. Provide your final answer between the tags <answer> and </answer>

Please convert the given protein sequence to one-letter amino acid codes."""
        if restricted_reasoning:
            prompt += "\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
        return prompt

    @staticmethod
    def get_protein_one_to_three_prompt(input_sequence: str, restricted_reasoning: bool = False) -> str:
        """Generate prompt for protein one-letter to three-letter conversion."""
        prompt = f"""You are a biochemistry expert. I need you to convert a protein sequence from one-letter amino acid codes to three-letter codes.

**Biological Knowledge Required:**
Each of the 20 standard amino acids has both a one-letter and three-letter abbreviation:

A = ALA (Alanine)      R = ARG (Arginine)     N = ASN (Asparagine)   D = ASP (Aspartic acid)
C = CYS (Cysteine)     Q = GLN (Glutamine)    E = GLU (Glutamic acid) G = GLY (Glycine)
H = HIS (Histidine)    I = ILE (Isoleucine)   L = LEU (Leucine)      K = LYS (Lysine)
M = MET (Methionine)   F = PHE (Phenylalanine) P = PRO (Proline)     S = SER (Serine)
T = THR (Threonine)    W = TRP (Tryptophan)   Y = TYR (Tyrosine)     V = VAL (Valine)

**Task:**
Convert the following protein sequence from one-letter amino acid codes to three-letter codes.

**Example:**
Input: GRF
Output answer: GLY-ARG-PHE

**Input protein sequence (one-letter codes):** {input_sequence}

**Instructions:**
1. The input sequence contains individual amino acid letters with no separators
2. Convert each one-letter code to its corresponding three-letter code using the table above
3. Join the three-letter codes with hyphens as separators
4. Provide your final answer between the tags <answer> and </answer>

Please convert the given protein sequence to three-letter amino acid codes."""
        if restricted_reasoning:
            prompt += "\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
        return prompt

    @staticmethod
    def get_prompt_for_task(task_type: str, input_sequence: str, restricted_reasoning: bool = False) -> str:
        """
        Get the appropriate prompt for a given task type.
        
        Args:
            task_type: Type of task
            input_sequence: Input sequence for the task
            
        Returns:
            Formatted prompt string
        """
        prompt_functions = {
            'dna_complement': PromptTemplates.get_dna_complement_prompt,
            'rna_complement': PromptTemplates.get_rna_complement_prompt,
            'protein_three_to_one': PromptTemplates.get_protein_three_to_one_prompt,
            'protein_one_to_three': PromptTemplates.get_protein_one_to_three_prompt
        }
        
        if task_type not in prompt_functions:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return prompt_functions[task_type](input_sequence, restricted_reasoning=restricted_reasoning)

    @staticmethod
    def extract_answer_from_response(response: str) -> str:
        """
        Extract answer from model response between <answer> tags.
        
        Args:
            response: Model's response string
            
        Returns:
            Extracted answer or original response if tags not found
        """
        start_tag = "<answer>"
        end_tag = "</answer>"
        
        start_idx = response.find(start_tag)
        end_idx = response.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            answer = response[start_idx + len(start_tag):end_idx].strip()
            return answer
        
        # If tags not found, try to find the most likely answer
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines that might contain the answer
            if line and not line.startswith(('**', 'Input:', 'Output:', 'Example:')):
                # Check if line contains typical sequence patterns
                if any(char in line for char in ['A', 'T', 'G', 'C', 'U']) or '-' in line:
                    return line
        
        return response.strip()


def main():
    """Example usage of prompt templates."""
    templates = PromptTemplates()
    
    # Test each prompt type
    print("DNA Complement Prompt:")
    print(templates.get_dna_complement_prompt("ATCG"))
    print("\n" + "="*50 + "\n")
    
    print("RNA Complement Prompt:")
    print(templates.get_rna_complement_prompt("AUCG"))
    print("\n" + "="*50 + "\n")
    
    print("Protein 3→1 Prompt:")
    print(templates.get_protein_three_to_one_prompt("GLY-ARG-PHE"))
    print("\n" + "="*50 + "\n")
    
    print("Protein 1→3 Prompt:")
    print(templates.get_protein_one_to_three_prompt("GRF"))  # Updated format


if __name__ == "__main__":
    main()
