import re
import json
from typing import List
import logging
import os

logger = logging.getLogger()
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TOOL_CALL_PATTERN = re.compile(r'<tool_call>\s*({.*?})\s*</tool_call>', re.DOTALL)

def normalize_function_call(name: str, args: dict) -> str:
    """
    Normalize function call format to ensure consistent parameter ordering
    """
    # Method 1: Sort parameters alphabetically (recommended)
    sorted_args = []
    for key in sorted(args.keys()):
        value = args[key]
        sorted_args.append(f"{key}='{value}'")
    
    args_str = ', '.join(sorted_args)
    return f"{name}({args_str})"

def extract_tool_calls(text: str) -> List[str]:
    """
    Extract tool_call from text, only supports format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    """
    tool_calls = []

    try:
        matches = TOOL_CALL_PATTERN.findall(text)
    except Exception as e:
        print(f"Debug: regex error: {e}")
        return tool_calls

    for match in matches:
        try:
            call_data = json.loads(match)
            if 'name' in call_data:
                name = call_data['name']
                args = call_data.get('arguments', {})
                
                # Use unified normalization function
                normalized_call = normalize_function_call(name, args)
                tool_calls.append(normalized_call)
        except json.JSONDecodeError:
            continue
    
    return tool_calls

def normalize_ground_truth_calls(ground_truth_calls: List[str]) -> List[str]:
    """
    Normalize function calls in ground truth
    """
    normalized_calls = []
    
    for call in ground_truth_calls:
        # Parse function call format: func_name(arg1='val1', arg2='val2')
        match = re.match(r'([^(]+)\((.*)\)', call.strip())
        if match:
            func_name = match.group(1)
            args_str = match.group(2)
            
            # Parse parameters
            args = {}
            if args_str.strip():
                # Simple parameter parsing (assuming parameter format is key='value')
                arg_pattern = re.compile(r"(\w+)='([^']*)'")
                for arg_match in arg_pattern.finditer(args_str):
                    key = arg_match.group(1)
                    value = arg_match.group(2)
                    args[key] = value
            
            # Use the same normalization function
            normalized_call = normalize_function_call(func_name, args)
            normalized_calls.append(normalized_call)
        else:
            # If unable to parse, keep as is
            normalized_calls.append(call)
    
    return normalized_calls

def parse_ground_truth(ground_truth: str) -> List[str]:
    """
    parse the ground_truth, support JSON list format
    """
    try:
        if ground_truth.strip().startswith('['):
            calls = json.loads(ground_truth)
        else:
            calls = [line.strip() for line in ground_truth.strip().split('\n') if line.strip()]
        
        # Normalize ground truth calls
        return normalize_ground_truth_calls(calls)
    except json.JSONDecodeError:
        calls = [line.strip() for line in ground_truth.strip().split('\n') if line.strip()]
        return normalize_ground_truth_calls(calls)

def compute_score(solution_str: str, ground_truth: str, format_score: float = 0.1, extra_info=None) -> float:
    """
    calculate the score of solution relative to ground_truth, only return the numerical value
    
    Args:
        solution_str: the solution text containing tool_calls
        ground_truth: the correct answer tool_call sequence
        format_score: the base score when the format is correct
        extra_info: extra information
    
    Returns:
        float: solution==grouth_truch - length_penalty + format_score
    """
    try:
        # 1. extract the tool_calls from solution
        solution_calls = extract_tool_calls(solution_str)
        
        # 2. parse the ground_truth (already normalized)
        ground_truth_calls = parse_ground_truth(ground_truth)

        # 3. if no tool_calls are extracted, return 0
        if not solution_calls:
            return 0.0
        
        # 4. if the ground_truth is empty, return 1
        if not ground_truth_calls:
            return 1.0
        
        # 5. check if the ground_truth is a subsequence of solution
        gt_idx = 0
        for sol_call in solution_calls:
            if gt_idx < len(ground_truth_calls) and sol_call == ground_truth_calls[gt_idx]:
                gt_idx += 1
        
        if gt_idx == len(ground_truth_calls):
            total_score = 1.0
        else:
            total_score = 0.0
        
        # 6. calculate the length penalty
        length_ratio = len(solution_calls) / len(ground_truth_calls) if len(ground_truth_calls) > 0 else 1.0
        length_penalty = 0.0
        
        if length_ratio > 2.0:
            length_penalty = min(0.3, (length_ratio - 2.0) * 0.1)
        elif length_ratio > 1.5:
            length_penalty = min(0.1, (length_ratio - 1.5) * 0.2)
        
        # 7. calculate the final content score(0-1)
        total_score = max(0.0, min(1.0, total_score - length_penalty + format_score))

        return round(total_score, 1)
        
    except Exception as e:
        logger.warning(f"Debug: Error in compute_score: {e}")
        return 0.0
