"""
Response logger for saving detailed model responses and token usage.
Location: /src/response_logger.py
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading


class ResponseLogger:
    """Logger for saving detailed model responses to JSON file"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize response logger
        
        Args:
            log_file: Path to JSON file for saving logs
        """
        self.log_file = log_file
        self.logs: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        # Create directory if it doesn't exist
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
    
    def log_response(
        self,
        model_name: str,
        board_size: int,
        test_case_index: int,
        board_representation: str,
        model_response: str,
        extracted_answer: Any,
        expected_answer: Any,
        is_correct: bool,
        usage_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single model response
        
        Args:
            model_name: Name of the model
            board_size: Size of the game board
            test_case_index: Index of the test case
            board_representation: String representation of the board
            model_response: Raw response from the model
            extracted_answer: Answer extracted by the pipeline
            expected_answer: Ground truth answer
            is_correct: Whether the extracted answer matches expected
            usage_info: Token usage information
            metadata: Additional metadata
        """
        if not self.log_file:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "board_size": board_size,
            "test_case_index": test_case_index,
            "board_representation": board_representation,
            "model_response": model_response,
            "extracted_answer": extracted_answer,
            "expected_answer": expected_answer,
            "is_correct": is_correct,
            "token_usage": {
                "total_tokens": usage_info.get("total_tokens", 0) if usage_info else 0,
                "prompt_tokens": usage_info.get("prompt_tokens", 0) if usage_info else 0,
                "completion_tokens": usage_info.get("completion_tokens", 0) if usage_info else 0,
                "reasoning_tokens": usage_info.get("reasoning_tokens", 0) if usage_info else 0,
                "output_tokens": usage_info.get("output_tokens", 0) if usage_info else 0,
            }
        }
        
        # Add raw_usage for debugging if available
        if usage_info and "raw_usage" in usage_info and usage_info["raw_usage"]:
            log_entry["token_usage"]["raw_usage"] = usage_info["raw_usage"]
        
        if metadata:
            log_entry["metadata"] = metadata
        
        with self.lock:
            self.logs.append(log_entry)
    
    def save(self):
        """Save all logs to the JSON file"""
        if not self.log_file:
            return
        
        try:
            with self.lock:
                # Custom JSON encoder to handle non-serializable objects
                def default_encoder(obj):
                    if hasattr(obj, 'isoformat'):
                        return obj.isoformat()
                    return str(obj)
                
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(self.logs, f, indent=2, ensure_ascii=False, default=default_encoder)
            print(f"Response logs saved to: {self.log_file}")
        except Exception as e:
            print(f"Error saving response logs: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged responses"""
        with self.lock:
            total_cases = len(self.logs)
            if total_cases == 0:
                return {}
            
            correct_cases = sum(1 for log in self.logs if log.get("is_correct", False))
            
            total_tokens = sum(log["token_usage"]["total_tokens"] for log in self.logs)
            total_reasoning = sum(log["token_usage"]["reasoning_tokens"] for log in self.logs)
            total_output = sum(log["token_usage"]["output_tokens"] for log in self.logs)
            
            return {
                "total_cases": total_cases,
                "correct_cases": correct_cases,
                "accuracy": correct_cases / total_cases if total_cases > 0 else 0,
                "total_tokens_used": total_tokens,
                "total_reasoning_tokens": total_reasoning,
                "total_output_tokens": total_output,
                "avg_tokens_per_case": total_tokens / total_cases if total_cases > 0 else 0,
                "avg_reasoning_tokens_per_case": total_reasoning / total_cases if total_cases > 0 else 0,
                "avg_output_tokens_per_case": total_output / total_cases if total_cases > 0 else 0,
            }
    
    def clear(self):
        """Clear all logs"""
        with self.lock:
            self.logs.clear()