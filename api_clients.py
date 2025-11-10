"""
API Client Wrappers - Together AI Only (with exponential backoff)
fairness_experiments/api_clients.py
"""

import os
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from together import Together
except ImportError:
    Together = None

from config import APIProvider, ExperimentConfig

# ============================================================================
# BASE CLIENT INTERFACE
# ============================================================================

class BaseAPIClient:
    """Base class for API clients"""
    
    def __init__(self, api_key: str, exp_config: ExperimentConfig):
        self.api_key = api_key
        self.exp_config = exp_config
        
    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        record_id: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Classify a single record
        
        Returns prediction record with exact schema:
        {
            'record_id': str,
            'experiment_id': str,
            'prediction_label': str | None,
            'reasoning': str | None,
            'api_success': bool,
            'attempts_made': int,
            'timestamp': str,
            'error_message': str | None
        }
        """
        raise NotImplementedError

# ============================================================================
# TOGETHER AI CLIENT
# ============================================================================

class TogetherClient(BaseAPIClient):
    """Together AI client wrapper with exponential backoff"""
    
    def __init__(self, api_key: str, exp_config: ExperimentConfig):
        super().__init__(api_key, exp_config)
        
        if Together is None:
            raise ImportError("together package not installed. Run: pip install together")
        
        self.client = Together(api_key=api_key)
        
    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        record_id: str,
        max_retries: int = 5  # Increased from 3
    ) -> Dict[str, Any]:
        """Classify using Together AI API with exponential backoff"""
        
        prediction_record = {
            'record_id': record_id,
            'experiment_id': self.exp_config.experiment_id,
            'prediction_label': None,
            'reasoning': None,
            'api_success': False,
            'attempts_made': 0,
            'timestamp': None,
            'error_message': None
        }
        
        for attempt in range(max_retries):
            prediction_record['attempts_made'] = attempt + 1
            prediction_record['timestamp'] = datetime.now().isoformat()
            
            try:
                # 405B model has JSON formatting issues - don't force JSON for it
                if '405B' in self.exp_config.model_name:
                    response = self.client.chat.completions.create(
                        model=self.exp_config.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.exp_config.temperature,
                        max_tokens=self.exp_config.max_tokens
                        # No response_format - 405B doesn't handle it well
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.exp_config.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.exp_config.temperature,
                        max_tokens=self.exp_config.max_tokens,
                        response_format={"type": "json_object"}
                    )
                
                content = response.choices[0].message.content
                
                # Try to parse JSON first
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback: try to fix common JSON issues for 405B
                    # Replace "prediction_label: value" with {"prediction_label": "value"}
                    import re
                    
                    # Try to extract prediction_label and reasoning
                    pred_match = re.search(r'prediction_label[:\s]+([^,\n]+)', content, re.IGNORECASE)
                    reason_match = re.search(r'reasoning[:\s]+(.+)', content, re.IGNORECASE | re.DOTALL)
                    
                    if pred_match:
                        pred = pred_match.group(1).strip().strip('"').strip("'")
                        reason = reason_match.group(1).strip() if reason_match else "No reasoning"
                        
                        # Create a proper dict
                        result = {
                            "prediction_label": pred,
                            "reasoning": reason[:200]  # Limit reasoning length
                        }
                    else:
                        # Couldn't parse at all
                        raise json.JSONDecodeError("Could not parse response", content, 0)
                
                prediction = result.get("prediction_label", "").strip().lower()
                reasoning = result.get("reasoning", "No reasoning provided")
                
                if prediction:
                    prediction_record['prediction_label'] = prediction
                    prediction_record['reasoning'] = reasoning
                    prediction_record['api_success'] = True
                    return prediction_record
                else:
                    error_msg = f"Empty prediction (keys: {list(result.keys())})"
                    prediction_record['error_message'] = error_msg
                    if attempt == max_retries - 1:
                        print(f"  ⚠️  {record_id}: {error_msg}")
                    continue
                    
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {str(e)}"
                prediction_record['error_message'] = error_msg
                if attempt == max_retries - 1:
                    print(f"  ⚠️  {record_id}: {error_msg}")
                else:
                    # Small delay for JSON errors
                    time.sleep(1)
                continue
                
            except Exception as e:
                error_str = str(e)
                
                # Check for rate limit or server overload (503, 429, etc.)
                if '503' in error_str or '429' in error_str or 'overload' in error_str.lower():
                    # Exponential backoff: 2, 4, 8, 16, 32 seconds
                    wait_time = 2 ** (attempt + 1)
                    error_msg = f"API overloaded (attempt {attempt + 1}/{max_retries})"
                    prediction_record['error_message'] = error_msg
                    
                    if attempt < max_retries - 1:
                        print(f"  ⏳ {record_id}: Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  ⚠️  {record_id}: {error_msg} - giving up")
                        return prediction_record
                else:
                    # Other errors - fail fast
                    error_msg = f"API error: {error_str}"
                    prediction_record['error_message'] = error_msg
                    if attempt == max_retries - 1:
                        print(f"  ⚠️  {record_id}: {error_msg}")
                    else:
                        time.sleep(1)  # Small delay
                    continue
        
        return prediction_record

# ============================================================================
# CLIENT FACTORY
# ============================================================================

def create_client(exp_config: ExperimentConfig) -> BaseAPIClient:
    """
    Factory function to create Together AI client
    
    Args:
        exp_config: Experiment configuration
    
    Returns:
        Initialized API client
    
    Raises:
        ValueError: If API key not found
    """
    
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    
    return TogetherClient(api_key, exp_config)