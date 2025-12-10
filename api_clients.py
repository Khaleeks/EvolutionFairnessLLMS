"""
API Client Wrappers - Together AI + Google Gemini + OpenAI (with exponential backoff)
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

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

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
        max_retries: int = 5
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
                if '405B' in self.exp_config.model_name:
                    response = self.client.chat.completions.create(
                        model=self.exp_config.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.exp_config.temperature,
                        max_tokens=self.exp_config.max_tokens
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
                
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    import re
                    
                    pred_match = re.search(r'prediction_label[:\s]+([^,\n]+)', content, re.IGNORECASE)
                    reason_match = re.search(r'reasoning[:\s]+(.+)', content, re.IGNORECASE | re.DOTALL)
                    
                    if pred_match:
                        pred = pred_match.group(1).strip().strip('"').strip("'")
                        reason = reason_match.group(1).strip() if reason_match else "No reasoning"
                        
                        result = {
                            "prediction_label": pred,
                            "reasoning": reason[:200]
                        }
                    else:
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
                        print(f"{record_id}: {error_msg}")
                    continue
                    
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {str(e)}"
                prediction_record['error_message'] = error_msg
                if attempt == max_retries - 1:
                    print(f"{record_id}: {error_msg}")
                else:
                    time.sleep(1)
                continue
                
            except Exception as e:
                error_str = str(e)
                
                if '503' in error_str or '429' in error_str or 'overload' in error_str.lower():
                    wait_time = 2 ** (attempt + 1)
                    error_msg = f"API overloaded (attempt {attempt + 1}/{max_retries})"
                    prediction_record['error_message'] = error_msg
                    
                    if attempt < max_retries - 1:
                        print(f"{record_id}: Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"{record_id}: {error_msg} - giving up")
                        return prediction_record
                else:
                    error_msg = f"API error: {error_str}"
                    prediction_record['error_message'] = error_msg
                    if attempt == max_retries - 1:
                        print(f"{record_id}: {error_msg}")
                    else:
                        time.sleep(1)
                    continue
        
        return prediction_record

# ============================================================================
# GOOGLE GEMINI CLIENT
# ============================================================================

class GeminiClient(BaseAPIClient):
    """Google Gemini client wrapper with exponential backoff"""
    
    def __init__(self, api_key: str, exp_config: ExperimentConfig):
        super().__init__(api_key, exp_config)
        
        if genai is None:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        self.generation_config = {
            "temperature": exp_config.temperature,
            "max_output_tokens": exp_config.max_tokens,
            "response_mime_type": "application/json",
        }
        
        self.model = genai.GenerativeModel(
            model_name=exp_config.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        self.rate_limit_delay = 0.5
        self.last_request_time = 0
        
    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        record_id: str,
        max_retries: int = 5
    ) -> Dict[str, Any]:
        """Classify using Google Gemini API with exponential backoff"""
        
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
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        for attempt in range(max_retries):
            prediction_record['attempts_made'] = attempt + 1
            prediction_record['timestamp'] = datetime.now().isoformat()
            
            if attempt == 0:
                time_since_last = time.time() - self.last_request_time
                if time_since_last < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - time_since_last
                    time.sleep(sleep_time)
            
            try:
                response = self.model.generate_content(full_prompt)
                self.last_request_time = time.time()
                
                if not response.candidates:
                    error_msg = "Response blocked by safety filters (no candidates)"
                    prediction_record['error_message'] = error_msg
                    print(f"{record_id}: Safety filter blocked response")
                    return prediction_record
                
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    if candidate.finish_reason == 3:
                        error_msg = "Response blocked by safety filters (finish_reason=SAFETY)"
                        prediction_record['error_message'] = error_msg
                        print(f"{record_id}: Safety filter blocked response")
                        return prediction_record
                    elif candidate.finish_reason == 2:
                        print(candidate.finish_message)
                        error_msg = "Response blocked (finish_reason=2, likely safety)"
                        prediction_record['error_message'] = error_msg
                        print(f"{record_id}: Response blocked (finish_reason=2)")
                        return prediction_record
                
                try:
                    content = response.text
                except ValueError as ve:
                    error_msg = f"No valid response parts: {str(ve)}"
                    prediction_record['error_message'] = error_msg
                    print(f"{record_id}: {error_msg}")
                    return prediction_record
                
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    import re
                    
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(1))
                    else:
                        pred_match = re.search(r'prediction_label["\s:]+([^,\n"]+)', content, re.IGNORECASE)
                        reason_match = re.search(r'reasoning["\s:]+(.+?)(?=\n|$|")', content, re.IGNORECASE | re.DOTALL)
                        
                        if pred_match:
                            pred = pred_match.group(1).strip().strip('"').strip("'")
                            reason = reason_match.group(1).strip() if reason_match else "No reasoning"
                            
                            result = {
                                "prediction_label": pred,
                                "reasoning": reason[:200]
                            }
                        else:
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
                        print(f"{record_id}: {error_msg}")
                    continue
                    
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {str(e)}"
                prediction_record['error_message'] = error_msg
                if attempt == max_retries - 1:
                    print(f"{record_id}: {error_msg}")
                else:
                    time.sleep(1)
                continue
                
            except Exception as e:
                error_str = str(e)
                
                if attempt == 0:
                    print(f"{record_id}: Error details: {error_str[:200]}")
                
                if '429' in error_str or 'Resource has been exhausted' in error_str:
                    wait_time = 2 ** (attempt + 1)
                    error_msg = f"API rate limited (attempt {attempt + 1}/{max_retries})"
                    prediction_record['error_message'] = error_msg
                    
                    if attempt < max_retries - 1:
                        print(f"{record_id}: Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"{record_id}: {error_msg} - giving up")
                        return prediction_record
                
                elif 'finish_reason' in error_str or 'safety' in error_str.lower() or 'blocked' in error_str.lower():
                    error_msg = "Content blocked by safety filters"
                    prediction_record['error_message'] = error_msg
                    print(f"{record_id}: {error_msg}")
                    return prediction_record
                
                elif 'safety' in error_str.lower() or 'blocked' in error_str.lower():
                    error_msg = f"Content filtered by safety settings: {error_str}"
                    prediction_record['error_message'] = error_msg
                    print(f"{record_id}: {error_msg}")
                    return prediction_record
                
                else:
                    error_msg = f"API error: {error_str}"
                    prediction_record['error_message'] = error_msg
                    if attempt == max_retries - 1:
                        print(f"{record_id}: {error_msg}")
                    else:
                        time.sleep(1)
                    continue
        
        return prediction_record

# ============================================================================
# OPENAI CLIENT
# ============================================================================

class OpenAIClient(BaseAPIClient):
    """OpenAI client wrapper with exponential backoff"""
    
    def __init__(self, api_key: str, exp_config: ExperimentConfig):
        super().__init__(api_key, exp_config)
        
        if OpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        
    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        record_id: str,
        max_retries: int = 5
    ) -> Dict[str, Any]:
        """Classify using OpenAI API with exponential backoff"""
        
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
                
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    import re
                    
                    pred_match = re.search(r'prediction_label[:\s]+([^,\n]+)', content, re.IGNORECASE)
                    reason_match = re.search(r'reasoning[:\s]+(.+)', content, re.IGNORECASE | re.DOTALL)
                    
                    if pred_match:
                        pred = pred_match.group(1).strip().strip('"').strip("'")
                        reason = reason_match.group(1).strip() if reason_match else "No reasoning"
                        
                        result = {
                            "prediction_label": pred,
                            "reasoning": reason[:200]
                        }
                    else:
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
                        print(f"{record_id}: {error_msg}")
                    continue
                    
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {str(e)}"
                prediction_record['error_message'] = error_msg
                if attempt == max_retries - 1:
                    print(f"{record_id}: {error_msg}")
                else:
                    time.sleep(1)
                continue
                
            except Exception as e:
                error_str = str(e)
                
                if '429' in error_str or 'rate_limit' in error_str.lower():
                    wait_time = 2 ** (attempt + 1)
                    error_msg = f"API rate limited (attempt {attempt + 1}/{max_retries})"
                    prediction_record['error_message'] = error_msg
                    
                    if attempt < max_retries - 1:
                        print(f"{record_id}: Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"{record_id}: {error_msg} - giving up")
                        return prediction_record
                else:
                    error_msg = f"API error: {error_str}"
                    prediction_record['error_message'] = error_msg
                    if attempt == max_retries - 1:
                        print(f"{record_id}: {error_msg}")
                    else:
                        time.sleep(1)
                    continue
        
        return prediction_record

# ============================================================================
# CLIENT FACTORY
# ============================================================================

def create_client(exp_config: ExperimentConfig) -> BaseAPIClient:
    """
    Factory function to create API client based on provider
    
    Args:
        exp_config: Experiment configuration
    
    Returns:
        Initialized API client
    
    Raises:
        ValueError: If API key not found or provider unknown
    """
    
    if exp_config.api_provider == APIProvider.TOGETHER:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        return TogetherClient(api_key, exp_config)
    
    elif exp_config.api_provider == APIProvider.GEMINI:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        return GeminiClient(api_key, exp_config)
    
    elif exp_config.api_provider == APIProvider.OPENAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIClient(api_key, exp_config)
    
    else:
        raise ValueError(f"Unknown API provider: {exp_config.api_provider}")