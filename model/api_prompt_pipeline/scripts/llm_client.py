# scripts/llm_client.py

import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import API_BASE_URL, API_TIMEOUT

def create_robust_session(retries=3, backoff_factor=1, status_forcelist=(500, 502, 503, 504)):
    """
    Create a requests Session with automatic retry logic.
    Useful for handling transient network errors or temporary server overloads.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(['POST'])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def call_llm_extraction(payload, api_key):
    """
    Execute a single LLM API call using the SiliconFlow (OpenAI-compatible) endpoint.
    
    Args:
        payload (dict): The JSON payload containing model, messages, etc.
        api_key (str): The specific API key to use for this request.
        
    Returns:
        dict: Standardized response dictionary {'status': '...', 'data': ...}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    session = create_robust_session()

    try:
        # Send Request
        response = session.post(
            API_BASE_URL, 
            json=payload, 
            headers=headers, 
            timeout=API_TIMEOUT
        )
        
        # Check for HTTP Errors (4xx, 5xx)
        # Note: 429 errors will raise an exception here, which is caught below
        response.raise_for_status() 
        
        # Parse Response
        result = response.json()
        
        # Check for API-level logic errors (if any)
        if 'error' in result:
             return {
                "status": "error",
                "error_msg": f"API Business Error: {result['error']}"
            }

        # Extract Content (Standard OpenAI Format)
        if 'choices' in result and len(result['choices']) > 0:
            content_str = result['choices'][0]['message']['content']
        else:
            return {
                "status": "error",
                "error_msg": "Invalid Response Structure (No choices found)",
                "raw_response": str(result)
            }
        
        # --- Clean Markdown Formatting ---
        # LLMs often wrap JSON in ```json ... ``` blocks. We must strip this.
        cleaned_str = content_str.strip()
        if cleaned_str.startswith("```"):
            # Remove the first line (e.g., ```json)
            first_newline = cleaned_str.find("\n")
            if first_newline != -1:
                cleaned_str = cleaned_str[first_newline+1:]
            
            # Remove the last line (```)
            if cleaned_str.endswith("```"):
                cleaned_str = cleaned_str[:-3]
        
        cleaned_str = cleaned_str.strip()

        return {
            "status": "success",
            "data": json.loads(cleaned_str),
            "raw_response": content_str # Keep original for debugging if needed
        }

    except requests.exceptions.HTTPError as e:
        # Handle specific HTTP errors
        status_code = e.response.status_code if e.response else 0
        return {
            "status": "error",
            "error_msg": f"HTTP {status_code} Error: {str(e)}",
            "raw_response": e.response.text if e.response else ""
        }
    except requests.exceptions.SSLError as e:
        return {
            "status": "error",
            "error_msg": f"SSL Certificate Error: {str(e)}"
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "status": "error",
            "error_msg": f"Connection Failed: {str(e)}"
        }
    except requests.exceptions.Timeout as e:
        return {
            "status": "error",
            "error_msg": f"Request Timed Out: {str(e)}"
        }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error_msg": "JSON Parsing Failed (LLM output was not valid JSON)",
            "raw_response": content_str if 'content_str' in locals() else "No Content"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_msg": f"Unexpected Error: {str(e)}"
        }
    finally:
        session.close()