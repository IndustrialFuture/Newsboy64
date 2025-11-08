from urllib.parse import urlparse
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Tuple
from HTMLContentExtractor import HTMLContentExtractor
import dotenv
import os
from browser import fetch_full_html

dotenv.load_dotenv()

API_KEY = os.getenv("BRIGHT_DATA_API_KEY")

class FastContentExtractor:
    def __init__(self, api_key: str = API_KEY, 
                 zone: str = "web_scraper"):
        self.api_key = api_key
        self.zone = zone
        self.api_url = "https://api.brightdata.com/request"
        self.html_extractor = HTMLContentExtractor()

    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def process_urls(self, urls: List[str]) -> Dict[str, Tuple[str, bool]]:
        results_dict = await self.extract_content(urls)
        processed_results = {}
        
        for url, result in results_dict.items():
            content = result.get('content', '')
            success = result.get('success', False)
            processed_results[url] = (content, success)
            
        return processed_results

    async def _fetch_url(self, url: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "url": url,
                "zone": self.zone,
                "format": "raw",
            }
            
            # Create a timeout for the request
            timeout = aiohttp.ClientTimeout(total=20)  # 20 second timeout for the whole operation
            
            async with session.post(self.api_url, headers=headers, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    print(f"Error: API returned status {response.status} for {url}")
                    return {
                        'url': url,
                        'domain': urlparse(url).netloc,
                        'raw_html': None,
                        'content': None,
                        'error': f"API error: {response.status}",
                        'success': False
                    }
                
                raw_html = await response.text()
                backup_html = await asyncio.to_thread(fetch_full_html, url)

                if not raw_html or len(raw_html.strip()) < 1400:
                    print(f"Error: Received empty or very short HTML for {url}: " + raw_html)
                    if backup_html and len(backup_html.strip() > 2000):
                        print(f"Using backup HTML for url: {url}")
                        raw_html = backup_html
                    else:  
                        return {
                            'url': url,
                            'domain': urlparse(url).netloc,
                            'raw_html': raw_html,
                            'content': "Empty or very short HTML received: " + raw_html,
                            'error': "Empty or very short HTML received",
                            'success': False
                        }
                    
                # Extract content using HTMLContentExtractor
                processed_content = self.html_extractor.extract(url, raw_html)
                
                if not processed_content:
                    print(f"Warning: Failed to extract content for {url}")
                    return {
                        'url': url,
                        'domain': urlparse(url).netloc,
                        'raw_html': raw_html,
                        'content': None,
                        'error': "Content extraction failed",
                        'success': False
                    }
                
                domain = urlparse(url).netloc
                print(f"Successfully extracted {len(processed_content)} characters from {url}")
                return {
                    'url': url,
                    'domain': domain,
                    'raw_html': raw_html,
                    'content': processed_content,
                    'success': True
                }
        except asyncio.TimeoutError:
            print(f"Timeout error for {url}")
            return {
                'url': url,
                'domain': urlparse(url).netloc,
                'raw_html': None,
                'content': None,
                'error': "Request timed out",
                'success': False
            }
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return {
                'url': url,
                'domain': urlparse(url).netloc,
                'raw_html': None,
                'content': None,
                'error': str(e),
                'success': False
            }

    async def extract_content(self, urls: List[str]) -> Dict[str, Any]:
        results = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Create tasks explicitly as required by asyncio
                tasks = [asyncio.create_task(self._fetch_url(url, session)) for url in urls]
                
                # Set a timeout for the entire operation
                timeout = 75  # 75 seconds total for all requests
                
                # Wait for tasks with timeout
                done, pending = await asyncio.wait(tasks, timeout=timeout)
                
                # Handle completed tasks
                for task in done:
                    try:
                        result = task.result()
                        url = result['url']
                        results[url] = result
                    except Exception as e:
                        print(f"Error getting task result: {str(e)}")
                
                # Handle pending tasks (timed out)
                for task in pending:
                    task.cancel()  # Cancel any pending tasks
                    try:
                        # Find the index of the task in our list (if possible)
                        task_index = None
                        for i, t in enumerate(tasks):
                            if t == task:
                                task_index = i
                                break
                        
                        # Get the URL if we can find it
                        url = urls[task_index] if task_index is not None else "unknown URL"
                        print(f"Task for {url} timed out and was cancelled")
                        results[url] = {
                            'url': url,
                            'domain': urlparse(url).netloc if url != "unknown URL" else "unknown",
                            'raw_html': None,
                            'content': None,
                            'error': "Operation timed out",
                            'success': False
                        }
                    except Exception as e:
                        print(f"Error handling cancelled task: {str(e)}")
        
        except Exception as e:
            print(f"Error in extract_content: {str(e)}")
        
        return results






