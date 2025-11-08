import re
from bs4 import BeautifulSoup
from trafilatura import extract as trafilatura_extract, extract_metadata
from readability import Document
from boilerpy3 import extractors
from trafilatura.settings import use_config
import html as html_lib
from typing import Optional, Dict, List, Tuple
import unicodedata
from difflib import SequenceMatcher
from bs4 import Tag

class HTMLContentExtractor:
    def __init__(self, site_configs: Optional[Dict[str, dict]] = None):
        self.extractor = extractors.ArticleExtractor()
        if site_configs is None:
            self.site_configs = {
            'nytimes.com': {
                'cookies': {'nyt-gdpr': 'accept'},
                'headers': {'Referer': 'https://www.google.com/'},
                'wait_time': 2,
                'selectors': ['article#story', 'div.StoryBodyCompanionColumn'],
                'remove_selectors': ['div.ad', 'div.comments', 'div.toolbar']
            },
            'reuters.com': {
                'cookies': {'reuters-gdpr': 'accept'},
                'headers': {'Referer': 'https://news.google.com/'},
                'wait_time': 1,
                'selectors': ['article.article-body', 'div[data-testid="Body"]'],
                'remove_selectors': ['div.article-header', 'div.article-share']
            },
            'washingtonpost.com': {
                'cookies': {'wp_gdpr': 'accept'},
                'headers': {'Referer': 'https://www.google.com/'},
                'wait_time': 2,
                'selectors': ['div.article-body', 'article.main-content'],
                'remove_selectors': ['div.interstitial', 'div.newsletter-inline-unit']
            },
            'economist.com': {
                'cookies': {
                    'economist-logged-in': 'true',
                    'economist-analytics': 'disable'
                },
                'headers': {
                    'Referer': 'https://www.google.com/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-User': '?1',
                    'Sec-Fetch-Dest': 'document'
                },
                'selectors': [
                    'article.article__body',
                    'div.article__lead',
                    'div.article__content',
                    'p.article__body-text'
                ],
                'remove_selectors': [
                    'div.advert',
                    'div.article__footnote',
                    'div.share-links',
                    'div.newsletter-signup'
                ]
            },
            'baltimoresun.com': {
                'cookies': {'bsun-consent': 'accept'},
                'headers': {
                    'Referer': 'https://www.google.com/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                },
                'wait_time': 2,
                'selectors': [
                    'div.body-copy',
                    'article.story'
                ],
                'remove_selectors': [
                    'div.map-container',
                    'div#president',
                    'div#senate',
                    'div.time',
                    'p.wp-remixd-voice-wrapper',
                    'p[id^="container-embed"]'
                ]
            },
            'bloomberg.com': {
                'cookies': {
                    'bb_geo_info': '{"country":"US","region":"NY"}',
                    'bb_consent': 'express',
                    'bb_subscriber': 'true'
                },
                'headers': {
                    'Referer': 'https://www.google.com/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-User': '?1',
                    'Sec-Fetch-Dest': 'document',
                    'Accept-Language': 'en-US,en;q=0.9'
                },
                'wait_time': 3,
                'selectors': [
                    'div.body-content',
                    'div.body-copy',
                    'article.article-body'
                ],
                'remove_selectors': [
                    'div.paywall',
                    'div.newsletter-signup',
                    'div.related-articles'
                ]
            },
            'channelnewsasia.com': {
                'cookies': {'cna-consent': 'accept'},
                'headers': {
                    'Referer': 'https://www.google.com/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
                'wait_time': 2,
                'selectors': [
                    'div.article-content',
                    'div.article-body',
                    'article',
                    'div.article-text',
                    'div.text-long'
                ],
                'remove_selectors': [
                    'div.advertisement',
                    'div.teaser',
                    'div.partner-recommendations',
                    'div.social-share',
                    'div.also-worth-reading',
                    'div.related-topics',
                    'div[class*="partner"]'
                ]
            },
            'cna.com.sg': {
                'cookies': {'cna-consent': 'accept'},
                'headers': {
                    'Referer': 'https://www.google.com/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
                'wait_time': 2,
                'selectors': [
                    'div.article-content',
                    'div.article-body',
                    'article',
                    'div.article-text',
                    'div.text-long'
                ],
                'remove_selectors': [
                    'div.advertisement',
                    'div.teaser',
                    'div.partner-recommendations',
                    'div.social-share',
                    'div.also-worth-reading',
                    'div[class*="partner"]'
                ]
            },
            'techcrunch.com': {
                'cookies': {'tc-consent': 'accept'},
                'headers': {
                    'Referer': 'https://www.google.com/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                },
                'wait_time': 2,
                'selectors': [
                    'div.article-content',
                    'p.wp-block-paragraph',
                    'ul.wp-block-list',
                    'li.wp-block-list-item',
                    'h2.wp-block-heading',
                    'h3.wp-block-heading'
                ],
                'remove_selectors': [
                    'div[class*="ad"]',
                    'div[class*="related"]',
                    'div[class*="promo"]',
                    'div[class*="footer"]',
                    'div[class*="author"]',
                    'aside',
                    'nav',
                    'header'
                ]
            }
        }
        else:
            self.site_configs = site_configs
            
        self.blacklist_patterns = [
            # Basic advertisement patterns
            r'Advertisement\s*',
            r'ADVERTISEMENT\s*',
            r'Recommended\s*',
            r'###Embeddable###',
            
            # Subscription and newsletter patterns
            r'Subscribe.*?(?=\n|$)',
            r'Sign up for.*?(?=\n|$)',
            r'Sign up here\.?',
            r'Newsletter.*?(?=\n|$)',
            r'Support quality journalism.*?(?=\n|$)',
            
            # Copyright and attribution patterns
            r'©\s*\d{4}.*?(?=\n|$)',
            r'All rights reserved.*?(?=\n|$)',
            r'PUBLISHED.*?, \d{2}:\d{2}.*?(?=\n|$)',
            r'PHOTO:.*?(?=\n|$)',
            
            # Social sharing patterns
            r'Follow us on.*?(?=\n|$)',
            r'Share this article.*?(?=\n|$)',
            r'Thanks for sharing!.*?(?=\n|$)',
            r'Share full article.*?(?=\n|$)',
            
            # Navigation and related content
            r'Read more:.*?(?=\n|$)',
            r'More on this Topic.*?(?=\n|$)',
            r'See more on.*?(?=\n|$)',
            
            # Other common patterns
            r'\[\s*\d+\s*chars\s*\]',
            r'^\s*https?://[^\s]+$',  # Matches only lines with pure URLs,
        ]

        
        # Additional removal terms
        self.removal_keywords = [
            'embeddable', 'subscribe', 'subscription', 'newsletter', 'sign up',
            'share this', 'follow us', 'copyright', 'all rights reserved',
            'read more', 'more on this', 'see more', 'gallery', 'slideshow',
            'click here', 'download', 'register', 'privacy policy', 'terms of service',
            'thanks for sharing', 'photo:', 'image:', 'published', 'updated'
        ]

    def _preprocess_html(self, html_content: str) -> str:
        """Pre-process HTML to improve extraction results."""
        try:
            # Remove common problematic elements
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script, style, and SVG tags
            for tag in soup.find_all(['script', 'style', 'svg', 'noscript']):
                tag.decompose()
                
            # Remove hidden elements
            for tag in soup.find_all(style=re.compile(r'display:\s*none|visibility:\s*hidden')):
                tag.decompose()
            
            # Handle specific pattern issues
            for div in soup.find_all('div', {'id': re.compile(r'footer|banner|sidebar|comment')}):
                div.decompose()
                
            # Keep header elements that might contain article titles
            for div in soup.find_all('div', {'id': re.compile(r'header')}):
                # Only remove if not likely to contain main content
                header_text = div.get_text()
                if len(header_text) < 200 or re.search(r'(menu|navigation|logo|sign in)', header_text.lower()):
                    div.decompose()
                
            return str(soup)
        except Exception:
            return html_content

    def extract(self, url: str, html_content: str) -> Optional[str]:
        if not html_content or len(html_content.strip()) < 100:
            return None

        html_content = self._preprocess_html(html_content)
        metadata = self.get_article_metadata(html_content, url)
        soup = BeautifulSoup(html_content, 'html.parser')

        cleaned_results = []

        # Site-specific selectors (if applicable)
        site_specific = self._extract_with_selectors(html_content, url)
        if site_specific and len(site_specific.strip()) > 500:
            cleaned_results.append((site_specific, 1.2, 'site-specific'))

        # Trafilatura
        trafilatura_result = self._extract_trafilatura(html_content)
        if trafilatura_result:
            cleaned_results.append((trafilatura_result, 1.0, 'trafilatura'))

        # Readability
        try:
            doc = Document(html_content)
            readability_soup = BeautifulSoup(doc.summary(), 'html.parser')
            readability_text = self._fallback_extract_paragraphs(readability_soup)
            if readability_text:
                cleaned_results.append((readability_text, 0.9, 'readability'))
        except Exception as e:
            print(f"[ERROR] Readability failed: {e}")

        # BoilerPy (optional)
        try:
            boilerpy_result = self._extract_boilerpy(html_content)
            if boilerpy_result:
                cleaned_results.append((boilerpy_result, 0.8, 'boilerpy'))
        except Exception:
            pass

        if cleaned_results:
            scored_results = []
            for content, base_weight, label in cleaned_results:
                score = self._calculate_content_quality(content)
                length_score = min(len(content) / 5000, 1.0)
                total_score = base_weight * score * length_score
                scored_results.append((content, total_score, label))

            best_result, _, label = max(scored_results, key=lambda x: x[1])
            print(f"[DEBUG] Selected content from {label} with length {len(best_result)}")
            return self._format_with_metadata(self._clean_content(best_result), metadata)

        # Fallback: auto-detect main content div
        guessed_div = self._guess_main_content_div(soup)
        if guessed_div:
            text = guessed_div.get_text(separator=" ", strip=True)
            if text:
                return self._format_with_metadata(self._clean_content(text), metadata)

        # Fallback: collect all <p> and <li> elements
        fallback = self._fallback_extract_paragraphs(soup)
        if fallback:
            return self._format_with_metadata(self._clean_content(fallback), metadata)

        return None

    def _extract_with_selectors(self, html_content: str, url: str) -> Optional[str]:
        """Extract content using custom CSS selectors for known sites."""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        config = {}

        for known_domain in self.site_configs:
            if known_domain in domain:
                config = self.site_configs[known_domain]
                break

        selectors = config.get("selectors", [])
        remove_selectors = config.get("remove_selectors", [])

        if not selectors:
            selectors = [
                'article', '.article', '.article-body', '.content', '.entry-content',
                'div[itemprop="articleBody"]', '.story-body', '.post-content', 'main'
            ]

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            for selector in remove_selectors:
                for element in soup.select(selector):
                    element.decompose()

            for tag in soup.find_all(['aside', 'nav', 'footer']):
                tag.decompose()

            for tag in soup.find_all(class_=re.compile(r'(ad-|banner|promo|sponsored|recommendation)')):
                tag.decompose()

            content_parts = []

            for selector in selectors:
                for element in soup.select(selector):
                    self._clean_element(element)

                    paragraphs = []
                    children = list(element.children)
                    i = 0
                    while i < len(children):
                        child = children[i]
                        if not isinstance(child, Tag):
                            i += 1
                            continue

                        tag_name = child.name
                        text = child.get_text(separator=" ", strip=True)

                        if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and text:
                            merged_text = [text]
                            j = i + 1
                            while j < len(children):
                                next_child = children[j]
                                if isinstance(next_child, Tag) and next_child.name in ['p', 'div', 'ul', 'ol']:
                                    para = next_child.get_text(separator=" ", strip=True)
                                    if para and len(para) > 30 and not self._is_boilerplate(para):
                                        merged_text.append(para)
                                        i = j
                                        break
                                j += 1
                            paragraphs.append("\n".join(merged_text))
                        elif tag_name in ['p', 'li'] and text and len(text) > 40 and not self._is_boilerplate(text):
                            paragraphs.append(text)
                        i += 1

                    if paragraphs:
                        content_parts.append('\n\n'.join(paragraphs))

            if not content_parts:
                paragraphs = []
                for p in soup.find_all(['p']):
                    text = p.get_text(separator = " ", strip=True)
                    if text and len(text) > 20 and not self._is_boilerplate(text):
                        paragraphs.append(text)
                if paragraphs:
                    return '\n\n'.join(paragraphs)
                return None

            return '\n\n'.join(content_parts)
        except Exception as e:
            print(f"Error in _extract_with_selectors: {e}")
            return None

    def _clean_element(self, element):
        """Clean an element's internal structure."""
        # Remove common clutter tags
        for tag in element.find_all(['script', 'style', 'button', 'form', 'input', 'iframe']):
            tag.decompose()
            
        # Remove share buttons and related elements
        for tag in element.find_all(class_=re.compile(r'share|social|comment|related|promo|ad|subscribe|newsletter')):
            tag.decompose()
            
        # Remove empty paragraphs
        for p in element.find_all('p'):
            if not p.get_text(separator = " ", strip=True) or len(p.get_text(separator = " ", strip=True)) < 5:
                p.decompose()
                
    def _is_boilerplate(self, text: str) -> bool:
        """Check if text is likely boilerplate content."""
        text_lower = text.lower()
        
        # Check against removal keywords
        if any(keyword in text_lower for keyword in self.removal_keywords):
            return True
            
        # Check for common boilerplate patterns
        if (len(text) < 20 and 
            (text.isupper() or 
             text.count('.') == 0 or 
             re.search(r'^\d+:\d+$', text) or
             re.search(r'^[^\w]*https?://', text))):
            return True
            
        return False
    

    def _extract_trafilatura(self, html_content: str) -> Optional[str]:
        try:
            config = use_config()
            if not config.has_section("EXTRACTION"):
                config.add_section("EXTRACTION")
            config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
            config.set("EXTRACTION", "favor_precision", "true")

            result = trafilatura_extract(html_content, config=config)
            return result if result and len(result) > 300 else None
        except Exception as e:
            print(f"[ERROR] Trafilatura failed: {e}")
            return None
        
    def _guess_main_content_div(self, soup: BeautifulSoup) -> Optional[Tag]:
        candidates = []
        for div in soup.find_all('div'):
            class_attr = ' '.join(div.get('class', []))
            if any(kw in class_attr for kw in ['footer', 'nav', 'sidebar', 'header', 'promo', 'share']):
                continue
            text = div.get_text(separator=' ', strip=True)
            p_count = len(div.find_all('p'))
            if len(text) > 500 and p_count >= 3:
                candidates.append((div, len(text)))

        if candidates:
            return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
        return None
    
    def _fallback_extract_paragraphs(self, soup: BeautifulSoup) -> Optional[str]:
        paragraphs = []
        for tag in soup.find_all(['p', 'li', 'h2', 'h3']):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) > 40 and not self._is_boilerplate(text):
                paragraphs.append(text)
        return '\n\n'.join(paragraphs) if paragraphs else None


        

    def _extract_readability(self, html_content: str) -> Optional[str]:
        """Extract content using readability."""
        try:
            doc = Document(html_content)
            title = doc.title()
            summary_html = doc.summary()
            
            # Process with BeautifulSoup for better formatting
            soup = BeautifulSoup(summary_html, 'html.parser')
            
            # Remove clutter
            for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
                
            for tag in soup.find_all(class_=re.compile(r'ads|share|comment|social|promo|related')):
                tag.decompose()
                
            # Extract paragraphs with proper spacing, including all possible content containers
            paragraphs = []
            
            # First try to get all paragraph-like elements
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = element.get_text(separator = " ", strip=True)
                if text and len(text) > 15 and not self._is_boilerplate(text):
                    paragraphs.append(text)
            
            # If we didn't find enough content, try extracting div elements directly
            if len(''.join(paragraphs)) < 1000:
                for div in soup.find_all('div'):
                    # Only process divs that don't have nested paragraphs (to avoid duplication)
                    if not div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        text = div.get_text(separator = " ", strip=True)
                        if text and len(text) > 50 and not self._is_boilerplate(text):
                            # Split into pseudo-paragraphs for better readability
                            for section in re.split(r'(?<=[.!?])\s+', text):
                                if len(section) > 40:  # Only add meaningful sections
                                    paragraphs.append(section)
                    
            if paragraphs:
                content = '\n\n'.join(paragraphs)
                if title and title not in content[:len(title)+20]:
                    return f"{title}\n\n{content}"
                return content
            return None
        except Exception as e:
            print(f"Error in _extract_readability: {e}")
            return None

    def _extract_boilerpy(self, html_content: str) -> Optional[str]:
        """Extract content using boilerpy3."""
        try:
            content = self.extractor.get_content(html_content)
            if content:
                lines = content.split('\n')
                # Filter out likely boilerplate
                filtered_lines = [line for line in lines if len(line) > 20 and not self._is_boilerplate(line)]
                return '\n\n'.join(filtered_lines) if filtered_lines else None
            return None
        except Exception:
            return None

    def _calculate_content_quality(self, content: str) -> float:
        """Calculate a quality score for the content."""
        if not content:
            return 0.0
            
        # Some quality indicators
        avg_line_length = sum(len(line) for line in content.splitlines()) / max(1, len(content.splitlines()))
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        word_count = len(content.split())
        
        # Higher score for content with more sentences and reasonable line lengths
        quality_score = min(1.0, (sentence_count / max(1, word_count / 20)) * 
                          (min(avg_line_length, 100) / 100))
        
        # Penalize content with too many non-alphanumeric characters
        alphanumeric_ratio = sum(c.isalnum() or c.isspace() for c in content) / max(1, len(content))
        quality_score *= max(0.5, alphanumeric_ratio)
        
        # Penalize content with too many uppercase characters
        uppercase_ratio = sum(c.isupper() for c in content) / max(1, len(content) - content.count(' '))
        if uppercase_ratio > 0.3:
            quality_score *= 0.8
            
        return quality_score

    def _clean_content(self, content: str) -> str:
        if not content:
            return ""

        import html as html_lib
        content = html_lib.unescape(content)

        # Remove invisible/control characters
        content = re.sub(r'[\x00-\x1F\x7F]', ' ', content)

        # Fix punctuation spacing issues
        content = re.sub(r'\s+([.,;!?])', r'\1', content)
        content = re.sub(r'([.,;!?])(?=\w)', r'\1 ', content)


        # Fix joined words
        content = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', content)
        content = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', content)
        content = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', content)
     

        # Normalize spacing
        content = re.sub(r' {2,}', ' ', content)
        content = re.sub(r'\n{3,}', '\n\n', content)


        for i, pattern in enumerate(self.blacklist_patterns):
            before = len(content)
            content = re.sub(pattern, ' ', content, flags=re.IGNORECASE | re.MULTILINE)
            after = len(content)

        # Deduplication disabled for debug
        # content = self._deduplicate_content(content, keep_longer=True)

        # Remove heuristic that splits content on sentence boundaries (temporarily)
        # if content.count('\n') < 3 and len(content) > 1000:
        #     parts = re.split(r'(?<=[.!?]) +(?=[A-Z])', content)
        #     content = '\n\n'.join(parts)

        # Final cleanup
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()
        
        return content
    
    def _deduplicate_content(self, content: str, keep_longer: bool = False) -> str:
        """Only remove exact line duplicates. Avoid using SequenceMatcher."""
        lines = content.splitlines()
        seen = set()
        result = []
        
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                result.append(stripped)
        
        return '\n\n'.join(result)

    def get_article_metadata(self, html_content: str, url: str) -> Dict:
        """Extract metadata from the article."""
        try:
            metadata = {}
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get title
            metadata['title'] = self._extract_title(soup)
            
            # Get author
            metadata['author'] = self._extract_author(soup)
            
            # Get publication date
            metadata['date'] = self._extract_date(soup)
            
            # Get source/publication
            metadata['source'] = self._extract_source(soup, url)
            
            return {k: v for k, v in metadata.items() if v}  # Remove empty values
        except Exception:
            return {}
            
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the article title."""
        # Try meta tags first
        for meta in soup.find_all('meta'):
            if meta.get('property') in ['og:title', 'twitter:title'] and meta.get('content'):
                return meta.get('content').strip()
                
        # Try h1 tags
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(separator = " ", strip=True)
            
        # Fallback to title tag
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(separator = " ", strip=True)
            
        return None
        
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the author name."""
        # Try meta tags
        for meta in soup.find_all('meta'):
            if meta.get('name') in ['author', 'article:author'] and meta.get('content'):
                return meta.get('content').strip()
                
        # Try common author selectors
        for selector in ['.author', '.byline', '.writer', '[rel="author"]']:
            author_elem = soup.select_one(selector)
            if author_elem:
                return author_elem.get_text(separator = " ", strip=True)
                
        return None
        
    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date."""
        # Try meta tags
        for meta in soup.find_all('meta'):
            if meta.get('property') in ['article:published_time', 'og:published_time'] and meta.get('content'):
                return meta.get('content').strip()
                
        # Try time tags
        time_elem = soup.find('time')
        if time_elem and time_elem.get('datetime'):
            return time_elem.get('datetime')
            
        # Try common date selectors
        for selector in ['.date', '.published', '.timestamp', '.pubdate']:
            date_elem = soup.select_one(selector)
            if date_elem:
                return date_elem.get_text(separator = " ", strip=True)
                
        return None
        
    def _extract_source(self, soup: BeautifulSoup, url: str) -> str:
        """Extract publication source name."""
        # Try meta tags
        for meta in soup.find_all('meta'):
            if meta.get('property') in ['og:site_name'] and meta.get('content'):
                return meta.get('content').strip()
                
        # Extract from URL
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        # Remove www. and .com/.org
        source = domain.replace('www.', '')
        source = re.sub(r'\.(com|org|net|gov|edu|io)$', '', source)
        return source.capitalize()

    def _format_with_metadata(self, content: str, metadata: Dict) -> str:
        """Format content with metadata header."""
        print(f"[DEBUG] Metadata formatting for content length: {len(content)}")
        if not content:
            return ""
            
        # Check if content already has title at the beginning
        has_title = False
        if 'title' in metadata and metadata['title']:
            title_start = metadata['title'][:30].lower()
            content_start = content[:100].lower()
            has_title = title_start in content_start
            
        header_parts = []
        
        # Add title if not already in content
        if 'title' in metadata and metadata['title'] and not has_title:
            header_parts.append(f"# {metadata['title']}")
        
        meta_parts = []
        
        # Add source
        if 'source' in metadata and metadata['source']:
            meta_parts.append(f"Content source: {metadata['source']}")
            
        # Add date
        if 'date' in metadata and metadata['date']:
            # Try to format date in a more readable way if possible
            try:
                from datetime import datetime
                import dateutil.parser
                parsed_date = dateutil.parser.parse(metadata['date'])
                formatted_date = parsed_date.strftime("%B %d, %Y")
                meta_parts.append(f"Date: {formatted_date}")
            except:
                meta_parts.append(f"Date: {metadata['date']}")
                
        # Add author
        if 'author' in metadata and metadata['author']:
            meta_parts.append(f"Author: {metadata['author']}")
            
        # Add metadata section if we have any metadata
        if meta_parts:
            header_parts.append("\n".join(meta_parts))
            
        # If we have header content, add it to the beginning
        if header_parts:
            return "\n\n".join(header_parts + [content])
        
        return content

    def _extract_cna_article(self, html_content: str) -> Optional[str]:
        """Special extraction for CNA (Channel News Asia) articles."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove clutter first
            for element in soup.select('.advertisement, .teaser, .partner-recommendations, div[class*="recommendation"], div[class*="partner"], .also-worth-reading'):
                element.decompose()
            
            # Try to find the main article content
            content_sections = []
            
            # Strategy 1: Find the main article container
            main_content = soup.select_one('article') or soup.select_one('div.article-body-wrapper') or soup.select_one('div.text-long')
            
            if main_content:
                # Extract paragraphs from the main content
                paragraphs = []
                for p in main_content.find_all('p'):
                    text = p.get_text(separator = " ", strip=True)
                    if text and len(text) > 15 and not self._is_boilerplate(text):
                        paragraphs.append(text)
                
                if paragraphs:
                    content_sections.append('\n\n'.join(paragraphs))
            
            # Strategy 2: Direct paragraph extraction (may include IMPACT ON SINGAPORE and other section headers)
            if not content_sections:
                paragraphs = []
                for p in soup.find_all('p'):
                    text = p.get_text(separator = " ", strip=True)
                    if text and len(text) > 15 and not self._is_boilerplate(text):
                        if text.isupper() and len(text) < 40:  # Likely a section header
                            paragraphs.append(f"\n{text}\n")
                        else:
                            paragraphs.append(text)
                
                if paragraphs:
                    content_sections.append('\n\n'.join(paragraphs))
            
            # Strategy 3: Look for Related content sections (like "Related:" followed by links)
            related_sections = soup.find_all(string=re.compile(r'Related:'))
            for related in related_sections:
                related_element = related.parent
                if related_element:
                    related_links = []
                    # Find all links within or after the related section
                    for link in related_element.find_all('a'):
                        link_text = link.get_text(separator = " ", strip=True)
                        if link_text and len(link_text) > 10:
                            related_links.append(f"- {link_text}")
                    
                    if related_links:
                        content_sections.append("\nRelated Articles:\n" + "\n".join(related_links))
            
            if content_sections:
                return '\n\n'.join(content_sections)
            
            return None
        except Exception as e:
            print(f"Error in _extract_cna_article: {e}")
            return None