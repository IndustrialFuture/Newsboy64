async def google_search_and_scrape(query, is_news, question_details, date_before=None):
    """
    Performs Google search and scrapes/summarizes articles.
    Uses Jina Reader as fallback when FastContentExtractor fails.
    """
    write(f"[google_search_and_scrape] Called with query='{query}', is_news={is_news}, date_before={date_before}")
    try:
        urls = await google_search(query, is_news, date_before)

        if not urls:
            write(f"[google_search_and_scrape] ❌ No URLs returned for query: '{query}'")
            return f"<Summary query=\"{query}\">No URLs returned from Google.</Summary>\n"

        # TRY FastContentExtractor first
        async with FastContentExtractor() as extractor:
            write(f"[google_search_and_scrape] 🔍 Starting content extraction for {len(urls)} URLs")
            results = await extractor.extract_content(urls)
            write(f"[google_search_and_scrape] ✅ Finished content extraction")

        # NEW: Check if FastContentExtractor failed, use Jina as fallback
        failed_urls = []
        for url in urls:
            if url not in results or not results[url].get('content', '').strip():
                failed_urls.append(url)
        
        if failed_urls:
            write(f"[google_search_and_scrape] 🔄 FastContentExtractor failed for {len(failed_urls)} URLs, trying Jina fallback")
            jina_tasks = [fetch_with_jina(url) for url in failed_urls[:3]]  # Limit to 3 to avoid rate limits
            jina_results = await asyncio.gather(*jina_tasks)
            
            for url, jina_result in zip(failed_urls[:3], jina_results):
                if jina_result.get('success'):
                    results[url] = jina_result
                    write(f"[google_search_and_scrape] ✅ Jina fallback succeeded for {url}")

        summarize_tasks = []
        no_results = 3
        valid_urls = []
        for url, data in results.items():
            if len(summarize_tasks) >= no_results:
                break  
            content = (data.get('content') or '').strip()
            if len(content.split()) < 100:
                write(f"[google_search_and_scrape] ⚠️ Skipping low-content article: {url}")
                continue
            if content:
                truncated = content[:8000]
                write(f"[google_search_and_scrape] ✂️ Truncated content for summarization: {len(truncated)} chars from {url}")
                summarize_tasks.append(
                    asyncio.create_task(summarize_article(truncated, question_details))
                )
                valid_urls.append(url)

        if not summarize_tasks:
            write("[google_search_and_scrape] ⚠️ Warning: No content to summarize (all extraction methods failed)")
            return f"<Summary query=\"{query}\">No usable content extracted from any URL.</Summary>\n"

        summaries = await asyncio.gather(*summarize_tasks, return_exceptions=True)

        output = ""
        for url, summary in zip(valid_urls, summaries):
            if isinstance(summary, Exception):
                write(f"[google_search_and_scrape] ❌ Error summarizing {url}: {summary}")
                output += f"\n<Summary source=\"{url}\">\nError summarizing content: {str(summary)}\n</Summary>\n"
            else:
                output += f"\n<Summary source=\"{url}\">\n{summary}\n</Summary>\n"

        return output
    except Exception as e:
        write(f"[google_search_and_scrape] Error: {str(e)}")
        traceback_str = traceback.format_exc()
        write(f"Traceback: {traceback_str}")
        return f"<Summary query=\"{query}\">Error during search and scrape: {str(e)}</Summary>\n"
