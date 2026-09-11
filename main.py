import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


MODEL = "qwen-local"

MAX_SEARCH_RESULTS = 3
MAX_SEARCH_BODY = 150
MAX_SCRAPE_CHARS = 1800
MAX_SUMMARY_CHARS = 2500
MAX_COMPARE_SOURCE_CHARS = 1800
MAX_REPORT_SOURCE_CHARS = 7000
MAX_TOOL_RESULT_CHARS = 3500
MAX_ITERATIONS = 6


def search_web(query):
    results = []

    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                max_results=MAX_SEARCH_RESULTS
            )

            for result in search_results:
                title = result.get("title", "")
                url = result.get("href", "")
                body = result.get("body", "")

                if url:
                    results.append(
                        {
                            "title": title,
                            "url": url,
                            "body": body[:MAX_SEARCH_BODY]
                        }
                    )

    except Exception as error:
        return {
            "error": f"Web search error: {error}"
        }

    return results


def scrape_page(url):
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for element in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header"
            ]
        ):
            element.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        if not text:
            return {
                "error": "No readable text was found on this page."
            }

        return {
            "url": url,
            "content": text[:MAX_SCRAPE_CHARS]
        }

    except Exception as error:
        return {
            "error": f"Could not scrape page: {error}"
        }


def summarize(text):
    if not text or not text.strip():
        return {
            "error": "No text was provided for summarization."
        }

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a research summarization tool.

Summarize the provided research text clearly and accurately.

Rules:
- Use only information from the provided text.
- Do not invent facts.
- Keep important facts.
- Keep important names, dates, numbers, and claims.
- Remove unnecessary repetition.
- Make the summary concise.
"""
                },
                {
                    "role": "user",
                    "content": text[:MAX_SUMMARY_CHARS]
                }
            ]
        )

        return {
            "summary": response["message"]["content"]
        }

    except Exception as error:
        return {
            "error": f"Summarization error: {error}"
        }


def compare_sources(source1, source2):
    if not source1 or not source2:
        return {
            "error": "Two research sources are required for comparison."
        }

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a research comparison tool.

Compare the two research sources provided below.

Identify:

- Similarities
- Differences
- Important points mentioned by both sources
- Important points mentioned by only one source

Important rules:

- Use ONLY the provided source content.
- Do NOT use outside knowledge.
- Do NOT browse the URLs.
- Do NOT assume that a URL is the source content.
- Do NOT invent facts.
- If the available source content is limited, clearly say so.
- Keep the comparison clear and concise.
"""
                },
                {
                    "role": "user",
                    "content": f"""
SOURCE 1:

{source1[:MAX_COMPARE_SOURCE_CHARS]}

SOURCE 2:

{source2[:MAX_COMPARE_SOURCE_CHARS]}
"""
                }
            ]
        )

        return {
            "comparison": response["message"]["content"]
        }

    except Exception as error:
        return {
            "error": f"Comparison error: {error}"
        }


def generate_report(sources):
    if not sources or not sources.strip():
        return {
            "error": "No research sources were provided."
        }

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a professional AI research report generator.

Create a clear and structured research report.

The report must contain:

# Research Report

## Introduction

## Findings

## Comparison

## Conclusion

## Sources

Important rules:

- Use ONLY the research information provided.
- Do NOT use outside knowledge.
- Do NOT invent facts.
- Do NOT invent statistics.
- Do NOT invent dates.
- Do NOT invent organizations.
- Do NOT invent sources.
- Do NOT create fake URLs.
- Keep source names and URLs exactly as provided.
- Clearly separate information from different sources.
- Use the comparison information when provided.
- If information is missing, do not guess.
- Keep the report professional and concise.
"""
                },
                {
                    "role": "user",
                    "content": sources[:MAX_REPORT_SOURCE_CHARS]
                }
            ]
        )

        return {
            "report": response["message"]["content"]
        }

    except Exception as error:
        return {
            "error": f"Report generation error: {error}"
        }


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current and relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_page",
            "description": "Read and extract useful text from a web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page."
                    }
                },
                "required": [
                    "url"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarize research text clearly and accurately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The research text to summarize."
                    }
                },
                "required": [
                    "text"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_sources",
            "description": "Compare two research sources using their actual scraped content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source1": {
                        "type": "string",
                        "description": "First research source content."
                    },
                    "source2": {
                        "type": "string",
                        "description": "Second research source content."
                    }
                },
                "required": [
                    "source1",
                    "source2"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a structured research report from collected research.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "string",
                        "description": "Collected research information from web sources and comparison results."
                    }
                },
                "required": [
                    "sources"
                ]
            }
        }
    }
]


system_message = """
You are an AI Research Assistant.

You have five tools:

1. search_web
2. scrape_page
3. summarize
4. compare_sources
5. generate_report

Your job is to research the user's topic efficiently.

IMPORTANT WORKFLOW RULES:

- Use search_web to find useful sources.
- For a normal research task, two good sources are enough.
- After finding two useful sources, stop searching unless both sources are clearly unusable.
- Use scrape_page to read useful sources.
- Once two useful sources have been read, do not search for more sources.
- If the user explicitly asks for comparison, use compare_sources.
- If the user asks for a research report, use generate_report after collecting enough information.
- Do not use every tool just because it is available.
- Use only the tools that are actually useful.
- Do not invent information.
- Use information returned by the tools.
- Never invent sources, URLs, organizations, research papers, models, statistics, dates, or facts.
- If a requested entity cannot be verified from reliable search results, clearly say that reliable information could not be found.
- Never create a fake source to satisfy the user's requested number of sources.
- Treat search results as evidence, not as proof that every claim is true.
- Prefer identifiable and credible sources.
- When a tool returns an error, understand the error and decide whether another available tool can help.
- Do not stop the entire program just because one tool fails.

COMPARISON RULE:

- When comparing sources, use the actual content collected by scrape_page.
- Do not compare URLs.
- Do not assume that URLs contain the source information.
- The program may provide the scraped source content automatically.

REPORT RULE:

- When generate_report is called and enough research has already been collected, generate the report.
- After generate_report succeeds, the research task is COMPLETE.
- Do not search again after generating the report.
- Do not scrape again after generating the report.
- Do not summarize again after generating the report.
- Do not compare again after generating the report.
"""


def execute_tool(tool_name, arguments):
    if tool_name == "search_web":
        return search_web(
            arguments.get("query", "")
        )

    if tool_name == "scrape_page":
        return scrape_page(
            arguments.get("url", "")
        )

    if tool_name == "summarize":
        return summarize(
            arguments.get("text", "")
        )

    if tool_name == "compare_sources":
        return compare_sources(
            arguments.get("source1", ""),
            arguments.get("source2", "")
        )

    if tool_name == "generate_report":
        return generate_report(
            arguments.get("sources", "")
        )

    return {
        "error": f"Unknown tool: {tool_name}"
    }


def run_agent(user_prompt):
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    scraped_sources = []
    comparison_results = []

    for iteration in range(
        1,
        MAX_ITERATIONS + 1
    ):
        print()
        print(
            f"--- Iteration {iteration} ---"
        )

        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=tools
            )

        except Exception as error:
            print()
            print(
                "--- OLLAMA ERROR ---"
            )
            print(error)

            return {
                "error": str(error)
            }

        assistant_message = response["message"]

        tool_calls = assistant_message.get(
            "tool_calls"
        )

        if not tool_calls:
            final_answer = assistant_message.get(
                "content",
                ""
            )

            print()
            print(
                "--- FINAL ANSWER ---"
            )
            print(final_answer)

            return final_answer

        messages.append(
            assistant_message
        )

        report_generated = False
        report_result = None

        for tool_call in tool_calls:
            function = tool_call["function"]

            tool_name = function["name"]

            arguments = function.get(
                "arguments",
                {}
            )

            print(
                f"Tool call: {tool_name}"
            )

            print(
                f"Arguments: {arguments}"
            )

            if tool_name == "compare_sources":
                if len(scraped_sources) >= 2:
                    first_source = scraped_sources[0]
                    second_source = scraped_sources[1]

                    first_content = first_source.get(
                        "content",
                        ""
                    )

                    second_content = second_source.get(
                        "content",
                        ""
                    )

                    tool_result = compare_sources(
                        first_content,
                        second_content
                    )
                else:
                    tool_result = {
                        "error": (
                            "At least two successfully "
                            "scraped sources are required "
                            "before comparison."
                        )
                    }

            elif tool_name == "generate_report":
                research_package_parts = []

                for source in scraped_sources:
                    source_url = source.get(
                        "url",
                        ""
                    )

                    source_content = source.get(
                        "content",
                        ""
                    )

                    research_package_parts.append(
                        f"""
SOURCE URL:
{source_url}

SOURCE CONTENT:
{source_content}
"""
                    )

                for comparison in comparison_results:
                    research_package_parts.append(
                        f"""
COMPARISON:

{comparison}
"""
                    )

                research_package = "\n".join(
                    research_package_parts
                )

                if not research_package.strip():
                    research_package = arguments.get(
                        "sources",
                        ""
                    )

                tool_result = generate_report(
                    research_package
                )

            else:
                tool_result = execute_tool(
                    tool_name,
                    arguments
                )

            if tool_name == "scrape_page":
                if (
                    isinstance(tool_result, dict)
                    and "content" in tool_result
                    and "url" in tool_result
                ):
                    scraped_sources.append(
                        {
                            "url": tool_result["url"],
                            "content": tool_result["content"]
                        }
                    )

            if tool_name == "compare_sources":
                if (
                    isinstance(tool_result, dict)
                    and "comparison" in tool_result
                ):
                    comparison_results.append(
                        tool_result["comparison"]
                    )

            tool_result_text = str(
                tool_result
            )

            if len(tool_result_text) > MAX_TOOL_RESULT_CHARS:
                tool_result_text = (
                    tool_result_text[
                        :MAX_TOOL_RESULT_CHARS
                    ]
                    + "\n[Tool result truncated]"
                )

            print(
                "Tool executed successfully."
            )

            print(
                f"Tool result preview: "
                f"{tool_result_text[:500]}"
            )

            messages.append(
                {
                    "role": "tool",
                    "content": tool_result_text
                }
            )

            if (
                tool_name == "generate_report"
                and isinstance(tool_result, dict)
                and "report" in tool_result
            ):
                report_generated = True
                report_result = tool_result["report"]

        if report_generated:
            print()
            print(
                "--- FINAL ANSWER ---"
            )
            print(report_result)

            return report_result

        messages.append(
            {
                "role": "user",
                "content": (
                    "Continue the research task "
                    "using the tool results above. "
                    "Follow the user's original request. "
                    "If a research report is requested "
                    "and enough information has been "
                    "collected, generate the report."
                )
            }
        )

    print()
    print(
        "--- MAX ITERATIONS REACHED ---"
    )

    return {
        "error": (
            "The research process reached "
            "the maximum number of iterations."
        )
    }


if __name__ == "__main__":
    print(
        "AI Research Assistant"
    )

    print(
        "===================="
    )

    print()

    print(
        "Ollama model:",
        MODEL
    )

    print()

    user_prompt = input(
        "Enter your research question: "
    )

    if not user_prompt.strip():
        print(
            "Please enter a research question."
        )
    else:
        run_agent(
            user_prompt
        )