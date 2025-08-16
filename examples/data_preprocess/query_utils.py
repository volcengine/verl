"""
Shared utilities for query understanding and processing
"""


def create_advanced_query_understanding_prompt(query_text):
    """Create a comprehensive prompt for query understanding and reformulation.

    Args:
        query_text (str): The input query to be processed

    Returns:
        str: The formatted prompt for the language model
    """
    prompt = f'''You are a query understanding system that converts natural language questions into specific, direct answers.

Your job is to identify what the user is really looking for and provide the most likely, specific answer they would want to see.

Query Analysis Framework:
1. FACTUAL QUERIES: Provide the direct fact/answer
2. RECOMMENDATION QUERIES: Give the most popular/recommended option
3. DEFINITION QUERIES: Provide the specific term or concept
4. LOCATION QUERIES: Give specific place names
5. COMPARISON QUERIES: Provide the most commonly preferred option

Response Rules:
- Be specific, not generic
- Prioritize well-known, popular, or commonly expected answers
- Use proper nouns when applicable
- Keep it concise (1-5 words maximum)
- Avoid explanatory text

Examples by category:

FACTUAL:
Query: "what is the capital of France" → "Paris"
Query: "who invented the telephone" → "Alexander Graham Bell"

RECOMMENDATIONS:
Query: "what is a good restaurant type for a date" → "Italian restaurant"
Query: "what programming language should I learn first" → "Python"

LOCATIONS:
Query: "where is a good place to visit in Japan" → "Tokyo"
Query: "what is a famous beach in California" → "Malibu"

DEFINITIONS:
Query: "what is the main ingredient in guacamole" → "Avocado"
Query: "what is used to make bread rise" → "Yeast"

Now process this query: "{query_text}"

#### [Your answer]:'''
    return prompt.strip()
