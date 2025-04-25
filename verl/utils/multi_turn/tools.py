import requests


# we can register our own tool here
supported_tools = {}

def register_tools(name=None):
    def wrapper(func):
        tool_name = func.__name__ if name is None else name
        assert tool_name not in supported_tools, f"Tool {tool_name} already exists, please choose another name."
        supported_tools[tool_name] = func
        return func
    return wrapper


@register_tools('answer')
def answer(response_str):
    return response_str, True


@register_tools('search')
def search(response_str):
    """
    search for query.
    Args:
        query: query to call the search engine
    Returns:
        search results which is concatenated into a string
    """
    # search_url = f"https://www.google.com/search?q={response_str}"
    # ret = requests.get(search_url)
    ret = f'工具调用({response_str})'
    info_str = f'\n\n<information>{ret}</information>\n\n'
    return info_str, False


