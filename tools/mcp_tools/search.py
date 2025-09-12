import requests
from mcp.server.fastmcp import FastMCP  # 假设您已有这个基础库

mcp = FastMCP("LocalServer")

@mcp.tool()
def query_rag(query: str, topk: int = 3):
    """MCP RAG Query Tool (Synchronous Version)
    
    Args:
        query: query text
        topk: The default number of documents returned is 3
        
    Returns:
        str: The formatted query result
    """
    try:
        # 构建请求数据
        request_data = {
            "queries": [query],
            "topk": topk,
            "return_scores": True
        }
        # 设置请求头和代理
        headers = {
            "Content-Type": "application/json"
        }
        # 使用本地连接，绕过代理
        proxies = {
            "http": None,
            "https": None
        }
        
        response = requests.post(
            "http://127.0.0.1:5003/retrieve",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=10
        )
        
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        
        if not result.get("result"):
            return "⚠️ No relevant documents found"
            
        return result
        
    except requests.exceptions.Timeout:
        return "⚠️ RAG service request timeout, please check if the service is running properly"
    except requests.exceptions.ConnectionError:
        return "⚠️ Unable to connect to RAG service, please ensure that the service is running"
    except requests.exceptions.RequestException as e:
        return f"⚠️ RAG service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') else 'No detail'}"
    except Exception as e:
        return f"⚠️ RAG query failed: {str(e)}\nError type: {type(e).__name__}"


if __name__ == "__main__":
    print("\nStart MCP service:")
    mcp.run(transport='stdio')