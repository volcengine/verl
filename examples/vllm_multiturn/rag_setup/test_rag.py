import requests
import json
import time
import asyncio
import aiohttp
from datetime import datetime

# Server configuration
SERVER_URL = "http://localhost:2223/retrieve"

# Test queries
test_queries = [
    ["What is machine learning?", "How does deep learning work?"],
    ["Explain retrieval augmented generation", "What are embedding models?"]
]

async def send_request(session, query_list, request_id):
    """Send a request to the RAG server and measure response time"""
    start_time = time.time()
    
    payload = {
        "queries": query_list,
        "topk_retrieval": 10,
        "topk_rerank": 3,
        "return_scores": True
    }
    
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Request {request_id} sending: {query_list}")
    
    try:
        async with session.post(SERVER_URL, json=payload) as response:
            elapsed = time.time() - start_time
            response_json = await response.json()
            
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Request {request_id} completed in {elapsed:.4f}s")
            print(f"Request {request_id} server processing time: {response_json['processing_time']:.4f}s")
            
            # Print first result title for each query (just to verify response content)
            for i, query_results in enumerate(response_json['result']):
                if query_results:
                    print(f"Request {request_id}, Query '{query_list[i]}' top result: '{query_results[0]['title']}'")
            
            return response_json, elapsed
    except Exception as e:
        print(f"Error with request {request_id}: {str(e)}")
        return None, elapsed

async def main():
    """Run concurrent requests to the RAG server"""
    print("\n=== Testing RAG Server Concurrent Requests ===\n")
    
    # First check if server is healthy
    try:
        health_check = requests.get("http://localhost:2223/health")
        health_data = health_check.json()
        print(f"Server health: {health_data['status']}")
        print(f"Pipeline loaded: {health_data['pipeline_loaded']}")
        print(f"Device: {health_data['device']}")
        print()
    except Exception as e:
        print(f"Error connecting to server: {str(e)}")
        print("Make sure the RAG server is running at http://localhost:2223")
        return
    
    async with aiohttp.ClientSession() as session:
        # Create tasks for concurrent execution
        tasks = []
        for i, queries in enumerate(test_queries):
            tasks.append(send_request(session, queries, i + 1))
        
        # Execute all tasks concurrently and gather results
        print("\n--- Starting concurrent requests ---\n")
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print("\n--- Results Summary ---\n")
        print(f"Total execution time for all requests: {total_time:.4f}s")
        
        for i, (response, elapsed) in enumerate(results):
            if response:
                request_id = i + 1
                print(f"Request {request_id} client-side time: {elapsed:.4f}s")
                print(f"Request {request_id} server processing time: {response['processing_time']:.4f}s")
        
        # Calculate time difference or overlap
        if len(results) >= 2:
            time_diff = abs(results[0][1] - results[1][1])
            print(f"\nTime difference between requests: {time_diff:.4f}s")
            
            if total_time < (results[0][1] + results[1][1]):
                overlap = (results[0][1] + results[1][1]) - total_time
                print(f"Requests overlapped by approximately: {overlap:.4f}s")
                print(f"Parallelization efficiency: {(results[0][1] + results[1][1]) / total_time:.2f}x")

if __name__ == "__main__":
    asyncio.run(main())