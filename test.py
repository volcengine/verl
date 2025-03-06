#encoding=utf-8
import httpx
import asyncio
import json

async def start_instance(instance_hash):
    url = "http://60.165.239.98:5000/start_instance"
    headers = {"Content-Type": "application/json"}
    payload = {"instance_hash": instance_hash}
    
    try:
        async with httpx.AsyncClient() as client:
            print(payload)
            response = await client.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            print(f"Start instance response: {json.dumps(result, indent=2)}")
            return result
    except Exception as e:
        print(f"Start instance API call failed: {e}")
        return None

async def process_action(sid, content):
    url = "http://60.165.239.98:5000/process_action"
    headers = {"Content-Type": "application/json"}
    payload = {"sid": sid, "content": content}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            print(f"Process action response: {json.dumps(result, indent=2)}")
            return result
    except Exception as e:
        print(f"Process action API call failed: {e}")
        return None

async def main():
    instance_hash = "3864552457764042195"
    sid = "5671949450826943757"
    instance_result = await start_instance(instance_hash)
    if instance_result:
        sid = instance_result.get("sid", sid)
    
    # Create a script that performs a more time-consuming operation
    content1 = """Test execute_command function

<function=execute_bash>
<parameter=command>echo "
import time
import math

# Calculate prime numbers up to 100000
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Sleep for a moment
time.sleep(3)

# Find prime numbers and their sum
prime_sum = 0
prime_count = 0
for num in range(2, 5):
    if is_prime(num):
        prime_sum += num
        prime_count += 1

print(f'Found {prime_count} prime numbers with a sum of {prime_sum}')
" > test.py</parameter>
</function>
"""
    await process_action(sid, content1)
    
    # Execute the time-consuming script
    content2 = """<function=execute_bash>
<parameter=command>python test.py</parameter>
</function>"""
    await process_action(sid, content2)

def run_example():
    asyncio.run(main())

if __name__ == "__main__":
    run_example()