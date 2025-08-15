import asyncio
import multiprocessing
import random
import socket

import httpx


def find_available_port(base_port: int):
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        if port < 60000:
            port += 42
        else:
            port -= 43


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except socket.error:
            return False
        except OverflowError:
            return False


def get_host_info():
    hostname = socket.gethostname()

    local_ip = socket.gethostbyname(hostname)

    return hostname, local_ip


def run_router(args):
    try:
        from sglang_router.launch_router import launch_router

        router = launch_router(args)
        if router is None:
            return 1
        return 0
    except Exception as e:
        print(e)
        return 1


def terminate_process(process: multiprocessing.Process, timeout: float = 1.0) -> None:
    """Terminate a process gracefully, with forced kill as fallback.

    Args:
        process: The process to terminate
        timeout: Seconds to wait for graceful termination before forcing kill
    """
    if not process.is_alive():
        return

    process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()


async def post(url, payload, use_http2=False, max_retries=60):
    # never timeout
    timeout = httpx.Timeout(None)
    max_retries = 60
    retry_count = 0
    while retry_count < max_retries:
        try:
            async with httpx.AsyncClient(http1=not use_http2, http2=use_http2, timeout=timeout) as client:
                response = await client.post(url, json=payload or {})
                response.raise_for_status()
                try:
                    output = response.json()
                except:
                    output = response.text
        except Exception as e:
            retry_count += 1
            print(f"Error: {e}, retrying... (attempt {retry_count}/{max_retries})")
            if retry_count >= max_retries:
                print(f"Max retries ({max_retries}) reached, failing...")
                raise e
            await asyncio.sleep(1)
            continue
        break

    return output


async def get(url, use_http2=False):
    # never timeout
    timeout = httpx.Timeout(None)
    async with httpx.AsyncClient(http1=not use_http2, http2=use_http2, timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        output = response.json()
    return output
