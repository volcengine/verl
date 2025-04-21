import ipaddress

def is_ipv4(ip_str: str) -> bool:
    """
    Check if the given string is an IPv4 address
    
    Args:
        ip_str: The IP address string to check
        
    Returns:
        bool: Returns True if it's an IPv4 address, False otherwise
    """
    try:
        ipaddress.IPv4Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False

def is_ipv6(ip_str: str) -> bool:
    """
    Check if the given string is an IPv6 address
    
    Args:
        ip_str: The IP address string to check
        
    Returns:
        bool: Returns True if it's an IPv6 address, False otherwise
    """
    try:
        ipaddress.IPv6Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False
