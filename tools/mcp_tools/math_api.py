from typing import List
from mcp.server.fastmcp import FastMCP
from tools.mcp_tools.func_source_code.math_api import MathAPI

mcp = FastMCP("Math")

math_api = MathAPI()

@mcp.tool()
def logarithm(value: float, base: float, precision: int):
    """
    Compute the logarithm of a number with adjustable precision using mpmath.

    Args:
        value (float): The number to compute the logarithm of.
        base (float): The base of the logarithm.
        precision (int): Desired precision for the result.

    Returns:
        result (float): The logarithm of the number with respect to the given base.
    """
    try:
        result = math_api.logarithm(value, base, precision)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def mean(numbers: List[float]):
    """
    Calculate the mean of a list of numbers.

    Args:
        numbers (List[float]): List of numbers to calculate the mean of.

    Returns:
        result (float): Mean of the numbers.
    """
    try:
        result = math_api.mean(numbers)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def standard_deviation(numbers: List[float]):
    """
    Calculate the standard deviation of a list of numbers.

    Args:
        numbers (List[float]): List of numbers to calculate the standard deviation of.

    Returns:
        result (float): Standard deviation of the numbers.
    """
    try:
        result = math_api.standard_deviation(numbers)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def si_unit_conversion(value: float, unit_in: str, unit_out: str):
    """
    Convert a value from one SI unit to another.

    Args:
        value (float): Value to be converted.
        unit_in (str): Unit of the input value.
        unit_out (str): Unit to convert the value to.

    Returns:
        result (float): Converted value in the new unit.
    """
    try:
        result = math_api.si_unit_conversion(value, unit_in, unit_out)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def imperial_si_conversion(value: float, unit_in: str, unit_out: str):
    """
    Convert a value between imperial and SI units.

    Args:
        value (float): Value to be converted.
        unit_in (str): Unit of the input value.
        unit_out (str): Unit to convert the value to.

    Returns:
        result (float): Converted value in the new unit.
    """
    try:
        result = math_api.imperial_si_conversion(value, unit_in, unit_out)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def add(a: float, b: float):
    """
    Add two numbers.

    Args:
        a (float): First number.
        b (float): Second number.

    Returns:
        result (float): Sum of the two numbers.
    """
    try:
        result = math_api.add(a, b)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def subtract(a: float, b: float):
    """
    Subtract one number from another.

    Args:
        a (float): Number to subtract from.
        b (float): Number to subtract.

    Returns:
        result (float): Difference between the two numbers.
    """
    try:
        result = math_api.subtract(a, b)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def multiply(a: float, b: float):
    """
    Multiply two numbers.

    Args:
        a (float): First number.
        b (float): Second number.

    Returns:
        result (float): Product of the two numbers.
    """
    try:
        result = math_api.multiply(a, b)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def divide(a: float, b: float):
    """
    Divide one number by another.

    Args:
        a (float): Numerator.
        b (float): Denominator.

    Returns:
        result (float): Quotient of the division.
    """
    try:
        result = math_api.divide(a, b)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def power(base: float, exponent: float):
    """
    Raise a number to a power.

    Args:
        base (float): The base number.
        exponent (float): The exponent.

    Returns:
        result (float): The base raised to the power of the exponent.
    """
    try:
        result = math_api.power(base, exponent)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def square_root(number: float, precision: int):
    """
    Calculate the square root of a number with adjustable precision using the decimal module.

    Args:
        number (float): The number to calculate the square root of.
        precision (int): Desired precision for the result.

    Returns:
        result (float): The square root of the number, or an error message.
    """
    try:
        result = math_api.square_root(number, precision)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def absolute_value(number: float):
    """
    Calculate the absolute value of a number.

    Args:
        number (float): The number to calculate the absolute value of.

    Returns:
        result (float): The absolute value of the number.
    """
    try:
        result = math_api.absolute_value(number)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def round_number(number: float, decimal_places: int = 0):
    """
    Round a number to a specified number of decimal places.

    Args:
        number (float): The number to round.
        decimal_places (int): [Optional] The number of decimal places to round to. Defaults to 0.

    Returns:
        result (float): The rounded number.
    """
    try:
        result = math_api.round_number(number, decimal_places)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def percentage(part: float, whole: float):
    """
    Calculate the percentage of a part relative to a whole.

    Args:
        part (float): The part value.
        whole (float): The whole value.

    Returns:
        result (float): The percentage of the part relative to the whole.
    """
    try:
        result = math_api.percentage(part, whole)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def min_value(numbers: List[float]):
    """
    Find the minimum value in a list of numbers.

    Args:
        numbers (List[float]): List of numbers to find the minimum from.

    Returns:
        result (float): The minimum value in the list.
    """
    try:
        result = math_api.min_value(numbers)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def max_value(numbers: List[float]):
    """
    Find the maximum value in a list of numbers.

    Args:
        numbers (List[float]): List of numbers to find the maximum from.

    Returns:
        result (float): The maximum value in the list.
    """
    try:
        result = math_api.max_value(numbers)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def sum_values(numbers: List[float]):
    """
    Calculate the sum of a list of numbers.

    Args:
        numbers (List[float]): List of numbers to sum.

    Returns:
        result (float): The sum of all numbers in the list.
    """
    try:
        result = math_api.sum_values(numbers)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("\nStarting MCP Math Server...")
    mcp.run(transport='stdio')