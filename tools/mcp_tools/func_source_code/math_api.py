import math
from decimal import Decimal, InvalidOperation, getcontext
from typing import Dict, List, Optional, Union

import mpmath


class MathAPI:
    def __init__(self):
        self._api_description = "This tool belongs to the Math API, which provides various mathematical operations."
        
    def logarithm(
        self, value: float, base: float, precision: int
    ) -> Dict[str, float]:
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
            # Set precision for mpmath
            mpmath.mp.dps = precision

            # Use mpmath for high-precision logarithmic calculations
            result = mpmath.log(value) / mpmath.log(base)

            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def mean(self, numbers: List[float]) -> Dict[str, float]:
        """
        Calculate the mean of a list of numbers.

        Args:
            numbers (List[float]): List of numbers to calculate the mean of.

        Returns:
            result (float): Mean of the numbers.
        """
        if not numbers:
            return {"error": "Cannot calculate mean of an empty list"}
        try:
            return {"result": sum(numbers) / len(numbers)}
        except TypeError:
            return {"error": "All elements in the list must be numbers"}

    def standard_deviation(self, numbers: List[float]) -> Dict[str, float]:
        """
        Calculate the standard deviation of a list of numbers.

        Args:
            numbers (List[float]): List of numbers to calculate the standard deviation of.

        Returns:
            result (float): Standard deviation of the numbers.
        """
        if not numbers:
            return {"error": "Cannot calculate standard deviation of an empty list"}
        try:
            mean = sum(numbers) / len(numbers)
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            return {"result": math.sqrt(variance)}
        except TypeError:
            return {"error": "All elements in the list must be numbers"}

    def si_unit_conversion(
        self, value: float, unit_in: str, unit_out: str
    ) -> Dict[str, float]:
        """
        Convert a value from one SI unit to another.

        Args:
            value (float): Value to be converted.
            unit_in (str): Unit of the input value.
            unit_out (str): Unit to convert the value to.

        Returns:
            result (float): Converted value in the new unit.
        """
        to_meters = {"km": 1000, "m": 1, "cm": 0.01, "mm": 0.001, "um": 1e-6, "nm": 1e-9}
        from_meters = {unit: 1 / factor for unit, factor in to_meters.items()}

        if not isinstance(value, (int, float)):
            return {"error": "Value must be a number"}

        if unit_in not in to_meters or unit_out not in from_meters:
            return {
                "error": f"Conversion from '{unit_in}' to '{unit_out}' is not supported"
            }

        try:
            value_in_meters = value * to_meters[unit_in]
            result = value_in_meters * from_meters[unit_out]
            return {"result": result}
        except OverflowError:
            return {"error": "Conversion resulted in a value too large to represent"}

    def imperial_si_conversion(
        self, value: float, unit_in: str, unit_out: str
    ) -> Dict[str, float]:
        """
        Convert a value between imperial and SI units.

        Args:
            value (float): Value to be converted.
            unit_in (str): Unit of the input value.
            unit_out (str): Unit to convert the value to.

        Returns:
            result (float): Converted value in the new unit.
        """
        conversion = {
            "cm_to_in": 0.393701,
            "in_to_cm": 2.54,
            "m_to_ft": 3.28084,
            "ft_to_m": 0.3048,
            "m_to_yd": 1.09361,
            "yd_to_m": 0.9144,
            "km_to_miles": 0.621371,
            "miles_to_km": 1.60934,
            "kg_to_lb": 2.20462,
            "lb_to_kg": 0.453592,
            "celsius_to_fahrenheit": 1.8,
            "fahrenheit_to_celsius": 5 / 9,
        }

        if not isinstance(value, (int, float)):
            return {"error": "Value must be a number"}

        if unit_in == unit_out:
            return {"result": value}

        conversion_key = f"{unit_in}_to_{unit_out}"
        if conversion_key not in conversion:
            return {
                "error": f"Conversion from '{unit_in}' to '{unit_out}' is not supported"
            }

        try:
            if unit_in == "celsius" and unit_out == "fahrenheit":
                result = (value * conversion[conversion_key]) + 32
            elif unit_in == "fahrenheit" and unit_out == "celsius":
                result = (value - 32) * conversion[conversion_key]
            else:
                result = value * conversion[conversion_key]

            return {"result": result}
        except OverflowError:
            return {"error": "Conversion resulted in a value too large to represent"}

    def add(self, a: float, b: float) -> Dict[str, float]:
        """
        Add two numbers.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            result (float): Sum of the two numbers.
        """
        try:
            return {"result": a + b}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    def subtract(self, a: float, b: float) -> Dict[str, float]:
        """
        Subtract one number from another.

        Args:
            a (float): Number to subtract from.
            b (float): Number to subtract.

        Returns:
            result (float): Difference between the two numbers.
        """
        try:
            return {"result": a - b}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    def multiply(self, a: float, b: float) -> Dict[str, float]:
        """
        Multiply two numbers.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            result (float): Product of the two numbers.
        """
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return {"error": "Both inputs must be numbers"}

        try:
            return {"result": a * b}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    def divide(self, a: float, b: float) -> Dict[str, float]:
        """
        Divide one number by another.

        Args:
            a (float): Numerator.
            b (float): Denominator.

        Returns:
            result (float): Quotient of the division.
        """
        try:
            if b == 0:
                return {"error": "Cannot divide by zero"}
            return {"result": a / b}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    def power(self, base: float, exponent: float) -> Dict[str, float]:
        """
        Raise a number to a power.

        Args:
            base (float): The base number.
            exponent (float): The exponent.

        Returns:
            result (float): The base raised to the power of the exponent.
        """
        try:
            return {"result": base**exponent}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    def square_root(self, number: float, precision: int) -> Dict[str, float]:
        """
        Calculate the square root of a number with adjustable precision using the decimal module.

        Args:
            number (float): The number to calculate the square root of.
            precision (int): Desired precision for the result.

        Returns:
            result (float): The square root of the number, or an error message.
        """
        try:
            if number < 0:
                return {"error": "Cannot calculate square root of a negative number"}

            # Set the precision for the decimal context
            getcontext().prec = precision

            # Use Decimal for high-precision square root calculation
            decimal_number = Decimal(number)

            result = decimal_number.sqrt()
            return {"result": result}
        except (TypeError, InvalidOperation):
            return {
                "error": "Input must be a number or computation resulted in an invalid operation"
            }

    def absolute_value(self, number: float) -> Dict[str, float]:
        """
        Calculate the absolute value of a number.

        Args:
            number (float): The number to calculate the absolute value of.

        Returns:
            result (float): The absolute value of the number.
        """
        try:
            return {"result": abs(number)}
        except TypeError:
            return {"error": "Input must be a number"}

    def round_number(
        self, number: float, decimal_places: int = 0
    ) -> Dict[str, float]:
        """
        Round a number to a specified number of decimal places.

        Args:
            number (float): The number to round.
            decimal_places (int): [Optional] The number of decimal places to round to. Defaults to 0.

        Returns:
            result (float): The rounded number.
        """
        try:
            return {"result": round(number, decimal_places)}
        except TypeError:
            return {
                "error": "First input must be a number, second input must be an integer"
            }

    def percentage(self, part: float, whole: float) -> Dict[str, float]:
        """
        Calculate the percentage of a part relative to a whole.

        Args:
            part (float): The part value.
            whole (float): The whole value.

        Returns:
            result (float): The percentage of the part relative to the whole.
        """
        try:
            if whole == 0:
                return {"error": "Whole value cannot be zero"}
            return {"result": (part / whole) * 100}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    def min_value(self, numbers: List[float]) -> Dict[str, float]:
        """
        Find the minimum value in a list of numbers.

        Args:
            numbers (List[float]): List of numbers to find the minimum from.

        Returns:
            result (float): The minimum value in the list.
        """
        if not numbers:
            return {"error": "Cannot find minimum of an empty list"}
        try:
            return {"result": min(numbers)}
        except TypeError:
            return {"error": "All elements in the list must be numbers"}

    def max_value(self, numbers: List[float]) -> Dict[str, float]:
        """
        Find the maximum value in a list of numbers.

        Args:
            numbers (List[float]): List of numbers to find the maximum from.

        Returns:
            result (float): The maximum value in the list.
        """
        if not numbers:
            return {"error": "Cannot find maximum of an empty list"}
        try:
            return {"result": max(numbers)}
        except TypeError:
            return {"error": "All elements in the list must be numbers"}

    def sum_values(self, numbers: List[float]) -> Dict[str, float]:
        """
        Calculate the sum of a list of numbers.

        Args:
            numbers (List[float]): List of numbers to sum.

        Returns:
            result (float): The sum of all numbers in the list.
        """
        if not numbers:
            return {"error": "Cannot calculate sum of an empty list"}
        try:
            return {"result": sum(numbers)}
        except TypeError:
            return {"error": "All elements in the list must be numbers"}
