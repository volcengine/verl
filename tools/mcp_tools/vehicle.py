from typing import Dict, List, Union
from mcp.server.fastmcp import FastMCP
from tools.mcp_tools.func_source_code.vehicle_control import VehicleControlAPI

mcp = FastMCP("Vehicle")

vehicle_api = VehicleControlAPI()

@mcp.tool()
def load_scenario(scenario: dict, long_context: bool = False):
    """
    Loads the scenario for the vehicle control.

    Args:
        scenario (dict): The scenario to load.
        long_context (bool): [Optional] Whether to enable long context. Defaults to False.
    """
    try:
        vehicle_api._load_scenario(scenario, long_context)
        return "Successfully loaded from scenario."
    except Exception as e:
        return f"Error: {str(e)}"
        
@mcp.tool()
def save_scenario():
    """
    Exports the current scenario state of the vehicle.
    Returns:
        scenario (Dict): The current scenario state of the vehicle.
    """
    try:
        scenario = vehicle_api.save_scenario()
        return scenario
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def startEngine(ignitionMode: str):
    """
    Starts the engine of the vehicle.

    Args:
        ignitionMode (str): The ignition mode of the vehicle. [Enum]: ["START", "STOP"]

    Returns:
        Engine state, fuel level and battery voltage.
    """
    try:
        result = vehicle_api.startEngine(ignitionMode)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Engine started successfully!\nEngine state: {result.get('engineState')}\nFuel level: {result.get('fuelLevel')} gallons\nBattery voltage: {result.get('batteryVoltage')} volts"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def fillFuelTank(fuelAmount: float):
    """
    Fills the fuel tank of the vehicle. The fuel tank can hold up to 50 gallons.

    Args:
        fuelAmount (float): The amount of fuel to fill in gallons; this is the additional fuel to add to the tank.

    Returns:
        fuelLevel (float): The fuel level of the vehicle in gallons.
    """
    try:
        result = vehicle_api.fillFuelTank(fuelAmount)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Fuel filled successfully! Current fuel level: {result.get('fuelLevel')} gallons"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def lockDoors(unlock: bool, door: List[str]):
    """
    Locks the doors of the vehicle.

    Args:
        unlock (bool): True if the doors are to be unlocked, False otherwise.
        door (List[str]): The list of doors to lock or unlock. [Enum]: ["driver", "passenger", "rear_left", "rear_right"]

    Returns:
        Door lock status and remaining unlocked doors count.
    """
    try:
        result = vehicle_api.lockDoors(unlock, door)
        action = "unlocked" if unlock else "locked"
        return f"Doors {action} successfully!\nLock status: {result.get('lockStatus')}\nRemaining unlocked doors: {result.get('remainingUnlockedDoors')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def adjustClimateControl(temperature: float, unit: str = "celsius", fanSpeed: int = 50, mode: str = "auto"):
    """
    Adjusts the climate control of the vehicle.

    Args:
        temperature (float): The temperature to set in degree. Default to be celsius.
        unit (str): [Optional] The unit of temperature. [Enum]: ["celsius", "fahrenheit"]
        fanSpeed (int): [Optional] The fan speed to set from 0 to 100. Default is 50.
        mode (str): [Optional] The climate mode to set. [Enum]: ["auto", "cool", "heat", "defrost"]

    Returns:
        Current temperature setting, climate mode and humidity level.
    """
    try:
        result = vehicle_api.adjustClimateControl(temperature, unit, fanSpeed, mode)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Climate control adjusted successfully!\nCurrent temperature: {result.get('currentACTemperature')}°{unit}\nClimate mode: {result.get('climateMode')}\nHumidity level: {result.get('humidityLevel')}%"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_outside_temperature_from_google():
    """
    Gets the outside temperature from Google.

    Returns:
        outsideTemperature (float): The outside temperature in degree Celsius.
    """
    try:
        result = vehicle_api.get_outside_temperature_from_google()
        return f"Outside temperature (from Google): {result.get('outsideTemperature'):.1f}°C"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_outside_temperature_from_weather_com():
    """
    Gets the outside temperature from Weather.com.

    Returns:
        Outside temperature or error information.
    """
    try:
        result = vehicle_api.get_outside_temperature_from_weather_com()
        if "error" in result:
            return f"Weather.com service unavailable (Error {result['error']})"
        return f"Outside temperature (from Weather.com): {result.get('outsideTemperature'):.1f}°C"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def setHeadlights(mode: str):
    """
    Sets the headlights of the vehicle.

    Args:
        mode (str): The mode of the headlights. [Enum]: ["on", "off", "auto"]

    Returns:
        headlightStatus (str): The status of the headlights. [Enum]: ["on", "off"]
    """
    try:
        result = vehicle_api.setHeadlights(mode)
        if "error" in result:
            return f"Error: {result['error']}"
        
        status = "on" if result.get('headlightStatus') == "on" else "off"
        return f"Headlights set successfully! Status: {status}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def displayCarStatus(option: str):
    """
    Displays the status of the vehicle based on the provided display option.

    Args:
        option (str): The option to display. [Enum]: ["fuel", "battery", "doors", "climate", "headlights", "parkingBrake", "brakePedal", "engine"]

    Returns:
        Vehicle status based on the option.
    """
    try:
        result = vehicle_api.displayCarStatus(option)
        if "error" in result:
            return f"Error: {result['error']}"
        
        status_info = []
        for key, value in result.items():
            if key == "metadata":
                continue
            if isinstance(value, dict):
                door_status = []
                for door, status in value.items():
                    door_status.append(f"{door}: {status}")
                status_info.append(f"Door status: {', '.join(door_status)}")
            else:
                status_info.append(f"{key}: {value}")
        
        return f"Vehicle status ({option}):\n" + "\n".join(status_info)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def activateParkingBrake(mode: str):
    """
    Activates the parking brake of the vehicle.

    Args:
        mode (str): The mode to set. [Enum]: ["engage", "release"]

    Returns:
        Parking brake status, brake force and slope angle.
    """
    try:
        result = vehicle_api.activateParkingBrake(mode)
        if "error" in result:
            return f"Error: {result['error']}"
        
        action = "engaged" if mode == "engage" else "released"
        return f"Parking brake {action} successfully!\nStatus: {result.get('parkingBrakeStatus')}\nBrake force: {result.get('_parkingBrakeForce')} Newtons\nSlope angle: {result.get('_slopeAngle')} degrees"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def pressBrakePedal(pedalPosition: float):
    """
    Presses the brake pedal based on pedal position. The brake pedal will be kept pressed until released.

    Args:
        pedalPosition (float): Position of the brake pedal, between 0 (not pressed) and 1 (fully pressed).

    Returns:
        Brake pedal status and applied force.
    """
    try:
        result = vehicle_api.pressBrakePedal(pedalPosition)
        if "error" in result:
            return f"Error: {result['error']}"
        
        status = "pressed" if result.get('brakePedalStatus') == "pressed" else "released"
        return f"Brake pedal operation successful!\nStatus: {status}\nBrake force: {result.get('brakePedalForce')} Newtons"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def releaseBrakePedal():
    """
    Releases the brake pedal of the vehicle.

    Returns:
        Brake pedal status and applied force.
    """
    try:
        result = vehicle_api.releaseBrakePedal()
        return f"Brake pedal released successfully!\nStatus: {result.get('brakePedalStatus')}\nBrake force: {result.get('brakePedalForce')} Newtons"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def setCruiseControl(speed: float, activate: bool, distanceToNextVehicle: float):
    """
    Sets the cruise control of the vehicle.

    Args:
        speed (float): The speed to set in km/h. The speed should be between 0 and 120 and a multiple of 5.
        activate (bool): True to activate the cruise control, False to deactivate.
        distanceToNextVehicle (float): The distance to the next vehicle in meters.

    Returns:
        Cruise control status, current speed and distance to next vehicle.
    """
    try:
        result = vehicle_api.setCruiseControl(speed, activate, distanceToNextVehicle)
        if "error" in result:
            return f"Error: {result['error']}"
        
        status = "active" if result.get('cruiseStatus') == "active" else "inactive"
        return f"Cruise control set successfully!\nStatus: {status}\nCurrent speed: {result.get('currentSpeed')} km/h\nDistance to next vehicle: {result.get('distanceToNextVehicle')} meters"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_current_speed():
    """
    Gets the current speed of the vehicle.

    Returns:
        currentSpeed (float): The current speed of the vehicle in km/h.
    """
    try:
        result = vehicle_api.get_current_speed()
        return f"Current speed: {result.get('currentSpeed'):.1f} km/h"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def estimate_drive_feasibility_by_mileage(distance: float):
    """
    Estimates the mileage of the vehicle given the distance needed to drive.

    Args:
        distance (float): The distance to travel in miles.

    Returns:
        canDrive (bool): True if the vehicle can drive the distance, False otherwise.
    """
    try:
        result = vehicle_api.estimate_drive_feasibility_by_mileage(distance)
        can_drive = result.get('canDrive', False)
        return f"Drive feasibility: {'Can drive' if can_drive else 'Insufficient fuel to drive'} {distance} miles"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def liter_to_gallon(liter: float):
    """
    Converts the liter to gallon.

    Args:
        liter (float): The amount of liter to convert.

    Returns:
        gallon (float): The amount of gallon converted.
    """
    try:
        result = vehicle_api.liter_to_gallon(liter)
        return f"Conversion result: {liter} liters = {result.get('gallon'):.3f} gallons"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def gallon_to_liter(gallon: float):
    """
    Converts the gallon to liter.

    Args:
        gallon (float): The amount of gallon to convert.

    Returns:
        liter (float): The amount of liter converted.
    """
    try:
        result = vehicle_api.gallon_to_liter(gallon)
        return f"Conversion result: {gallon} gallons = {result.get('liter'):.3f} liters"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def estimate_distance(cityA: str, cityB: str):
    """
    Estimates the distance between two cities.

    Args:
        cityA (str): The zipcode of the first city.
        cityB (str): The zipcode of the second city.

    Returns:
        Distance between the two cities in km.
    """
    try:
        result = vehicle_api.estimate_distance(cityA, cityB)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Distance estimation: Distance from {cityA} to {cityB} is {result.get('distance')} km"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_zipcode_based_on_city(city: str):
    """
    Gets the zipcode based on the city.

    Args:
        city (str): The name of the city.

    Returns:
        zipcode (str): The zipcode of the city.
    """
    try:
        result = vehicle_api.get_zipcode_based_on_city(city)
        zipcode = result.get('zipcode')
        if zipcode == "00000":
            return f"Zipcode not found for city '{city}'"
        return f"Zipcode for city '{city}': {zipcode}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def set_navigation(destination: str):
    """
    Navigates to the destination.

    Args:
        destination (str): The destination to navigate in the format of street, city, state.

    Returns:
        status (str): The status of the navigation.
    """
    try:
        result = vehicle_api.set_navigation(destination)
        return f"Navigation status: {result.get('status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def check_tire_pressure():
    """
    Checks the tire pressure of the vehicle.

    Returns:
        Tire pressure information including individual tire pressures and health status.
    """
    try:
        result = vehicle_api.check_tire_pressure()
        healthy = "Normal" if result.get('healthy_tire_pressure') else "Abnormal"
        
        return f"Tire pressure check results:\nFront left tire: {result.get('frontLeftTirePressure')} psi\nFront right tire: {result.get('frontRightTirePressure')} psi\nRear left tire: {result.get('rearLeftTirePressure')} psi\nRear right tire: {result.get('rearRightTirePressure')} psi\nTire pressure status: {healthy}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def find_nearest_tire_shop():
    """
    Finds the nearest tire shop.

    Returns:
        shopLocation (str): The location of the nearest tire shop.
    """
    try:
        result = vehicle_api.find_nearest_tire_shop()
        return f"Nearest tire shop: {result.get('shopLocation')}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("\nStarting MCP Vehicle Control Server...")
    mcp.run(transport='stdio')