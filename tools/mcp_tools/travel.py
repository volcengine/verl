from typing import Dict, List, Optional, Union
from mcp.server.fastmcp import FastMCP
from tools.mcp_tools.func_source_code.travel_booking import TravelAPI

mcp = FastMCP("Travel")

travel_api = TravelAPI()

@mcp.tool()
def load_scenario(scenario: dict, long_context: bool = False):
    """
    Load a scenario from the scenarios folder.

    Args:
        scenario (dict): The scenario to load.
        long_context (bool): [Optional] Whether to enable long context. Defaults to False.
    """
    try:
        travel_api._load_scenario(scenario, long_context)
        return "Successfully loaded from scenario."
    except Exception as e:
        return f"Error: {str(e)}"
        
@mcp.tool()
def save_scenario():
    """
    保存当前 TravelAPI 的场景状态
    
    Returns:
        scenario (dict): 当前场景状态的字典
    """
    try:
        result = travel_api.save_scenario()
        if "error" in result:
            return f"Error: {result['error']}"
        return result.get("scenario", {})
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def authenticate_travel(client_id: str, client_secret: str, refresh_token: str, grant_type: str, user_first_name: str, user_last_name: str):
    """
    Authenticate the user with the travel API.

    Args:
        client_id (str): The client applications client_id supplied by App Management.
        client_secret (str): The client applications client_secret supplied by App Management.
        refresh_token (str): The refresh token obtained from the initial authentication.
        grant_type (str): The grant type of the authentication request. Options: read_write, read, write.
        user_first_name (str): The first name of the user.
        user_last_name (str): The last name of the user.

    Returns:
        Authentication information including expires_in, access_token, token_type and scope.
    """
    try:
        result = travel_api.authenticate_travel(client_id, client_secret, refresh_token, grant_type, user_first_name, user_last_name)
        return f"Authentication successful!\nExpires in: {result.get('expires_in')} hours\nAccess token: {result.get('access_token')}\nToken type: {result.get('token_type')}\nScope: {result.get('scope')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def travel_get_login_status():
    """
    Get the status of the login.

    Returns:
        status (bool): The status of the login.
    """
    try:
        result = travel_api.travel_get_login_status()
        status = result.get('status', False)
        return f"Login status: {'Logged in' if status else 'Not logged in'}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_budget_fiscal_year(lastModifiedAfter: Optional[str] = None, includeRemoved: Optional[str] = None):
    """
    Get the budget fiscal year.

    Args:
        lastModifiedAfter (str): [Optional] Use this field if you only want Fiscal Years that were changed after the supplied date.
        includeRemoved (str): [Optional] If true, the service will return all Fiscal Years, including those that were previously removed.

    Returns:
        budget_fiscal_year (str): The budget fiscal year.
    """
    try:
        result = travel_api.get_budget_fiscal_year(lastModifiedAfter, includeRemoved)
        return f"Budget fiscal year: {result.get('budget_fiscal_year')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def register_credit_card(access_token: str, card_number: str, expiration_date: str, cardholder_name: str, card_verification_number: int):
    """
    Register a credit card.

    Args:
        access_token (str): The access token obtained from the authenticate method.
        card_number (str): The credit card number.
        expiration_date (str): The expiration date of the credit card in the format MM/YYYY.
        cardholder_name (str): The name of the cardholder.
        card_verification_number (int): The card verification number.

    Returns:
        card_id (str): The ID of the registered credit card.
    """
    try:
        result = travel_api.register_credit_card(access_token, card_number, expiration_date, cardholder_name, card_verification_number)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Credit card registered successfully!\nCard ID: {result.get('card_id')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_flight_cost(travel_from: str, travel_to: str, travel_date: str, travel_class: str):
    """
    Get the list of cost of a flight in USD based on location, date, and class.

    Args:
        travel_from (str): The 3 letter code of the departing airport.
        travel_to (str): The 3 letter code of the arriving airport.
        travel_date (str): The date of the travel in the format 'YYYY-MM-DD'.
        travel_class (str): The class of the travel. Options are: economy, business, first.

    Returns:
        travel_cost_list (List[float]): The list of cost of the travel.
    """
    try:
        result = travel_api.get_flight_cost(travel_from, travel_to, travel_date, travel_class)
        cost_list = result.get('travel_cost_list', [])
        
        if len(cost_list) == 1:
            return f"Flight cost: ${cost_list[0]}"
        else:
            return f"Flight cost list: {[f'${cost}' for cost in cost_list]}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_credit_card_balance(access_token: str, card_id: str):
    """
    Get the balance of a credit card.

    Args:
        access_token (str): The access token obtained from the authenticate.
        card_id (str): The ID of the credit card.

    Returns:
        card_balance (float): The balance of the credit card.
    """
    try:
        result = travel_api.get_credit_card_balance(access_token, card_id)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Credit card balance: ${result.get('card_balance')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def book_flight(access_token: str, card_id: str, travel_date: str, travel_from: str, travel_to: str, travel_class: str):
    """
    Book a flight given the travel information. From and To should be the airport codes in the IATA format.

    Args:
        access_token (str): The access token obtained from the authenticate.
        card_id (str): The ID of the credit card to use for the booking.
        travel_date (str): The date of the travel in the format YYYY-MM-DD.
        travel_from (str): The location the travel is from.
        travel_to (str): The location the travel is to.
        travel_class (str): The class of the travel.

    Returns:
        Booking information including booking_id, transaction_id and booking_status.
    """
    try:
        result = travel_api.book_flight(access_token, card_id, travel_date, travel_from, travel_to, travel_class)
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get('booking_status'):
            return f"Flight booked successfully!\nBooking ID: {result.get('booking_id')}\nTransaction ID: {result.get('transaction_id')}\nStatus: Success"
        else:
            return f"Flight booking failed: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def retrieve_invoice(access_token: str, booking_id: Optional[str] = None, insurance_id: Optional[str] = None):
    """
    Retrieve the invoice for a booking.

    Args:
        access_token (str): The access token obtained from the authenticate.
        booking_id (str): [Optional] The ID of the booking.
        insurance_id (str): [Optional] The ID of the insurance.

    Returns:
        Invoice information for the booking.
    """
    try:
        result = travel_api.retrieve_invoice(access_token, booking_id, insurance_id)
        if "error" in result:
            return f"Error: {result['error']}"
        
        invoice = result.get('invoice', {})
        return f"Invoice details:\nBooking ID: {invoice.get('booking_id')}\nTravel date: {invoice.get('travel_date')}\nFrom: {invoice.get('travel_from')}\nTo: {invoice.get('travel_to')}\nClass: {invoice.get('travel_class')}\nCost: ${invoice.get('travel_cost')}\nTransaction ID: {invoice.get('transaction_id')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def list_all_airports():
    """
    List all available airports.

    Returns:
        airports (List[str]): A list of all available airports.
    """
    try:
        result = travel_api.list_all_airports()
        return f"Available airports: {', '.join(result)}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def cancel_booking(access_token: str, booking_id: str):
    """
    Cancel a booking.

    Args:
        access_token (str): The access token obtained from the authenticate.
        booking_id (str): The ID of the booking.

    Returns:
        cancel_status (bool): The status of the cancellation, True if successful, False if failed.
    """
    try:
        result = travel_api.cancel_booking(access_token, booking_id)
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get('cancel_status'):
            return "Booking cancelled successfully!"
        else:
            return f"Booking cancellation failed: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def compute_exchange_rate(base_currency: str, target_currency: str, value: float):
    """
    Compute the exchange rate between two currencies.

    Args:
        base_currency (str): The base currency. [Enum]: USD, RMB, EUR, JPY, GBP, CAD, AUD, INR, RUB, BRL, MXN
        target_currency (str): The target currency. [Enum]: USD, RMB, EUR, JPY, GBP, CAD, AUD, INR, RUB, BRL, MXN
        value (float): The value to convert.

    Returns:
        exchanged_value (float): The value after the exchange.
    """
    try:
        result = travel_api.compute_exchange_rate(base_currency, target_currency, value)
        return f"Exchange rate conversion: {value} {base_currency} = {result.get('exchanged_value')} {target_currency}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def verify_traveler_information(first_name: str, last_name: str, date_of_birth: str, passport_number: str):
    """
    Verify the traveler information.

    Args:
        first_name (str): The first name of the traveler.
        last_name (str): The last name of the traveler.
        date_of_birth (str): The date of birth of the traveler in the format YYYY-MM-DD.
        passport_number (str): The passport number of the traveler.

    Returns:
        Verification status and failure reason if any.
    """
    try:
        result = travel_api.verify_traveler_information(first_name, last_name, date_of_birth, passport_number)
        if result.get('verification_status'):
            return "Traveler information verified successfully!"
        else:
            return f"Traveler information verification failed: {result.get('verification_failure')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def set_budget_limit(access_token: str, budget_limit: float):
    """
    Set the budget limit for the user.

    Args:
        access_token (str): The access token obtained from the authentication process or initial configuration.
        budget_limit (float): The budget limit to set in USD.

    Returns:
        budget_limit (float): The budget limit set in USD.
    """
    try:
        result = travel_api.set_budget_limit(access_token, budget_limit)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Budget limit set successfully: ${result.get('budget_limit')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_nearest_airport_by_city(location: str):
    """
    Get the nearest airport to the given location.

    Args:
        location (str): The name of the location.

    Returns:
        nearest_airport (str): The nearest airport to the given location.
    """
    try:
        result = travel_api.get_nearest_airport_by_city(location)
        airport = result.get('nearest_airport')
        if airport == "Unknown":
            return f"Nearest airport not found for location '{location}'"
        return f"Nearest airport for '{location}': {airport}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def purchase_insurance(access_token: str, insurance_type: str, booking_id: str, insurance_cost: float, card_id: str):
    """
    Purchase insurance.

    Args:
        access_token (str): The access token obtained from the authenticate.
        insurance_type (str): The type of insurance to purchase.
        booking_id (str): The ID of the booking.
        insurance_cost (float): The cost of the insurance.
        card_id (str): The ID of the credit card to use for the purchase.

    Returns:
        Insurance purchase result including insurance_id and status.
    """
    try:
        result = travel_api.purchase_insurance(access_token, insurance_type, booking_id, insurance_cost, card_id)
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get('insurance_status'):
            return f"Insurance purchased successfully!\nInsurance ID: {result.get('insurance_id')}"
        else:
            return f"Insurance purchase failed: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def contact_customer_support(booking_id: str, message: str):
    """
    Contact travel booking customer support, get immediate support on an issue with an online call.

    Args:
        booking_id (str): The ID of the booking.
        message (str): The message to send to customer support.

    Returns:
        customer_support_message (str): The message from customer support.
    """
    try:
        result = travel_api.contact_customer_support(booking_id, message)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Customer support response: {result.get('customer_support_message')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_all_credit_cards():
    """
    Get all registered credit cards.

    Returns:
        Information about all registered credit cards.
    """
    try:
        result = travel_api.get_all_credit_cards()
        card_list = result.get('credit_card_list', {})
        
        if not card_list:
            return "No registered credit cards."
        
        cards_info = []
        for card_id, card_info in card_list.items():
            cards_info.append(f"Card ID: {card_id}, Cardholder: {card_info.get('cardholder_name')}, Card number: {card_info.get('card_number')}, Balance: ${card_info.get('balance')}")
        
        return f"Registered credit cards ({len(card_list)} cards):\n" + "\n".join(cards_info)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("\nStarting MCP Travel Booking Server...")
    mcp.run(transport='stdio')