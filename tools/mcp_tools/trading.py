from typing import Dict, List, Union
from mcp.server.fastmcp import FastMCP
from tools.mcp_tools.func_source_code.trading_bot import TradingBot

mcp = FastMCP("Trading")

trading_bot = TradingBot()

@mcp.tool()
def load_scenario(scenario: dict, long_context: bool = False):
    """
    Load a scenario into the TradingBot.

    Args:
        scenario (dict): A scenario dictionary containing data to load.
        long_context (bool): [Optional] Whether to enable long context. Defaults to False.
    """
    try:
        trading_bot._load_scenario(scenario, long_context)
        return "Successfully loaded from scenario."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_current_time():
    """
    Get the current time.

    Returns:
        current_time (str): Current time in HH:MM AM/PM format.
    """
    try:
        result = trading_bot.get_current_time()
        return f"Current time: {result.get('current_time')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def update_market_status(current_time_str: str):
    """
    Update the market status based on the current time.

    Args:
        current_time_str (str): Current time in HH:MM AM/PM format.

    Returns:
        status (str): Status of the market. [Enum]: ["Open", "Closed"]
    """
    try:
        result = trading_bot.update_market_status(current_time_str)
        status = result.get('status')
        return f"Market status: {'Open' if status == 'Open' else 'Closed'}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_symbol_by_name(name: str):
    """
    Get the symbol of a stock by company name.

    Args:
        name (str): Name of the company.

    Returns:
        symbol (str): Symbol of the stock or "Stock not found" if not available.
    """
    try:
        result = trading_bot.get_symbol_by_name(name)
        symbol = result.get('symbol')
        if symbol == "Stock not found":
            return f"Stock symbol not found for company '{name}'"
        return f"Stock symbol for '{name}': {symbol}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_stock_info(symbol: str):
    """
    Get the details of a stock.

    Args:
        symbol (str): Symbol that uniquely identifies the stock.

    Returns:
        Stock details including price, percent change, volume and moving averages.
    """
    try:
        result = trading_bot.get_stock_info(symbol)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Stock {symbol} information:\nPrice: ${result.get('price')}\nPercent change: {result.get('percent_change')}%\nVolume: {result.get('volume')}M\n5-day MA: ${result.get('MA(5)')}\n20-day MA: ${result.get('MA(20)')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_order_details(order_id: int):
    """
    Get the details of an order.

    Args:
        order_id (int): ID of the order.

    Returns:
        Order details including ID, type, symbol, price, amount and status.
    """
    try:
        result = trading_bot.get_order_details(order_id)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Order details:\nID: {result.get('id')}\nType: {result.get('order_type')}\nSymbol: {result.get('symbol')}\nPrice: ${result.get('price')}\nAmount: {result.get('amount')}\nStatus: {result.get('status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def cancel_order(order_id: int):
    """
    Cancel an order.

    Args:
        order_id (int): ID of the order to cancel.

    Returns:
        Order ID and status after cancellation attempt.
    """
    try:
        result = trading_bot.cancel_order(order_id)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Order {result.get('order_id')} cancelled, Status: {result.get('status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def place_order(order_type: str, symbol: str, price: float, amount: int):
    """
    Place an order.

    Args:
        order_type (str): Type of the order (Buy/Sell).
        symbol (str): Symbol of the stock to trade.
        price (float): Price at which to place the order.
        amount (int): Number of shares to trade.

    Returns:
        Details of the newly placed order.
    """
    try:
        result = trading_bot.place_order(order_type, symbol, price, amount)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Order placed successfully!\nOrder ID: {result.get('order_id')}\nType: {result.get('order_type')}\nStatus: {result.get('status')}\nPrice: ${result.get('price')}\nAmount: {result.get('amount')}"
    except Exception as e:
        return f"Error: {str(e)}"
        
@mcp.tool()
def save_scenario():
    """
    Save current scenario from the TradingBot instance.
    
    Returns:
        scenario (dict): The current scenario state as a dictionary.
    """
    try:
        result = trading_bot.save_scenario()
        if "error" in result:
            return f"Error: {result['error']}"
        return result.get("scenario", {})
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def make_transaction(account_id: int, xact_type: str, amount: float):
    """
    Make a deposit or withdrawal based on specified amount.

    Args:
        account_id (int): ID of the account.
        xact_type (str): Transaction type (deposit or withdrawal).
        amount (float): Amount to deposit or withdraw.

    Returns:
        Transaction status and updated account balance.
    """
    try:
        result = trading_bot.make_transaction(account_id, xact_type, amount)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Transaction status: {result.get('status')}\nNew balance: ${result.get('new_balance')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_account_info():
    """
    Get account information.

    Returns:
        Account information including ID, balance and binding card.
    """
    try:
        result = trading_bot.get_account_info()
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Account information:\nAccount ID: {result.get('account_id')}\nBalance: ${result.get('balance')}\nBinding card: {result.get('binding_card')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def trading_login(username: str, password: str):
    """
    Handle user login.

    Args:
        username (str): Username for authentication.
        password (str): Password for authentication.

    Returns:
        status (str): Login status message.
    """
    try:
        result = trading_bot.trading_login(username, password)
        return f"Login status: {result.get('status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def trading_get_login_status():
    """
    Get the login status.

    Returns:
        status (bool): Login status.
    """
    try:
        result = trading_bot.trading_get_login_status()
        status = result.get('status', False)
        return f"Login status: {'Logged in' if status else 'Not logged in'}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def trading_logout():
    """
    Handle user logout for trading system.

    Returns:
        status (str): Logout status message.
    """
    try:
        result = trading_bot.trading_logout()
        return f"Logout status: {result.get('status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def fund_account(amount: float):
    """
    Fund the account with the specified amount.

    Args:
        amount (float): Amount to fund the account with.

    Returns:
        Funding status and updated account balance.
    """
    try:
        result = trading_bot.fund_account(amount)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Funding status: {result.get('status')}\nNew balance: ${result.get('new_balance')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def remove_stock_from_watchlist(symbol: str):
    """
    Remove a stock from the watchlist.

    Args:
        symbol (str): Symbol of the stock to remove.

    Returns:
        status (str): Status of the removal operation.
    """
    try:
        result = trading_bot.remove_stock_from_watchlist(symbol)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Operation status: {result.get('status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_watchlist():
    """
    Get the watchlist.

    Returns:
        watchlist (List[str]): List of stock symbols in the watchlist.
    """
    try:
        result = trading_bot.get_watchlist()
        if isinstance(result, list) and len(result) > 0 and "Error" in result[0]:
            return result[0]
        
        if isinstance(result, dict):
            watchlist = result.get('watchlist', [])
        else:
            watchlist = result
        
        if not watchlist:
            return "Watchlist is empty."
        
        return f"Watchlist: {', '.join(watchlist)}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_order_history():
    """
    Get the stock order ID history.

    Returns:
        order_history (List[int]): List of orders ID in the order history.
    """
    try:
        result = trading_bot.get_order_history()
        if isinstance(result, list) and len(result) > 0 and "error" in result[0]:
            return result[0]["error"]
        
        history = result.get('history', [])
        if not history:
            return "Order history is empty."
        
        return f"Order history: {', '.join(map(str, history))}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_transaction_history(start_date: str = None, end_date: str = None):
    """
    Get the transaction history within a specified date range.

    Args:
        start_date (str): [Optional] Start date for the history (format: 'YYYY-MM-DD').
        end_date (str): [Optional] End date for the history (format: 'YYYY-MM-DD').

    Returns:
        List of transactions within the specified date range.
    """
    try:
        result = trading_bot.get_transaction_history(start_date, end_date)
        if isinstance(result, list) and len(result) > 0 and "error" in result[0]:
            return result[0]["error"]
        
        history = result.get('transaction_history', [])
        if not history:
            return "Transaction history is empty."
        
        transactions_info = []
        for transaction in history:
            transactions_info.append(f"Type: {transaction.get('type')}, Amount: ${transaction.get('amount')}, Time: {transaction.get('timestamp')}")
        
        date_range = ""
        if start_date or end_date:
            date_range = f" ({start_date or 'start'} to {end_date or 'now'})"
        
        return f"Transaction history{date_range} ({len(history)} transactions):\n" + "\n".join(transactions_info)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def update_stock_price(symbol: str, new_price: float):
    """
    Update the price of a stock.

    Args:
        symbol (str): Symbol of the stock to update.
        new_price (float): New price of the stock.

    Returns:
        Updated stock information including symbol, old price and new price.
    """
    try:
        result = trading_bot.update_stock_price(symbol, new_price)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Stock price updated successfully:\nSymbol: {result.get('symbol')}\nOld price: ${result.get('old_price')}\nNew price: ${result.get('new_price')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_available_stocks(sector: str):
    """
    Get a list of stock symbols in the given sector.

    Args:
        sector (str): The sector to retrieve stocks from (e.g., 'Technology').

    Returns:
        stock_list (List[str]): List of stock symbols in the specified sector.
    """
    try:
        result = trading_bot.get_available_stocks(sector)
        stock_list = result.get('stock_list', [])
        
        if not stock_list:
            return f"No stocks available in sector '{sector}'."
        
        return f"Stocks in '{sector}' sector: {', '.join(stock_list)}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def filter_stocks_by_price(stocks: List[str], min_price: float, max_price: float):
    """
    Filter stocks based on a price range.

    Args:
        stocks (List[str]): List of stock symbols to filter.
        min_price (float): Minimum stock price.
        max_price (float): Maximum stock price.

    Returns:
        filtered_stocks (List[str]): Filtered list of stock symbols within the price range.
    """
    try:
        result = trading_bot.filter_stocks_by_price(stocks, min_price, max_price)
        filtered_stocks = result.get('filtered_stocks', [])
        
        if not filtered_stocks:
            return f"No stocks found in price range ${min_price}-${max_price}."
        
        return f"Stocks in price range ${min_price}-${max_price}: {', '.join(filtered_stocks)}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def add_to_watchlist(stock: str):
    """
    Add a stock to the watchlist.

    Args:
        stock (str): the stock symbol to add to the watchlist.

    Returns:
        symbol (str): the symbol that were successfully added to the watchlist.
    """
    try:
        result = trading_bot.add_to_watchlist(stock)
        watchlist = result.get('symbol', [])
        return f"Added to watchlist. Current watchlist: {', '.join(watchlist)}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def notify_price_change(stocks: List[str], threshold: float):
    """
    Notify if there is a significant price change in the stocks.

    Args:
        stocks (List[str]): List of stock symbols to check.
        threshold (float): Percentage change threshold to trigger a notification.

    Returns:
        notification (str): Notification message about the price changes.
    """
    try:
        result = trading_bot.notify_price_change(stocks, threshold)
        return f"Price change notification: {result.get('notification')}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("\nStarting MCP Trading Server...")
    mcp.run(transport='stdio')