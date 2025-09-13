from mcp.server.fastmcp import FastMCP
from tools.mcp_tools.func_source_code.message_api import MessageAPI

mcp = FastMCP("Message")

message_api = MessageAPI()

@mcp.tool()
def load_scenario(scenario: dict, long_context: bool = False):
    """
    Load a scenario into the message system.

    Args:
        scenario (dict): The scenario to load.
        long_context (bool): [Optional] Whether to enable long context. Defaults to False.
    """
    try:
        message_api._load_scenario(scenario, long_context)
        return "Successfully loaded from scenario."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def save_scenario():
    """
    Save the current scenario to the database.

    Returns:
        scenario (dict): The current scenario state as a dictionary.
        message (str): A message describing the result of the save operation.
    """
    try:
        scenario = {
            "generated_ids": list(message_api.generated_ids),  # Convert set to list for JSON serialization
            "user_count": message_api.user_count,
            "user_map": message_api.user_map,
            "inbox": message_api.inbox,
            "message_count": message_api.message_count,
            "current_user": message_api.current_user,
        }
        return scenario
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def list_users():
    """
    List all users in the workspace.

    Returns:
        user_list (List[str]): List of all users in the workspace.
    """
    try:
        result = message_api.list_users()
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_user_id(user: str):
    """
    Get user ID from user name.

    Args:
        user (str): User name of the user.

    Returns:
        user_id (str): User ID of the user.
    """
    try:
        result = message_api.get_user_id(user)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def message_login(user_id: str):
    """
    Log in a user with the given user ID to message application.

    Args:
        user_id (str): User ID of the user to log in.

    Returns:
        login_status (bool): True if login was successful, False otherwise.
        message (str): A message describing the result of the login attempt.
    """
    try:
        result = message_api.message_login(user_id)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def message_get_login_status():
    """
    Get the login status of the current user.

    Returns:
        login_status (bool): True if the current user is logged in, False otherwise.
    """
    try:
        result = message_api.message_get_login_status()
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def send_message(receiver_id: str, message: str):
    """
    Send a message to a user.

    Args:
        receiver_id (str): User ID of the user to send the message to.
        message (str): Message to be sent.

    Returns:
        sent_status (bool): True if the message was sent successfully, False otherwise.
        message_id (int): ID of the sent message.
        message (str): A message describing the result of the send attempt.
    """
    try:
        result = message_api.send_message(receiver_id, message)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def delete_message(receiver_id: str):
    """
    Delete the latest message sent to a receiver.

    Args:
        receiver_id (str): User ID of the user to send the message to.

    Returns:
        deleted_status (bool): True if the message was deleted successfully, False otherwise.
        message_id (int): ID of the deleted message.
        message (str): A message describing the result of the deletion attempt.
    """
    try:
        result = message_api.delete_message(receiver_id)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def view_messages_sent():
    """
    View all historical messages sent by the current user.

    Returns:
        messages (Dict): Dictionary of messages grouped by receiver.
    """
    try:
        result = message_api.view_messages_sent()
        return result
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def add_contact(user_name: str):
    """
    Add a contact to the workspace.

    Args:
        user_name (str): User name of contact to be added.

    Returns:
        added_status (bool): True if the contact was added successfully, False otherwise.
        user_id (str): User ID of the added contact.
        message (str): A message describing the result of the addition attempt.
    """
    try:
        result = message_api.add_contact(user_name)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def search_messages(keyword: str):
    """
    Search for messages containing a specific keyword.

    Args:
        keyword (str): The keyword to search for in messages.

    Returns:
        results (List[Dict]): List of dictionaries containing matching messages.
    """
    try:
        result = message_api.search_messages(keyword)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_message_stats():
    """
    Get statistics about messages for the current user.

    Returns:
        stats (Dict): Dictionary containing message statistics.
    """
    try:
        result = message_api.get_message_stats()
        return result
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("\nStarting MCP Message Management Server...")
    mcp.run(transport='stdio')