from mcp.server.fastmcp import FastMCP
from tools.mcp_tools.func_source_code.ticket_api import TicketAPI

mcp = FastMCP("Ticket")

ticket_api = TicketAPI()

@mcp.tool()
def load_scenario(scenario: dict, long_context: bool = False):
    """
    Load a scenario into the ticket queue.

    Args:
        scenario (dict): A dictionary containing ticket data.
        long_context (bool): [Optional] Whether to enable long context. Defaults to False.
    """
    try:
        ticket_api._load_scenario(scenario, long_context)
        return "Successfully loaded from scenario."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def save_scenario():
    """
    Save current scenario from the ticket queue.
    """
    try:
        result = ticket_api.save_scenario()
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def create_ticket(title: str, description: str = "", priority: int = 1):
    """
    Create a ticket in the system and queue it.

    Args:
        title (str): Title of the ticket.
        description (str): [Optional] Description of the ticket. Defaults to an empty string.
        priority (int): [Optional] Priority of the ticket, from 1 to 5. Defaults to 1. 5 is the highest priority.

    Returns:
        Ticket information including ID, title, description, status and priority.
    """
    try:
        result = ticket_api.create_ticket(title, description, priority)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Ticket created successfully!\nID: {result.get('id')}\nTitle: {result.get('title')}\nDescription: {result.get('description')}\nStatus: {result.get('status')}\nPriority: {result.get('priority')}\nCreated by: {result.get('created_by')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_ticket(ticket_id: int):
    """
    Get a specific ticket by its ID.

    Args:
        ticket_id (int): ID of the ticket to retrieve.

    Returns:
        Ticket information including ID, title, description, status, priority and creator.
    """
    try:
        result = ticket_api.get_ticket(ticket_id)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"Ticket details:\nID: {result.get('id')}\nTitle: {result.get('title')}\nDescription: {result.get('description')}\nStatus: {result.get('status')}\nPriority: {result.get('priority')}\nCreated by: {result.get('created_by')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def close_ticket(ticket_id: int):
    """
    Close a ticket.

    Args:
        ticket_id (int): ID of the ticket to be closed.

    Returns:
        status (str): Status of the close operation.
    """
    try:
        result = ticket_api.close_ticket(ticket_id)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Operation status: {result.get('status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def resolve_ticket(ticket_id: int, resolution: str):
    """
    Resolve a ticket with a resolution.

    Args:
        ticket_id (int): ID of the ticket to be resolved.
        resolution (str): Resolution details for the ticket.

    Returns:
        status (str): Status of the resolve operation.
    """
    try:
        result = ticket_api.resolve_ticket(ticket_id, resolution)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Operation status: {result.get('status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def edit_ticket(ticket_id: int, title: str = None, description: str = None, status: str = None, priority: str = None):
    """
    Modify the details of an existing ticket.

    Args:
        ticket_id (int): ID of the ticket to be changed.
        title (str): [Optional] New title for the ticket.
        description (str): [Optional] New description for the ticket.
        status (str): [Optional] New status for the ticket.
        priority (int): [Optional] New priority for the ticket.

    Returns:
        status (str): Status of the update operation.
    """
    try:
        updates = {}
        if title is not None:
            updates["title"] = title
        if description is not None:
            updates["description"] = description
        if status is not None:
            updates["status"] = status
        if priority is not None:
            updates["priority"] = priority
        
        if not updates:
            return "No update fields provided."
        
        result = ticket_api.edit_ticket(ticket_id, updates)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Operation status: {result.get('status')}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def ticket_login(username: str, password: str):
    """
    Authenticate a user for ticket system.

    Args:
        username (str): Username of the user.
        password (str): Password of the user.

    Returns:
        success (bool): True if login was successful, False otherwise.
    """
    try:
        result = ticket_api.ticket_login(username, password)
        success = result.get('success', False)
        return f"Login status: {'Success' if success else 'Failed'}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def ticket_get_login_status():
    """
    Get the username of the currently authenticated user.

    Returns:
        username (bool): True if a user is logged in, False otherwise.
    """
    try:
        result = ticket_api.ticket_get_login_status()
        status = result.get('username', False)
        return f"Login status: {'Logged in' if status else 'Not logged in'}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def logout():
    """
    Log out the current user.

    Returns:
        success (bool): True if logout was successful, False otherwise.
    """
    try:
        result = ticket_api.logout()
        success = result.get('success', False)
        return f"Logout status: {'Success' if success else 'Failed'}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_user_tickets(status: str = None):
    """
    Get all tickets created by the current user, optionally filtered by status.

    Args:
        status (str): [Optional] Status to filter tickets by. If None, return all tickets.

    Returns:
        List of user tickets, each containing ID, title, description, status, priority and creator.
    """
    try:
        result = ticket_api.get_user_tickets(status)
        if isinstance(result, list) and len(result) == 1 and "error" in result[0]:
            return f"Error: {result[0]['error']}"
        
        if not result:
            status_filter = f"with status '{status}' " if status else ""
            return f"No tickets found {status_filter}."
        
        tickets_info = []
        for ticket in result:
            tickets_info.append(f"ID: {ticket.get('id')}, Title: {ticket.get('title')}, Status: {ticket.get('status')}, Priority: {ticket.get('priority')}")
        
        status_filter = f" (Status: {status})" if status else ""
        return f"User tickets{status_filter} ({len(result)} tickets):\n" + "\n".join(tickets_info)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("\nStarting MCP Ticket Management Server...")
    mcp.run(transport='stdio')