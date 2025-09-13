from mcp.server.fastmcp import FastMCP
from tools.mcp_tools.func_source_code.gorilla_file_system import *
mcp = FastMCP("FileSystem")

file_system = GorillaFileSystem()

@mcp.tool(
    name="file_system-load_scenario",
)
def load_scenario(scenario: dict, long_context: bool = False):
    """
        Load a scenario into the file system.

        Args:
            scenario (dict): The scenario to load.

        The scenario always starts with a root directory. Each directory can contain files or subdirectories.
        The key is the name of the file or directory, and the value is a dictionary with the following keys
        An example scenario:
        Here John is the root directory and it contains a home directory with a user directory inside it.
        The user directory contains a file named file1.txt and a directory named directory1.
        Root is not a part of the scenario and it's just easy for parsing. During generation, you should have at most 2 layers.
        {
            "root": {
                "john": {
                    "type": "directory",
                    "contents": {
                        "home": {
                            "type": "directory",
                            "contents": {
                                "user": {
                                    "type": "directory",
                                    "contents": {
                                        "file1.txt": {
                                            "type": "file",
                                            "content": "Hello, world!"
                                        },
                                        "directory1": {
                                            "type": "directory",
                                            "contents": {}
                                        }
                                    }
                                }
                            }
                        }
                }
            }
        }
    """
    try:
        file_system._load_scenario(scenario, long_context)
        return "Sucessfully load from scenario."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def save_scenario():
    """Save a scenario. """
    try:
        if not hasattr(file_system, "root") or file_system.root is None:
            return None

        def serialize_directory(directory):
            contents = {}
            for name, item in directory.contents.items():
                # Directory
                if isinstance(item, Directory):
                    contents[name] = {
                        "type": "directory",
                        "contents": serialize_directory(item),
                    }
                # File
                elif isinstance(item, File):
                    contents[name] = {
                        "type": "file",
                        "content": item.content,
                    }
            return contents

        root_dir = file_system.root
        scenario = {
            "root": {
                root_dir.name: {
                    "type": "directory",
                    "contents": serialize_directory(root_dir),
                }
            }
        }
        return scenario
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def pwd() -> str:
    """ Return the current working directory path.
        Args:
            None
        Returns:
            current_working_directory (str): The current working directory path.
    """
    try:
        result = file_system.pwd()
        current_dir = result.get('current_working_directory', '/')
        return f"{current_dir}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def ls(show_hidden: bool = False) -> str:
    """
        List the contents of the current directory.

        Args:
            a (bool): [Optional] Show hidden files and directories. Defaults to False.

        Returns:
            current_directory_content (List[str]): A list of the contents of the specified directory.
    """
    try:
        result = file_system.ls(show_hidden)
        contents = result.get('current_directory_content', [])
        return f"{', '.join(contents)}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def cd(folder: str) -> str:
    """
    Change the current working directory to the specified folder.

    Args:
        folder (str): The folder of the directory to change to. You can only change one folder at a time.

    Returns:
        current_working_directory (str): The new current working directory path.
    """
    try:
        result = file_system.cd(folder)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def mkdir(dir_name: str) -> str:
    """
    Create a new directory in the current directory.

    Args:
        dir_name (str): The name of the new directory at current directory. You can only create directory at current directory.
    """
    try:
        result = file_system.mkdir(dir_name)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def touch(file_name: str) -> str:
    """
    Create a new file of any extension in the current directory.

    Args:
        file_name (str): The name of the new file in the current directory. file_name is local to the current directory and does not allow path.
    """
    try:
        result = file_system.touch(file_name)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def echo(content: str, file_name: str = None) -> str:
    """
    Write content to a file at current directory or display it in the terminal.

    Args:
        content (str): The content to write or display.
        file_name (str): [Optional] The name of the file at current directory to write the content to. Defaults to None.

    Returns:
        terminal_output (str): The content if no file name is provided, or None if written to file.
    """
    try:
        result = file_system.echo(content, file_name)
        if file_name:
            return f"{result}"
        else:
            return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def cat(file_name: str) -> str:
    """
    Display the contents of a file of any extension from currrent directory.

    Args:
        file_name (str): The name of the file from current directory to display. No path is allowed.

    Returns:
        file_content (str): The content of the file.
    """
    try:
        result = file_system.cat(file_name)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def find(path: str = ".", name: str = None) -> str:
    """
    Find any file or directories under specific path that contain name in its file name.

    This method searches for files of any extension and directories within a specified path that match
    the given name. If no name is provided, it returns all files and directories
    in the specified path and its subdirectories.
    Note: This method performs a recursive search through all subdirectories of the given path.

    Args:
        path (str): The directory path to start the search. Defaults to the current directory (".").
        name (str): [Optional] The name of the file or directory to search for. If None, all items are returned.

    Returns:
        matches (List[str]): A list of matching file and directory paths relative to the given path.
    """
    try:
        result = file_system.find(path, name)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def wc(file_name: str, mode: str = "l") -> str:
    """
    Count the number of lines, words, and characters in a file of any extension from current directory.

    Args:
        file_name (str): Name of the file of current directory to perform wc operation on.
        mode (str): Mode of operation ('l' for lines, 'w' for words, 'c' for characters).

    Returns:
        count (int): The count of the number of lines, words, or characters in the file.
        type (str): The type of unit we are counting. [Enum]: ["lines", "words", "characters"]
    """
    try:
        result = file_system.wc(file_name, mode)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def sort(file_name: str) -> str:
    """
    Sort the contents of a file line by line.

    Args:
        file_name (str): The name of the file appeared at current directory to sort.

    Returns:
        sorted_content (str): The sorted content of the file.
    """
    try:
        result = file_system.sort(file_name)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def grep(file_name: str, pattern: str) -> str:
    """
    Search for lines in a file of any extension at current directory that contain the specified pattern.

    Args:
        file_name (str): The name of the file to search. No path is allowed and you can only perform on file at local directory.
        pattern (str): The pattern to search for.

    Returns:
        matching_lines (List[str]): Lines that match the pattern.
    """
    try:
        result = file_system.grep(file_name, pattern)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def du(human_readable: bool = False) -> str:
    """
    Estimate the disk usage of a directory and its contents.

    Args:
        human_readable (bool): If True, returns the size in human-readable format (e.g., KB, MB).

    Returns:
        disk_usage (str): The estimated disk usage.
    """
    try:
        result = file_system.du(human_readable)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def tail(file_name: str, lines: int = 10) -> str:
    """
    Display the last part of a file of any extension.

    Args:
        file_name (str): The name of the file to display. No path is allowed and you can only perform on file at local directory.
        lines (int): The number of lines to display from the end of the file. Defaults to 10.

    Returns:
        last_lines (str): The last part of the file.
    """
    try:
        result = file_system.tail(file_name, lines)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def diff(file_name1: str, file_name2: str) -> str:
    """
    Compare two files of any extension line by line at the current directory.

    Args:
        file_name1 (str): The name of the first file in current directory.
        file_name2 (str): The name of the second file in current directorry.

    Returns:
        diff_lines (str): The differences between the two files.
    """
    try:
        result = file_system.diff(file_name1, file_name2)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def mv(source: str, destination: str) -> str:
    """
    Move a file or directory from one location to another. so

    Args:
        source (str): Source name of the file or directory to move. Source must be local to the current directory.
        destination (str): The destination name to move the file or directory to. Destination must be local to the current directory and cannot be a path. If destination is not an existing directory like when renaming something, destination is the new file name.

    Returns:
        result (str): The result of the move operation.
    """
    try:
        result = file_system.mv(source, destination)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def rm(file_name: str) -> str:
    """
    Remove a file or directory.

    Args:
        file_name (str): The name of the file or directory to remove.

    Returns:
        result (str): The result of the remove operation.
    """
    try:
        result = file_system.rm(file_name)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def rmdir(dir_name: str) -> str:
    """
    Remove a directory at current directory.

    Args:
        dir_name (str): The name of the directory to remove. Directory must be local to the current directory.

    Returns:
        result (str): The result of the remove operation.
    """
    try:
        result = file_system.rmdir(dir_name)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def cp(source: str, destination: str) -> str:
    """
    Copy a file or directory from one location to another.

    If the destination is a directory, the source file or directory will be copied
    into the destination directory.

    Both source and destination must be local to the current directory.

    Args:
        source (str): The name of the file or directory to copy.
        destination (str): The destination name to copy the file or directory to.
                        If the destination is a directory, the source will be copied
                        into this directory. No file paths allowed.

    Returns:
        result (str): The result of the copy operation or an error message if the operation fails.
    """
    try:
        result = file_system.cp(source, destination)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("\nStarting MCP File System Management Server...")
    mcp.run(transport='stdio')