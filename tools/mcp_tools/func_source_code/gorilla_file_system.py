import datetime
import subprocess
from copy import deepcopy
from typing import Dict, List, Optional, Union

from .long_context import (FILE_CONTENT_EXTENSION, FILES_TAIL_USED, POPULATE_FILE_EXTENSION)


class File:

    def __init__(self, name: str, content: str = "") -> None:
        """
        Initialize a file with a name and optional content.

        Args:
            name (str): The name of the file.
            content (str, optional): The initial content of the file. Defaults to an empty string.
        """
        self.name: str = name
        self.content: str = content
        self._last_modified: datetime.datetime = datetime.datetime.now()

    def _write(self, new_content: str) -> None:
        """
        Write new content to the file and update the last modified time.

        Args:
            new_content (str): The new content to write to the file.
        """
        self.content = new_content
        self._last_modified = datetime.datetime.now()

    def _read(self) -> str:
        """
        Read the content of the file.

        Returns:
            content (str): The current content of the file.
        """
        return self.content

    def _append(self, additional_content: str) -> None:
        """
        Append content to the existing file content.

        Args:
            additional_content (str): The content to append to the file.
        """
        self.content += additional_content
        self._last_modified = datetime.datetime.now()

    def __repr__(self):
        return f"<<File: {self.name}, Content: {self.content}>>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, File):
            return False
        return self.name == other.name and self.content == other.content


class Directory:

    def __init__(self, name: str, parent: Optional["Directory"] = None) -> None:
        """
        Initialize a directory with a name.

        Args:
            name (str): The name of the directory.
        """
        self.name: str = name
        self.parent: Optional["Directory"] = parent
        self.contents: Dict[str, Union["File", "Directory"]] = {}

    def _add_file(self, file_name: str, content: str = "") -> None:
        """
        Add a new file to the directory.

        Args:
            file_name (str): The name of the file.
            content (str, optional): The content of the new file. Defaults to an empty string.
        """
        if file_name in self.contents:
            raise ValueError(
                f"File '{file_name}' already exists in directory '{self.name}'."
            )
        new_file = File(file_name, content)
        self.contents[file_name] = new_file

    def _add_directory(self, dir_name: str) -> None:
        """
        Add a new subdirectory to the directory.

        Args:
            dir_name (str): The name of the subdirectory.
        """
        if dir_name in self.contents:
            raise ValueError(
                f"Directory '{dir_name}' already exists in directory '{self.name}'."
            )
        new_dir = Directory(dir_name, self)
        self.contents[dir_name] = new_dir

    def _get_item(self, item_name: str) -> Union["File", "Directory", None]:
        """
        Get an item (file or subdirectory) from the directory.

        Args:
            item_name (str): The name of the item to retrieve.

        Returns:
            item (any): The retrieved item or None if it does not exist.
        """
        if item_name == ".":
            return self
        return self.contents.get(item_name)

    def _list_contents(self) -> List[str]:
        """
        List the names of all contents in the directory.

        Returns:
            contents (List[str]): A list of names of the files and subdirectories in the directory.
        """
        return list(self.contents.keys())

    def __repr__(self):
        return f"<Directory: {self.name}, Parent: {self.parent.name if self.parent else None}, Contents: {self.contents}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Directory):
            return False
        return self.name == other.name and self.contents == other.contents


DEFAULT_STATE = {"root": Directory("/", None)}


class GorillaFileSystem:

    def __init__(self) -> None:
        """
        Initialize the Gorilla file system with a root directory
        """
        # 初始化根目录
        self.root: Directory = Directory("/", None)
        self._current_dir: Directory = self.root
        self._api_description = "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc."

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GorillaFileSystem):
            return False
        return self.root == other.root

    def _load_scenario(self, scenario: dict, long_context: bool = False) -> None:
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
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self.long_context = long_context
        self.root = DEFAULT_STATE_COPY["root"]
        if "root" in scenario:
            root_dir = Directory(list(scenario["root"].keys())[0], None)
            self.root = self._load_directory(
                scenario["root"][list(scenario["root"].keys())[0]]["contents"], root_dir
            )
        self._current_dir = self.root

    def _load_directory(
        self, current: dict, parent: Optional[Directory] = None
    ) -> Directory:
        """
        Load a directory and its contents from a dictionary.

        Args:
            data (dict): The dictionary representing the directory.
            parent (Directory, optional): The parent directory. Defaults to None.

        Returns:
            Directory: The loaded directory.
        """
        is_bottommost = True
        for dir_name, dir_data in current.items():

            if dir_data["type"] == "directory":
                is_bottommost = False
                new_dir = Directory(dir_name, parent)
                new_dir = self._load_directory(dir_data["contents"], new_dir)
                parent.contents[dir_name] = new_dir

            elif dir_data["type"] == "file":
                content = dir_data["content"]
                if self.long_context and dir_name not in FILES_TAIL_USED:
                    content += FILE_CONTENT_EXTENSION
                new_file = File(dir_name, content)
                parent.contents[dir_name] = new_file

        if is_bottommost and self.long_context:
            self._populate_directory(parent)

        return parent

    def _populate_directory(
        self, directory: Directory
    ) -> None:  # Used only for long context
        """
        Populate an innermost directory with multiple empty files.

        Args:
            directory (Directory): The innermost directory to populate.
        """
        for i in range(len(POPULATE_FILE_EXTENSION)):
            name = POPULATE_FILE_EXTENSION[i]
            file_name = f"{name}"
            directory._add_file(file_name)

    def pwd(self):
        """
        Return the current working directory path.
        Args:
            None
        Returns:
            current_working_directory (str): The current working directory path.

        """
        path = []
        dir = self._current_dir
        while dir is not None and dir.name != self.root:
            path.append(dir.name)
            dir = dir.parent
        return {"current_working_directory": "/" + "/".join(reversed(path))}

    def ls(self, a: bool = False) -> Dict[str, List[str]]:
        """
        List the contents of the current directory.

        Args:
            a (bool): [Optional] Show hidden files and directories. Defaults to False.

        Returns:
            current_directory_content (List[str]): A list of the contents of the specified directory.
        """
        contents = self._current_dir._list_contents()
        if not a:
            contents = [item for item in contents if not item.startswith(".")]
        return {"current_directory_content": contents}

    def cd(self, folder: str) -> Union[None, Dict[str, str]]:
        """
        Change the current working directory to the specified folder.

        Args:
            folder (str): The folder of the directory to change to. You can only change one folder at a time.

        Returns:
            current_working_directory (str): The new current working directory path.
        """
        # Handle navigating to the parent directory with "cd .."
        if folder == "..":
            if self._current_dir.parent:
                self._current_dir = self._current_dir.parent
            elif self.root == self._current_dir:
                return {"error": "Cuurent directory is already the root. Cannot go back."}
            else:
                return {"error": "cd: ..: No such directory"}
            return {}

        # Handle absolute or relative paths
        target_dir = self._navigate_to_directory(folder)
        if isinstance(target_dir, dict):  # Error condition check
            return {
                "error": f"cd: {folder}: No such directory. You cannot use path to change directory."
            }
        self._current_dir = target_dir
        return {"current_working_directory": target_dir.name}

    def _validate_file_or_directory_name(self, dir_name: str) -> bool:
        if any(c in dir_name for c in '|/\\?%*:"><'):
            return False
        return True

    def mkdir(self, dir_name: str) -> Union[None, Dict[str, str]]:
        """
        Create a new directory in the current directory.

        Args:
            dir_name (str): The name of the new directory at current directory. You can only create directory at current directory.
        """
        if not self._validate_file_or_directory_name(dir_name):
            return {
                "error": f"mkdir: cannot create directory '{dir_name}': Invalid character"
            }
        if dir_name in self._current_dir.contents:
            return {"error": f"mkdir: cannot create directory '{dir_name}': File exists"}

        self._current_dir._add_directory(dir_name)
        return None

    def touch(self, file_name: str) -> Union[None, Dict[str, str]]:
        """
        Create a new file of any extension in the current directory.

        Args:
            file_name (str): The name of the new file in the current directory. file_name is local to the current directory and does not allow path.
        """
        if not self._validate_file_or_directory_name(file_name):
            return {"error": f"touch: cannot touch '{file_name}': Invalid character"}

        if file_name in self._current_dir.contents:
            return {"error": f"touch: cannot touch '{file_name}': File exists"}

        self._current_dir._add_file(file_name)
        return None

    def echo(
        self, content: str, file_name: Optional[str] = None
    ) -> Union[Dict[str, str], None]:
        """
        Write content to a file at current directory or display it in the terminal.

        Args:
            content (str): The content to write or display.
            file_name (str): [Optional] The name of the file at current directory to write the content to. Defaults to None.

        Returns:
            terminal_output (str): The content if no file name is provided, or None if written to file.
        """
        if file_name is None:
            return {"terminal_output": content}
        if not self._validate_file_or_directory_name(file_name):
            return {"error": f"echo: cannot touch '{file_name}': Invalid character"}

        if file_name:
            if file_name in self._current_dir.contents:
                self._current_dir._get_item(file_name)._write(content)
            else:
                self._current_dir._add_file(file_name, content)
        else:
            return {"terminal_output": content}

    def cat(self, file_name: str) -> Dict[str, str]:
        """
        Display the contents of a file of any extension from currrent directory.

        Args:
            file_name (str): The name of the file from current directory to display. No path is allowed.

        Returns:
            file_content (str): The content of the file.
        """
        if not self._validate_file_or_directory_name(file_name):
            return {"error": f"cat: '{file_name}': Invalid character"}

        if file_name in self._current_dir.contents:
            item = self._current_dir._get_item(file_name)
            if isinstance(item, File):
                return {"file_content": item._read()}
            else:
                return {"error": f"cat: {file_name}: Is a directory"}
        else:
            return {"error": f"cat: {file_name}: No such file or directory"}

    def find(self, path: str = ".", name: Optional[str] = None) -> Dict[str, List[str]]:
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
        matches = []
        target_dir = self._current_dir

        def recursive_search(directory: Directory, base_path: str) -> None:
            for item_name, item in directory.contents.items():
                item_path = f"{base_path}/{item_name}"
                if name is None or name in item_name:
                    matches.append(item_path)
                if isinstance(item, Directory):
                    recursive_search(item, item_path)

        recursive_search(target_dir, path.rstrip("/"))
        return {"matches": matches}

    def wc(self, file_name: str, mode: str = "l") -> Dict[str, Union[int, str]]:
        """
        Count the number of lines, words, and characters in a file of any extension from current directory.

        Args:
            file_name (str): Name of the file of current directory to perform wc operation on.
            mode (str): Mode of operation ('l' for lines, 'w' for words, 'c' for characters).

        Returns:
            count (int): The count of the number of lines, words, or characters in the file.
            type (str): The type of unit we are counting. [Enum]: ["lines", "words", "characters"]
        """
        if mode not in ["l", "w", "c"]:
            return {"error": f"wc: invalid mode '{mode}'"}

        if file_name in self._current_dir.contents:
            file = self._current_dir._get_item(file_name)
            if isinstance(file, File):
                content = file._read()

                if mode == "l":
                    line_count = len(content.splitlines())
                    return {"count": line_count, "type": "lines"}

                elif mode == "w":
                    word_count = len(content.split())
                    return {"count": word_count, "type": "words"}

                elif mode == "c":
                    char_count = len(content)
                    return {"count": char_count, "type": "characters"}

        return {"error": f"wc: {file_name}: No such file or directory"}

    def sort(self, file_name: str) -> Dict[str, str]:
        """
        Sort the contents of a file line by line.

        Args:
            file_name (str): The name of the file appeared at current directory to sort.

        Returns:
            sorted_content (str): The sorted content of the file.
        """
        if file_name in self._current_dir.contents:
            file = self._current_dir._get_item(file_name)
            if isinstance(file, File):
                content = file._read()

                sorted_content = "\n".join(sorted(content.splitlines()))

                return {"sorted_content": sorted_content}

        return {"error": f"sort: {file_name}: No such file or directory"}

    def grep(self, file_name: str, pattern: str) -> Dict[str, List[str]]:
        """
        Search for lines in a file of any extension at current directory that contain the specified pattern.

        Args:
            file_name (str): The name of the file to search. No path is allowed and you can only perform on file at local directory.
            pattern (str): The pattern to search for.

        Returns:
            matching_lines (List[str]): Lines that match the pattern.
        """
        if file_name in self._current_dir.contents:
            file = self._current_dir._get_item(file_name)
            if isinstance(file, File):
                content = file._read()

                matching_lines = [line for line in content.splitlines() if pattern in line]

                return {"matching_lines": matching_lines}

        return {"error": f"grep: {file_name}: No such file or directory"}

    def du(self, human_readable: bool = False) -> Dict[str, str]:
        """
        Estimate the disk usage of a directory and its contents.

        Args:
            human_readable (bool): If True, returns the size in human-readable format (e.g., KB, MB).

        Returns:
            disk_usage (str): The estimated disk usage.
        """

        def get_size(item: Union[File, Directory]) -> int:
            if isinstance(item, File):
                return len(item._read().encode("utf-8"))
            elif isinstance(item, Directory):
                return sum(get_size(child) for child in item.contents.values())
            return 0

        target_dir = self._navigate_to_directory(None)
        if isinstance(target_dir, dict):  # Error condition check
            return target_dir

        total_size = get_size(target_dir)

        if human_readable:
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if total_size < 1024:
                    size_str = f"{total_size:.2f} {unit}"
                    break
                total_size /= 1024
            else:
                size_str = f"{total_size:.2f} PB"
        else:
            size_str = f"{total_size} bytes"

        return {"disk_usage": size_str}

    def tail(self, file_name: str, lines: int = 10) -> Dict[str, str]:
        """
        Display the last part of a file of any extension.

        Args:
            file_name (str): The name of the file to display. No path is allowed and you can only perform on file at local directory.
            lines (int): The number of lines to display from the end of the file. Defaults to 10.

        Returns:
            last_lines (str): The last part of the file.
        """
        if file_name in self._current_dir.contents:
            file = self._current_dir._get_item(file_name)
            if isinstance(file, File):
                content = file._read().splitlines()

                if lines > len(content):
                    lines = len(content)

                last_lines = content[-lines:]
                return {"last_lines": "\n".join(last_lines)}

        return {"error": f"tail: {file_name}: No such file or directory"}

    def diff(self, file_name1: str, file_name2: str) -> Dict[str, str]:
        """
        Compare two files of any extension line by line at the current directory.

        Args:
            file_name1 (str): The name of the first file in current directory.
            file_name2 (str): The name of the second file in current directorry.

        Returns:
            diff_lines (str): The differences between the two files.
        """
        if (
            file_name1 in self._current_dir.contents
            and file_name2 in self._current_dir.contents
        ):
            file1 = self._current_dir._get_item(file_name1)
            file2 = self._current_dir._get_item(file_name2)

            if isinstance(file1, File) and isinstance(file2, File):
                content1 = file1._read().splitlines()
                content2 = file2._read().splitlines()

                diff_lines = [
                    f"- {line1}\n+ {line2}"
                    for line1, line2 in zip(content1, content2)
                    if line1 != line2
                ]

                return {"diff_lines": "\n".join(diff_lines)}

        return {"error": f"diff: {file_name1} or {file_name2}: No such file or directory"}

    def mv(self, source: str, destination: str) -> Dict[str, str]:
        """
        Move a file or directory from one location to another. so

        Args:
            source (str): Source name of the file or directory to move. Source must be local to the current directory.
            destination (str): The destination name to move the file or directory to. Destination must be local to the current directory and cannot be a path. If destination is not an existing directory like when renaming something, destination is the new file name.

        Returns:
            result (str): The result of the move operation.
        """
        if source not in self._current_dir.contents:
            return {"error": f"mv: cannot move '{source}': No such file or directory"}

        item = self._current_dir._get_item(source)

        if not isinstance(item, (File, Directory)):
            return {"error": f"mv: cannot move '{source}': Not a file or directory"}

        if "/" in destination:
            return {
                "error": f"mv: no path allowed in destination. Only file name and folder name is supported for this operation."
            }

        # Check if the destination is an existing directory
        if destination in self._current_dir.contents:
            dest_item = self._current_dir._get_item(destination)
            if isinstance(dest_item, Directory):
                # Move source into the destination directory
                new_destination = f"{source}"
                if new_destination in dest_item.contents:
                    return {
                        "error": f"mv: cannot move '{source}' to '{destination}/{source}': File exists"
                    }
                else:
                    self._current_dir.contents.pop(source)
                    if isinstance(item, File):
                        dest_item._add_file(source, item.content)
                    else:
                        dest_item._add_directory(source)
                        dest_item.contents[source].contents = item.contents
                    return {"result": f"'{source}' moved to '{destination}/{source}'"}
            else:
                return {
                    "error": f"mv: cannot move '{source}' to '{destination}': Not a directory"
                }
        else:
            # Destination is not an existing directory, move/rename the item
            self._current_dir.contents.pop(source)
            if isinstance(item, File):
                self._current_dir._add_file(destination, item.content)
            else:
                self._current_dir._add_directory(destination)
                self._current_dir.contents[destination].contents = item.contents
            return {"result": f"'{source}' moved to '{destination}'"}

    def rm(self, file_name: str) -> Dict[str, str]:
        """
        Remove a file or directory.

        Args:
            file_name (str): The name of the file or directory to remove.

        Returns:
            result (str): The result of the remove operation.
        """
        if file_name in self._current_dir.contents:
            item = self._current_dir._get_item(file_name)
            if isinstance(item, File) or isinstance(item, Directory):
                self._current_dir.contents.pop(file_name)
                return {"result": f"'{file_name}' removed"}
            else:
                return {
                    "error": f"rm: cannot remove '{file_name}': Not a file or directory"
                }
        else:
            return {"error": f"rm: cannot remove '{file_name}': No such file or directory"}

    def rmdir(self, dir_name: str) -> Dict[str, str]:
        """
        Remove a directory at current directory.

        Args:
            dir_name (str): The name of the directory to remove. Directory must be local to the current directory.

        Returns:
            result (str): The result of the remove operation.
        """
        if dir_name in self._current_dir.contents:
            item = self._current_dir._get_item(dir_name)
            if isinstance(item, Directory):
                if item.contents:  # Check if directory is not empty
                    return {
                        "error": f"rmdir: failed to remove '{dir_name}': Directory not empty"
                    }
                else:
                    self._current_dir.contents.pop(dir_name)
                    return {"result": f"'{dir_name}' removed"}
            else:
                return {"error": f"rmdir: cannot remove '{dir_name}': Not a directory"}
        else:
            return {
                "error": f"rmdir: cannot remove '{dir_name}': No such file or directory"
            }

    def cp(self, source: str, destination: str) -> Dict[str, str]:
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
        if source not in self._current_dir.contents:
            return {"error": f"cp: cannot copy '{source}': No such file or directory"}

        item = self._current_dir._get_item(source)

        if not isinstance(item, (File, Directory)):
            return {"error": f"cp: cannot copy '{source}': Not a file or directory"}

        if "/" in destination:
            return {
                "error": f"cp: don't allow path in destination. Only file name and folder name is supported for this operation."
            }
        # Check if the destination is an existing directory
        if destination in self._current_dir.contents:
            dest_item = self._current_dir._get_item(destination)
            if isinstance(dest_item, Directory):
                # Copy source into the destination directory
                new_destination = f"{destination}/{source}"
                if new_destination in dest_item.contents:
                    return {
                        "error": f"cp: cannot copy '{source}' to '{destination}/{source}': File exists"
                    }
                else:
                    if isinstance(item, File):
                        dest_item._add_file(source, item.content)
                    else:
                        dest_item._add_directory(source)
                        dest_item.contents[source].contents = item.contents.copy()
                    return {"result": f"'{source}' copied to '{destination}/{source}'"}
            else:
                return {
                    "error": f"cp: cannot copy '{source}' to '{destination}': Not a directory"
                }
        else:
            # Destination is not an existing directory, perform the copy
            if isinstance(item, File):
                self._current_dir._add_file(destination, item.content)
            else:
                self._current_dir._add_directory(destination)
                self._current_dir.contents[destination].contents = item.contents.copy()
            return {"result": f"'{source}' copied to '{destination}'"}

    def _navigate_to_directory(
        self, path: Optional[str]
    ) -> Union[Directory, Dict[str, str]]:
        """
        Navigate to a specified directory path from the current directory.

        Args:
            path (str): [Optional] The path to navigate to. Defaults to None (current directory).

        Returns:
            target_directory (Directory or dict): The target directory object or error message.
        """
        if path is None or path == ".":
            return self._current_dir
        elif path == "/":
            return self.root

        dirs = path.strip("/").split("/")
        temp_dir = self._current_dir if not path.startswith("/") else self.root

        for dir_name in dirs:
            next_dir = temp_dir._get_item(dir_name)
            if isinstance(next_dir, Directory):
                temp_dir = next_dir
            else:
                return {"error": f"cd: '{path}': No such file or directory"}

        return temp_dir

    def _parse_positions(self, positions: str) -> List[int]:
        """
        Helper function to parse position strings, e.g., '1,3,5', '1-5', '-3', or '3-'.

        Args:
            positions (str): The position string to parse.

        Returns:
            list (List[int]): A list of integers representing the positions.
        """
        result = []
        if "," in positions:
            for part in positions.split(","):
                result.extend(self._parse_positions(part))
        elif "-" in positions:
            start, end = positions.split("-")
            start = int(start) if start else 1
            end = int(end) if end else float("inf")
            result.extend(range(start, end + 1))
        else:
            result.append(int(positions))
        return result
