import os
import inspect
import logging
import textwrap
import importlib.util as importutil

import_command = """
import torch
import torch.nn as nn
import torch.nn.functional as F
"""


def read_python_file(file_path):
    try:
        with open(file_path, "r") as f:
            code_string = f.read()
        return code_string
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
    except Exception as e:
        logging.error(f"Error occurred while reading file {file_path}: {e}")
    return ""


def extract_code_section(code_string: str):
    """
    This function converts a generated code string to a Python file.
    It loops through each line in the code string and writes it to the file.
    If it encounters a line containing "return", it stops writing further lines.

    Args:
        code_string (str): The generated code as a string.
    Returns:
        str: a string containing the extracted line of code, including an import command.

    """
    try:
        is_net = False
        lines = code_string.splitlines()
        extracted_code = []
        for line in lines:
            extracted_code.append(line)
            if line.strip().startswith("return") or "return" in line.strip():
                is_net = True
                break

        if not is_net:
            return ""
        residual_class_code = "\n".join(extracted_code)
        return import_command + "\n" + residual_class_code

    except Exception as e:
        logging.error(f"Error occurred while extracting code section: {e}")
        return ""


def write_codestring_to_file(code_string: str, output_file_name: str):
    """This function supports to export generated code to a file.
    Args:
        code_string (str): The generated code as a string
        filename (str): The name of the Python file to
    """
    # remove any common leading whitespace from code_string
    code_string = textwrap.dedent(code_string)
    with open(f"{output_file_name}", "w") as f:
        f.write(code_string)


def get_class(file_path):
    """Extract class from code in file .py"""
    module = os.path.basename(file_path)
    spec = importutil.spec_from_file_location(module[:-3], file_path)
    module = importutil.module_from_spec(spec)
    spec.loader.exec_module(module)
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            return obj
    return None


def create_instance(cls, *args, **kwargs):
    """Create an instance of a class with optional arguments. Use it instead of cls() normal.
    Args:
        cls (class): The class to create an instance of.
        *args: Optional positional arguments to pass to the class constructor.
        **kwargs: Optional keyword arguments to pass to the class constructor.
    Returns:
         An instance of the class with the given arguments.
    """
    # get signature of __init__
    signature = inspect.signature(cls.__init__)
    parameters = signature.parameters

    # process in case there isn't any signature except "self"
    if len(parameters) == 1 and "self" in parameters:
        return cls()

    # process in case need parameters
    init_args = {}
    for name, param in parameters.items():
        if name == "self":
            continue
        if name in kwargs:
            init_args[name] = kwargs[name]
        elif param.default != inspect.Parameter.empty:
            init_args[name] = param.default
        else:
            raise TypeError(f"Missing required argument: '{name}'")
    return cls(**init_args)
