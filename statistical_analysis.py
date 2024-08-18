import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Initialize OpenAI client with API key and custom base URL
from openai import OpenAI

client = OpenAI(
    api_key="xxxx", #Put your api keys here
    base_url="xxxx" #Put your  base url here
)

def greatest_read_csv(file_path, encodings=None):
    """
    Attempts to read a CSV file with pandas, trying multiple encodings and handling common errors.
    
    Args:
        file_path (str): The full path to the CSV file.
        encodings (list, optional): A list of encodings to try. Defaults to a commonly used set of encodings.
        
    Returns:
        pd.DataFrame: The loaded DataFrame if successful.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file could not be read with any of the specified encodings.
    """
    if encodings is None:
        encodings = ['utf-8', 'ISO-8859-1', 'Windows-1252']

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    for encoding in encodings:
        try:
            print(f"Attempting to read the file with {encoding} encoding...")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"File successfully read with {encoding} encoding!")
            return df
        except UnicodeDecodeError:
            print(f"Failed to read file with {encoding} encoding. Trying next encoding...")
        except Exception as e:
            print(f"An error occurred while reading the file with {encoding} encoding: {e}")

    # If all encodings fail, raise an error
    raise ValueError(f"Could not read the file with any of the provided encodings: {encodings}")

def generate_system_prompt(dataframe):
    """Generate a system prompt to instruct the AI on how to create the code."""
    return f"""
    You are an intelligent code generator. You have been provided with a Pandas DataFrame:
    {dataframe.head().to_string()}

    Your task is to write a Python function using Pandas that can perform the analysis and plotting as requested by the user.

    The function should:
    - Be valid and executable in Python.
    - Perform the analysis or operations requested by the user without referencing any external files or needing to access any specific CSV files don't mention any files in example usage either.
    - Use Matplotlib to generate any necessary plots or visualizations.
    - Use the 'dataframe' variable directly for any operations.
    - Include data validation steps such as converting data types and handling missing values.
    - Provide a brief explanation or summary of the analysis results in a section labeled 'Explanation:'
    - Return the analysis results, the generated plots, and the explanation.

    Remember to include all necessary imports and ensure the code is ready to execute without modification.
    """

def generate_pandas_code_and_plot(system_prompt, user_prompt):
    """Generate Python code based on the system and user prompts."""
    # Combine system and user prompts to create a comprehensive instruction for the AI
    prompt = f"""
    {system_prompt}
    
    The user has requested the following:
    {user_prompt}
    
    Based on the above, generate the complete Python code with data validation, preprocessing steps, and an explanation section.
    """
    
    # Generate a completion using the GPT-4o-mini model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}]
    )

    # Extract the generated code from the response
    code = response.choices[0].message.content

    # Ensure code is not None and clean it up
    if code:
        code = code.strip()
        if "```" in code:
            code = code.replace("```python", "").replace("```", "").strip()

    return code

def save_generated_code(code, filename='generated_code.py'):
    """Save the generated code to a Python file in the current working directory."""
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, filename)
    
    with open(file_path, 'w') as f:
        f.write(code)
    
    print(f"Generated code saved to {file_path}")

def log_execution_results(result, error=None, log_file='execution_log.txt'):
    """Log the execution results and errors."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as f:
        f.write(f"--- {timestamp} ---\n")
        if result is not None:
            f.write(f"Result:\n{result}\n")
        if error is not None:
            f.write(f"Error:\n{error}\n")
    print(f"Execution results logged to {log_file}")

def execute_code(code, dataframe):
    """Execute the generated code."""
    local_context = {'dataframe': dataframe}
    
    if not code:
        print("No valid code was generated.")
        return None

    save_generated_code(code)

    user_input = input("Do you want to execute the following code? (y/n)\n\n" + code + "\n\n")
    if user_input.lower() != 'y':
        print("Execution aborted by user.")
        return None

    try:
        exec(code, globals(), local_context)
        plt.show()
        result = local_context.get('result', None)
        log_execution_results(result)
        return result
    except Exception as e:
        log_execution_results(None, error=str(e))
        print(f"Error executing code: {e}")
        return None

def describe_output(result):
    """Describe the output received after code execution."""
    if result is None:
        print("No result to describe.")
        return

    if isinstance(result, pd.DataFrame):
        print("\nOutput is a DataFrame. Here's a summary:")
        print(result.describe())
    elif isinstance(result, pd.Series):
        print("\nOutput is a Series. Here's a summary:")
        print(result.describe())
    elif isinstance(result, dict):
        print("\nOutput is a Dictionary. Keys are:")
        for key in result.keys():
            print(f"- {key}")
    elif isinstance(result, (list, tuple)):
        print(f"\nOutput is a {type(result).__name__}. Length: {len(result)}")
        print(f"First few items: {result[:5]}")
    else:
        print(f"\nOutput is of type {type(result).__name__}. Value: {result}")
def display_dataframe_summary(dataframe):
    """Display a summary of the DataFrame including descriptive statistics and missing values."""
    print("\nDataFrame Summary:")
    print(dataframe.describe())
    print("\nMissing Values:")
    print(dataframe.isnull().sum())

# Example usage
if __name__ == "__main__":
    # Path to the CSV file
    file_path = r"Path/to/your/file"
    # Use greatest_read_csv to load the DataFrame
    try:
        df = greatest_read_csv(file_path)
        print("DataFrame loaded successfully!")
        display_dataframe_summary(df)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ValueError as val_error:
        print(val_error)
    
    # User prompt: Ask any question about the data
    user_prompt = input("Enter your question or request regarding the data analysis: ")

    # Generate the system prompt
    system_prompt = generate_system_prompt(df)
    
    # Generate Python code based on system and user prompts
    generated_code = generate_pandas_code_and_plot(system_prompt, user_prompt)
    print("Generated Code:")
    print(generated_code)
    
    # Execute the generated code and get the result
    result = execute_code(generated_code, df)
    print("Result:")
    print(result)
    
    # Describe the output
    describe_output(result)
