from abc import ABC, abstractmethod
from typing import Any, Tuple
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class LLMInterface(ABC):
    """
    Abstract base class for integrating Large Language Models (LLMs) into a 
    competitive programming context.

    Attributes:
        prompt (str): Initial prompt to set the context for the LLM.
    """

    def __init__(self):
        """
        Initialize the LLMInterface with a predefined prompt for generating 
        competitive programming solutions.
        """
        self.prompt = """
        You are a competitive programmer. You will be given a problem statement, please implement solution in C++. The execution time and memory limit are also stated in the statement so be aware of the complexity of the program. Please wrap the code in ```cpp and ``` so that it is properly formatted.
        """

    @abstractmethod
    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Abstract method to interact with the LLM.

        Args:
            user_prompt (str): The prompt containing the problem statement for the LLM.

        Returns:
            Tuple[str, Any]: A tuple containing the generated solution and additional metadata.
        """
        pass

    def generate_solution(self, problem_statement: str) -> Tuple[str, Any]:
        """
        Generates a solution to a given competitive programming problem using the LLM.

        Args:
            problem_statement (str): The competitive programming problem statement.

        Returns:
            Tuple[str, Any]: The generated solution and associated metadata.
        """
        user_prompt = self.prompt + problem_statement
        response, meta = self.call_llm(user_prompt)
        return response, meta


class GeminiLLM(LLMInterface):
    """
    Concrete implementation of LLMInterface using Google's Gemini 1.5 Pro model.

    Attributes:
        model (genai.GenerativeModel): Instance for interacting with the Gemini API.
    """

    def __init__(self):
        """
        Initializes the GeminiLLM class by configuring the API key and creating an 
        instance of the Gemini model.
        """
        super().__init__()
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            genai.configure(api_key=api_key)
            # Using a powerful and recent model. You can change this to other available models.
            self.model = genai.GenerativeModel('gemini-2.5-pro')
        except Exception as e:
            print(f"Error during Gemini initialization: {e}")
            self.model = None

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the user prompt to the Gemini model and retrieves the solution.
        """
        if not self.model:
            return "Error: Model not initialized.", None
            
        try:
            # Add the request_options parameter to set a timeout
            # Timeout is in seconds. 600 seconds = 10 minutes.
            response = self.model.generate_content(
                user_prompt,
                request_options={"timeout": 600} 
            )
            solution_text = response.text
            return solution_text, response
        except Exception as e:
            print(f"An error occurred while calling the Gemini API: {e}")
            return f"Error: {e}", None

if __name__ == "__main__":
    """
    Example execution demonstrating how to use the GeminiLLM class to generate solutions.
    """
    llm = GeminiLLM()

    # IMPORTANT: Replace this path with the actual path to your problem statement file.
    problem_file_path = r'C:\Users\alexd\OneDrive\Documents\GitHub\CompetitveProgrammingLLM\Permutation\Permutation_Zip\statement.txt' 

    if not os.path.exists(problem_file_path):
        print(f"Error: The file '{problem_file_path}' was not found.")
        print("Please create the file with a problem statement or update the path in the script.")
        # As a fallback, create a dummy file for the script to run
        with open(problem_file_path, 'w', encoding='utf-8') as file:
            file.write("This is a placeholder problem. Please replace it with a real one.")
        print(f"A placeholder file has been created at '{problem_file_path}'.")


    try:
        with open(problem_file_path, 'r', encoding='utf-8') as file:
            statement = file.read()
        
        print("Generating solution with Gemini...")
        response, meta = llm.generate_solution(statement)
        
        if response and "Error:" not in response:
            output_filename = 'permutation_gemini_solution.txt'
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Solution successfully saved to {output_filename}")
            print("\n--- Metadata ---")
            print(meta)
        else:
            print("\nFailed to generate a valid solution.")
            print(f"Response from API: {response}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
