from litellm import api_base
import dspy
import os
import dspy
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

def main():
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
    dspy.configure(lm=lm)

    math = dspy.ChainOfThought("question -> ansert: float")
    res = math(question="Two dice are tossed. What is probability of getting sum more than 10 ?")
    print(math.inspect_history())
    
    print(res)

def example_to_understand_dspy_signatures():
    signatures = {
        "QuestionAnswering": "question -> answer",
        "SentimentClassification": "sentence -> sentiment",
        "Summarization": [
            "document -> summary",
            "text -> gist",
            "long_context -> tldr"
        ],
        "RAG": "context, question -> answer",
        "Multi-Choice": "question, choices -> reasoning, selection",
    }



def example_to_use_multiple_models():
    predict_qa = dspy.Predict("question -> answer")
    question = "Who is the fastest man and woman ?"

    # Using different model for different purpose
    gemini = dspy.LM("gemini/gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
    mistral = dspy.LM("ollama/mistral", api_base="http://localhost:11434")
    qwen = dspy.LM("ollama/qwen2:7b", api_base="http://localhost:11434")

    with dspy.context(lm=mistral):
        print("Response by mistral: ", predict_qa(question=question))

    with dspy.context(lm=qwen):
        print("Response by qwen: ", predict_qa(question=question))

    with dspy.context(lm=gemini):
        print("Response by Gemini: ", predict_qa(question=question))


if __name__ == "__main__":
    example_to_use_multiple_models()
