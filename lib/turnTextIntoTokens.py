# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
import tiktoken
# encoding = tiktoken.get_encoding("cl100k_base")
# # Use tiktoken.encoding_for_model() to automatically load the correct encoding for a given model name.
# encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
# encoding.encode("tiktoken is great!")
# # [83, 1609, 5963, 374, 2294, 0]

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# count = num_tokens_from_string("tiktoken is great!", "cl100k_base")
# print(count)