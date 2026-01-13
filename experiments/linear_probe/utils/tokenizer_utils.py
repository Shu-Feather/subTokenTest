import re

def tokenizer_remove_trim(tokenizer) -> None:
    """
    (1) Some tokenizers have a `trim` method in their chat template. We should remove it to keep the correct format.
    (2) Turn off the `clean_up_tokenization_spaces` to keep the correct format.
    """
    chat_template = tokenizer.chat_template
    assert isinstance(chat_template, str)
    
    # we replace all |trim' or '| trim' with ''
    chat_template = re.sub(r"\| *trim *", "", chat_template)
    tokenizer.chat_template = chat_template
    tokenizer.clean_up_tokenization_spaces = False