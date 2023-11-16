import argparse

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv(".env")


def read_changelog_from_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def summarize_changelog(changelog: str) -> str:
    system_template = """
        Using the provided changelog for our React Native app, generate a 
        concise, upbeat summary for our app's 'What's New' section in the 
        app store. Focus on highlighting the key improvements, new features, 
        and enhancements, while ensuring to remove any technical jargon, 
        personally identifying information, or specific details that might not 
        be relevant to a general audience. The summary should be brief, 
        engaging, and crafted to give users a positive impression of the app's 
        latest update. Aim to encapsulate the overall value and improvements 
        brought by this update in a few sentences. Always end your summary with "Thanks for using the WebstaurantStore app!"
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """
        Changelog: {changelog}
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=ChatOpenAI(), prompt=chat_prompt)

    summary = chain.run({"changelog": changelog})

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate a summary of a changelog for an app's 'What's New' section."
    )
    parser.add_argument("file_path", help="The file path of the changelog text file")
    args = parser.parse_args()

    changelog = read_changelog_from_file(args.file_path)
    summary = summarize_changelog(changelog)
    print(f'Summary: "{summary}"')


if __name__ == "__main__":
    main()
