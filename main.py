import os
import langchain
import argparse

from cidpro import create_question_with_search, get_qa, create_answer, get_source_files
from vectorstore import read_from_std, read_from_url, read_from_source_dict

parser = argparse.ArgumentParser()
parser.add_argument('--stdin', action='store_true', help='enable std input mode')
parser.add_argument('--url', type=str, help='question from url contents')
args = parser.parse_args()

def main():
    # vector store
    topic = ""
    if args.stdin:
        vectorstore = read_from_std()
    elif args.url:
        vectorstore = read_from_url(args.url)
    else:
        persist_directory = 'db'
        vectorstore = read_from_source_dict(get_source_files(), persist_directory)
        topic = input("Topic/Category: ")

    qa = get_qa(vectorstore)

    # while True:
    for i in range(3):
        # 問題提示
        print("")

        # for i in range(10):
        question = create_question_with_search(topic, vectorstore)
        print(question)
        chat_history = [("user", question)]

        # 回答の提示
        print("")
        # input("<Enter for Answer>")
        answer = create_answer(question, vectorstore)
        print(answer)
        chat_history.append(("assistant", answer))

        # # 回答に対する質問
        # while True:
        #     print("")
        #     query = input("You: ")
        #     if query.lower() == "q":
        #         break
        #
        #     result = qa({"question": query, "chat_history": chat_history})
        #     chat_history.append(("user", query))
        #     chat_history.append(("assistant", result["answer"]))
        #
        #     print("")
        #     print(f"GPT: {result['answer']}")
        #     print(f"{result['source_documents'][1].metadata['source']}")

    # 回答に対する質問
    while True:
        print("")
        query = input("You: ")
        if query.lower() == "q":
            break

        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append(("user", query))
        chat_history.append(("assistant", result["answer"]))

        print("")
        print(f"GPT: {result['answer']}")
        print(f"{result['source_documents'][1].metadata['source']}")


if __name__ == "__main__":
    main()
