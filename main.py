import os
import sys
import langchain
import argparse

from cidpro import create_question_with_search, get_qa, create_answer, get_source_files
from vectorstore import read_from_std, read_from_url, read_from_source_dict

parser = argparse.ArgumentParser()
parser.add_argument('--stdin', action='store_true', help='enable std input mode')
parser.add_argument('--url', type=str, help='question from url contents')
parser.add_argument('--chunk_size', default=1000, type=int, help='chunk_size for stdin and url')
parser.add_argument('--chunk_overlap', default=500, type=int, help='chunk_overlap for stdin and url')
parser.add_argument('--show_page_content', action='store_true', help='enable show original sentence')
parser.add_argument('--all', action='store_true', help='enable show original sentence')
args = parser.parse_args()

def main():
    # vector store
    topic = ""
    if args.stdin:
        vectorstore = read_from_std(args.chunk_size, args.chunk_overlap)
        topic = "stdin"
    elif args.url:
        vectorstore = read_from_url(args.url, args.chunk_size, args.chunk_overlap)
    else:
        persist_directory = 'db'
        vectorstore = read_from_source_dict(get_source_files(), persist_directory)
        topic = input("Topic/Category: ")

    print(len(vectorstore._collection.get()["documents"]))

    while True:
        loop_num = len(vectorstore._collection.get()["documents"]) if args.all else 3
        for i in range(loop_num):
            # 問題提示
            print("")

            # for i in range(10):
            if args.all:
                question = create_question_with_search(topic, vectorstore, show_page_content=args.show_page_content, doc_number=i)
            else:
                question = create_question_with_search(topic, vectorstore, args.show_page_content)
            print(question)
            chat_history = [("user", question)]

            # 回答の提示
            print("")
            # input("<Enter for Answer>")
            answer = create_answer(question, vectorstore)
            print(answer)
            chat_history.append(("assistant", answer))

        qa = get_qa(vectorstore, model="gpt-3.5-turbo")

        # 回答に対する質問
        while True:
            print("")
            query = input("You: ")
            if query.lower() == "n":
                break
            if query.lower() == "q":
                sys.exit(1)

            result = qa({"question": query, "chat_history": chat_history})
            chat_history.append(("user", query))
            chat_history.append(("assistant", result["answer"]))

            print("")
            print(f"GPT: {result['answer']}")
            print(f"{result['source_documents'][1].metadata['source']}")


if __name__ == "__main__":
    main()
