# -*- coding: utf-8 -*-
import os
import openai
import time
import re

from cidpro import create_question_with_search, get_qa, create_answer, get_source_files
from vectorstore import read_from_url, read_from_source_dict

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

openai.api_key = os.environ["OPENAI_API_KEY"]

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])


persist_directory = 'db'
vectorstore = read_from_source_dict(get_source_files(), persist_directory)

existing_source = set([k["source"] for k in vectorstore._collection.get()["metadatas"]])

url_pattern = re.compile(r'https?://\S+')

# 問題及び回答のモデル
qa = get_qa(vectorstore)

def get_vectorstore(message):
    url = ""
    match = re.search(url_pattern, message)
    if match:
        url = match.group()
        url = url.replace(">", "")
    if url != "" and url not in existing_source:
        return read_from_url(url), False
    return vectorstore, True

@app.action("button_click")
def action_button_click(body, ack, say):
    ack()

    # print(body)
    # print(body['message'])
    print(body['message']['ts'])
    print(body['channel']['id'])

    question = body['message']['blocks'][0]['text']['text']

    print(question)

    # ボタン削除
    response = client.chat_update(
        channel=body["channel"]["id"],
        ts=body["message"]["ts"],
        text=question,
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": question},
            }
        ]
    )

    # vectorestore
    _vectorstore, exists = get_vectorstore(question)

    # vectorestore
    answer = create_answer(question, _vectorstore)

    # スレッドに返信する
    app.client.chat_postMessage(
        channel=body['channel']['id'],
        thread_ts=body['message']['ts'],
        text=answer,
    )


# 問題を作成して投稿する
@app.message("<@U0554E73FCH>")
def question(message, say):

    print(message)

    if f"<@U0554E73FCH>" not in message['text']:
        return

    # TODO: 最初にリアクションつける
    # app.client.reactions_add(
    #     channel=message["channel"],
    #     timestamp=message["ts"],
    #     name="eyes",
    # )

    # question
    if 'thread_ts' not in message.keys():
        text = message['text'].replace(f"<@U0554E73FCH>", "").strip()
        _vectorstore, exists = get_vectorstore(text)
        if exists:
            question = create_question_with_search(text, _vectorstore)
        else:
            question = create_question_with_search("", _vectorstore)


        # 問題を投稿する
        say(
            text=question,
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": question},
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text":"Answer"},
                        "action_id": "button_click"
                    }
                }
            ],
            )
        return

    # followup
    thread_ts = message["thread_ts"]
    channel_id = message["channel"]
    replies = client.conversations_replies(channel=channel_id, ts=thread_ts)

    # 最初の投稿がボットのものだけ処理する
    if replies["messages"][0].get("user") != "U0554E73FCH":
        return

    question = replies["messages"][0].get("text")

    # vectorestore
    _vectorstore, exists = get_vectorstore(question)

    # 回答
    qa = get_qa(_vectorstore)
    result = qa({"question": question, "chat_history": _chat_history(replies)})

    response_msg = f"{result['answer']}\n\n"
    for d in set([u.metadata['source'] for u in result['source_documents']]):
        response_msg += f"{d}\n"

    # スレッドに返信する
    app.client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text=response_msg,
    )

def _chat_history(replies):
    # chat_historyを作る
    chat_history = [("user", replies["messages"][0].get("text"))]
    for rep in replies["messages"][1:]:
        if rep.get("user") == "U052DPC05NW":
            chat_history.append(("assistant", rep.get("text")))
        else:
            chat_history.append(("user", rep.get("text")))
    return chat_history

# アプリを起動します
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
