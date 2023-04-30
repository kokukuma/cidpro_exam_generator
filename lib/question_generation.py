import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel, BaseMessage, BaseRetriever, Document
from langchain.vectorstores.base import VectorStore
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain


question_prompt = """
Please create one multiple-choice question related to the specified question and context.

Make sure there is only one correct answer. Please don't provide obviously incorrect options.

Don't reveal the answer.

Feel free to create a question that asks for the best solution in a hypothetical organization's situation.

Use the following examples as reference for the question format:

    Question: (Situation example: Company A has decided to replace its authentication and authorization system as it scales up, which laws should they be concerned about?)
    Choose the correct answer from the options below:
    1. (Option 1)
    2. (Option 2)
    3. (Option 3)
    4. (Option 4)

    Question: (Specific topic example: Authentication) Choose the correct answer from the options below:
    1. (Option 1)
    2. (Option 2)
    3. (Option 3)
    4. (Option 4)

---
{question}:{context}
"""

# question_prompt = """
# 最後に指定されたquestion及びcontextに関する、選択問題を1問、出題してください。
#
# 正解は一つだけにしてください、また正解は言わないでください。明らかに間違ってる選択肢を出さないでください。
#
# 架空の組織におけるシチュエーションの説明があり、そのシチュエーションにおけるベストなソリューションを選ぶような問題にしてもよいです。
#
# 質問の形式は、以下の例を参考にしてください.
#
#     問題: (シチュエーション例: 会社Aでは規模の拡大とともに認証認可のシステムを入れ替えることとした、このときどの法律を気にする必要があるか？)
#     以下の選択肢のうち、正しいものを一つ選択せよ.
#     1. (選択肢1)
#     2. (選択肢2)
#     3. (選択肢3)
#     4. (選択肢4)
#
#     問題: (特定の話題例: 認証)に関して、以下の選択肢のうち、正しいものを一つ選択せよ.
#     1. (選択肢1)
#     2. (選択肢2)
#     3. (選択肢3)
#     4. (選択肢4)
#
# ---
# {question}:{context}
#
# 回答は日本語でお願いします。
# """

QUESTION_PROMPT = PromptTemplate(
    template=question_prompt, input_variables=["context", "question"]
)

answer_prompt = """
Please choose the correct answer for the following question and explain why it is the correct answer.
{question}:{context}
"""
# answer_prompt = """
# 次の質問について正解を選択して、なぜそれが正解なのかを解説してください.
# {question}:{context}
#
# 回答は日本語でお願いします。
# """
ANSWER_PROMPT = PromptTemplate(
    template=answer_prompt, input_variables=["context", "question"]
)

keyword_template = """
Please list 3 random words related to {question}.
"""
# keyword_template = """{question}に関連する単語を、ランダムで3つあげてください.
# 以下は無視してください.
# """
KEYWORD_PROMPT = PromptTemplate.from_template(keyword_template)

class ExamQuestionGenerator():
    vectorstore: VectorStore
    question_chain: BaseCombineDocumentsChain
    answer_chain: BaseCombineDocumentsChain
    keyword_generator: LLMChain
    max_tokens_limit: Optional[int] = None

    def __init__(self,
            vectorstore: VectorStore,
            question_chain: BaseCombineDocumentsChain,
            answer_chain: BaseCombineDocumentsChain,
            keyword_generator: LLMChain,
            ):
        self.vectorstore = vectorstore
        self.question_chain = question_chain
        self.answer_chain = answer_chain
        self.keyword_generator = keyword_generator

    def source(self, sources):
        source = random.choice(sources)
        docs = self.vectorstore.similarity_search(".", k=self.vectorstore._collection.count(), filter={"source":source})
        srcdoc = [random.choice(docs), random.choice(docs), random.choice(docs)]
        question = self.question_chain.run(input_documents=srcdoc, question="")
        return {
            "question": question,
            "source": srcdoc,
        }

    def keyword(self, keyword):
        updated_keyword = self.keyword_generator.run(question=keyword)
        print(keyword)
        print(updated_keyword)
        srcdoc = self.vectorstore.similarity_search(updated_keyword, k=5)
        question = self.question_chain.run(input_documents=srcdoc, question="")
        return {
            "question": question,
            "source": srcdoc,
        }

    def random(self):
        # randomでsourceを選ぶ
        source = random.choice(list(set([k["source"] for k in self.vectorstore._collection.get()["metadatas"]])))
        docs = self.vectorstore.similarity_search(".", k=self.vectorstore._collection.count(), filter={"source":source})
        srcdoc = [random.choice(docs), random.choice(docs), random.choice(docs)]
        question = self.question_chain.run(input_documents=srcdoc, question="")
        return {
            "question": question,
            "source": srcdoc,
        }

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.max_tokens_limit and isinstance(
            self.question_chain, StuffDocumentsChain
        ):
            tokens = [
                self.question_chain.llm_chain.llm.get_num_tokens(doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(self, question: str) -> List[Document]:
        docs = self.vectorstore.as_retriever().get_relevant_documents(question)
        return self._reduce_tokens_below_limit(docs)

    def answer(self, question):
        docs = self._get_docs(question)
        inputs= {"question": question}
        answer = self.answer_chain.run(input_documents=docs, **inputs)
        return {
            "answer": answer,
            "source": docs,
        }

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        vectorstore: VectorStore,
        chain_type: str = "stuff",
        verbose: bool = False,
        **kwargs: Any,
    ):
        """Load chain from LLM."""
        question_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            prompt=QUESTION_PROMPT,
        )
        answer_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            prompt=ANSWER_PROMPT,
        )
        keyword_generator = LLMChain(
            llm=llm, prompt=KEYWORD_PROMPT, verbose=verbose
        )
        return cls(
            vectorstore=vectorstore,
            question_chain=question_chain,
            answer_chain=answer_chain,
            keyword_generator=keyword_generator,
        )


