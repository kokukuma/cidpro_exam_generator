import os
import openai
import langchain

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from lib.question_generation import ExamQuestionGenerator

SOURCES = {
    "ENISA": [
        "./pdf_docs/ENISA Report - Data Protection Engineering.pdf",
        "./pdf_docs/ENISA Threat Landscape 2022.pdf",
    ],
    "NIST": [
        "./pdf_docs/NIST.SP.800-37r2.pdf",
        "./pdf_docs/NIST.SP.800-63A-4.ipd.pdf",
        "./pdf_docs/NIST.SP.800-63C-4.ipd.pdf",
        "./pdf_docs/NIST.SP.800-63-4.ipd.pdf",
        "./pdf_docs/NIST.SP.800-63B-4.ipd.pdf",
        "./pdf_docs/NIST.IR.8062.pdf",
    ],
    "NISTSP80063": [
        "./pdf_docs/NIST.SP.800-37r2.pdf",
        "./pdf_docs/NIST.SP.800-63A-4.ipd.pdf",
        "./pdf_docs/NIST.SP.800-63C-4.ipd.pdf",
        "./pdf_docs/NIST.SP.800-63-4.ipd.pdf",
        "./pdf_docs/NIST.SP.800-63B-4.ipd.pdf",
        "./pdf_docs/NIST.IR.8062.pdf",
    ],
    "OAUTH": [
        "https://www.ietf.org/archive/id/draft-ietf-oauth-dpop-16.html",
        "https://www.ietf.org/archive/id/draft-ietf-oauth-step-up-authn-challenge-15.html",
        "https://www.ietf.org/archive/id/draft-ietf-oauth-selective-disclosure-jwt-04.html",
        "https://www.ietf.org/archive/id/draft-ietf-oauth-browser-based-apps-13.html",
        "https://www.ietf.org/archive/id/draft-ietf-oauth-cross-device-security-01.html",
        "https://www.ietf.org/archive/id/draft-ietf-oauth-security-topics-22.html",
        "https://www.ietf.org/archive/id/draft-ietf-oauth-v2-1-08.html",
        "https://www.ietf.org/archive/id/draft-ietf-oauth-rar-23.html",
        "https://www.ietf.org/archive/id/draft-ietf-oauth-jwt-introspection-response-12.html",
        "https://www.ietf.org/archive/id/draft-ietf-oauth-step-up-authn-challenge-13.html",
        "https://www.rfc-editor.org/rfc/rfc9126.html",
        "https://www.rfc-editor.org/rfc/rfc8705.html",
        "https://www.rfc-editor.org/rfc/rfc8707.html",
        "https://www.rfc-editor.org/rfc/rfc8693.html",
        "https://www.rfc-editor.org/rfc/rfc8628.html",
        "https://www.rfc-editor.org/rfc/rfc8252.html",
        "https://www.rfc-editor.org/rfc/rfc7662.html",
        "https://www.rfc-editor.org/rfc/rfc7009.html",
        "https://www.rfc-editor.org/rfc/rfc6819.html",
        "https://www.rfc-editor.org/rfc/rfc6749.html",
    ],
    "OIDC": [
        "https://openid.net/specs/openid-connect-core-1_0.html",
        "https://openid.net/specs/openid-connect-discovery-1_0.html",
        "https://openid.net/specs/openid-connect-registration-1_0.html",
        "https://openid.net/specs/oauth-v2-multiple-response-types-1_0.html",
        "https://openid.net/specs/oauth-v2-form-post-response-mode-1_0.html",
        "https://openid.net/specs/openid-connect-rpinitiated-1_0.html",
        "https://openid.net/specs/openid-connect-session-1_0.html",
        "https://openid.net/specs/openid-connect-frontchannel-1_0.html",
        "https://openid.net/specs/openid-connect-backchannel-1_0.html",
        "https://openid.net/specs/openid-connect-federation-1_0.html",
        "https://openid.net/specs/openid-connect-prompt-create-1_0.html",
        "https://openid.net/specs/openid-connect-basic-1_0.html",
        "https://openid.net/specs/openid-connect-implicit-1_0.html",
        "https://openid.net/specs/openid-connect-migration-1_0.html",
        "https://openid.net/specs/openid-connect-self-issued-v2-1_0.html",
        "https://openid.net/specs/openid-4-verifiable-presentations-1_0.html",
    ],
    "BOK": [
        "https://bok.idpro.org/article/id/49/",
        "https://bok.idpro.org/article/id/92/",
        "https://bok.idpro.org/article/id/86/",
        "https://bok.idpro.org/article/id/90/",
        "https://bok.idpro.org/article/id/41/",
        "https://bok.idpro.org/article/id/88/",
        "https://bok.idpro.org/article/id/85/",
        "https://bok.idpro.org/article/id/25/",
        "https://bok.idpro.org/article/id/84/",
        "https://bok.idpro.org/article/id/51/",
        "https://bok.idpro.org/article/id/31/",
        "https://bok.idpro.org/article/id/52/",
        "https://bok.idpro.org/article/id/80/",
        "https://bok.idpro.org/article/id/76/",
        "https://bok.idpro.org/article/id/78/",
        "https://bok.idpro.org/article/id/79/",
        "https://bok.idpro.org/article/id/11/",
        "https://bok.idpro.org/article/id/8/",
        "https://bok.idpro.org/article/id/27/",
        "https://bok.idpro.org/article/id/61/",
        "https://bok.idpro.org/article/id/62/",
        "https://bok.idpro.org/article/id/65/",
        "https://bok.idpro.org/article/id/64/",
        "https://bok.idpro.org/article/id/49/",
        "https://bok.idpro.org/article/id/38/",
        "https://bok.idpro.org/article/id/39/",
        "https://bok.idpro.org/article/id/30/",
        "https://bok.idpro.org/article/id/42/",
        "https://bok.idpro.org/article/id/44/",
        "https://bok.idpro.org/article/id/45/",
        "https://bok.idpro.org/article/id/18/",
        "https://bok.idpro.org/article/id/24/",
        "https://bok.idpro.org/article/id/16/",
    ],
}

def get_source_files():
    files =[u for sublist in SOURCES.values() for u in sublist]
    return files


def create_question_with_search(target, vectorstore, show_page_content=False, doc_number=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=1.0, model_name=os.getenv("GPT_MODEL"))
    eqg = ExamQuestionGenerator.from_llm(llm, vectorstore, chain_type="stuff")

    # TODO: 整理する...
    if target.startswith("https://"):
        result = eqg.source([target])
    elif target.upper() in ["NIST", "ENISA", "NISTSP80063", "OAUTH", "OIDC", "BOK"]:
        result = eqg.source(SOURCES[target.upper()])
    elif doc_number is not None:
        result = eqg.docs(target, doc_number)
    elif target == "":
        result = eqg.random()
    else:
        result = eqg.keyword(target)

    text = f"{result['question']}\n\n"
    if show_page_content:
        text += "---------- page contents\n"
        for s in result['source']:
            text += f"{s.metadata['source']}\n"
            text += f"{s.page_content}\n\n"
        text += "----------\n"
    else:
        for s in set([s.metadata['source'] for s in result['source']]):
            text += f"{s}\n"

    return text

def create_answer(question, vectorstore):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0.1, model_name=os.getenv("GPT_MODEL"))
    eqg = ExamQuestionGenerator.from_llm(llm, vectorstore, chain_type="stuff")

    result = eqg.answer(question)

    text = f"{result['answer']}\n\n"
    for s in set([s.metadata['source'] for s in result['source']]):
        text += f"{s}\n"
    return text


def get_qa(vectorstore, model=os.getenv("GPT_MODEL")):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0.5, model_name=os.getenv("GPT_MODEL"))

    prompt_template = """Use the following pieces of context to answer the question at the end. The answer must be explained why.

{context}

Question: {question}
"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        question_generator=LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT),
        combine_docs_chain=doc_chain,
        return_source_documents=True,
    )
    return qa

