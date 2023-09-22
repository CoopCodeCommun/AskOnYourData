from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain

_template = """A partir de la conversation suivante et d'une question complémentaire, reformulez la question complémentaire pour en faire une question indépendante.
Vous pouvez supposer que la question porte sur la solution logicielle libre TiBillet.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Vous êtes un assistant IA chargé de répondre à des questions sur la solution logicielle TiBillet.
Vous disposez des extraits suivants d'un long document et d'une question. Fournissez une réponse conversationnelle.
Si vous ne connaissez pas la réponse, dites simplement "Hmm, je ne suis pas sûr". N'essayez pas d'inventer une réponse.
Si la question ne porte pas sur TiBillet, informez poliment la personne que vous n'êtes habilité à répondre qu'aux questions portant sur TiBillet.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain
