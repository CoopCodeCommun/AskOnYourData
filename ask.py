from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from uuid import uuid4
from datetime import datetime
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
from langchain.document_loaders import JSONLoader

import json
from pathlib import Path

file_path = './data.json'
data = json.loads(Path(file_path).read_text())

loader = JSONLoader(
    file_path = file_path,
    jq_schema=".[]")


from langchain.agents import create_json_agent, AgentExecutor
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec

json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

model = OpenAI(model_name='gpt-4', temperature=0)
json_agent_executor = create_json_agent(
    llm=model, toolkit=json_toolkit, verbose=True
)
prompt = """
        Tu es un assistant qui fabrique des dictionnaire python de ce type : 
         
        nouvel_evenement = {
          name:str,
          datetime: datetime,
          lieux : uuid,
          artists: uuid,
          products: uuid,
          options : uuid,
          tags: uuid,
          short_description: str,
          categorie:str,
        }
        
        Avec l'aide du contexte fourni, construit un dictionnaire qui contient toutes les informations nécessaires à la création de cet évènement :
        
        Un concert de Ziskakan le deuxieme vendredi de janvier 2024 à 20h30 à la salle de spectacle "Le Billetistan". Un tarif pour les adhérants, et un tarif plein.
        """
json_agent_executor.run(prompt)


class Event(BaseModel):
    name: str = Field(description="Nom de l'évènement")
    datetime: datetime = Field(description="Date et heure de l'évènement")
    lieux: uuid4 = Field(description="Uuid du lieu de l'évènement")
    artists: List[uuid4] = Field(description="Liste des uuid des artistes")
    prix: List[uuid4] = Field(description="Liste des uuid des prix des billets")
    options: List[uuid4] = Field(description="Liste des uuid des options")
    tags: List[uuid4] = Field(description="Liste des uuid des tags")
    short_description: str = Field(description="Description courte de l'évènement")


# model_name = "text-davinci-003"
# model_name = "gpt4"
model_name = "gpt3.5-turbo"

temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

event_query = "Un concert de Ziskakan le deuxieme vendredi de janvier 2024 à 20h30 à la salle de spectacle 'Le Bisik'. Billet d'entrée avec deux tarif :  5€ pour les adhérants de l'association et 10€ en plein tarif"
# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Event)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query=event_query)
output = model(_input.to_string())
parser.parse(output)
