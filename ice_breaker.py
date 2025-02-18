from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
#from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from third_parties.linkedin import scrape_linkedin_profile

information = """
Carl Gustav Jung (/jʊŋ/ YUUNG;[1][2] German: [kaʁl ˈjʊŋ]; 26 July 1875 – 6 June 1961) was a Swiss psychiatrist, psychotherapist, and psychologist who founded the school of analytical psychology.[3][a] He was a prolific author, illustrator, and correspondent, and a complex and controversial character, in certain ways best known through his autobiography Memories, Dreams, Reflections.[6]

Jung's work has been influential in the fields of psychiatry, anthropology, archaeology, literature, philosophy, psychology,[7] and religious studies. He worked as a research scientist at the Burghölzli psychiatric hospital in Zurich, under Eugen Bleuler. Jung established himself as an influential mind, developing a friendship with Sigmund Freud, founder of psychoanalysis, conducting a lengthy correspondence paramount to their joint vision of human psychology. Jung is widely regarded as one of the most influential psychologists in history.[8][9]
    """

if __name__ == "__main__":
    print("hello LangChain!")
    load_dotenv()
    print(os.environ["OPENAI_API_KEY"])
    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    #llm = ChatOllama(model="mistral")
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    chain = summary_prompt_template | llm | StrOutputParser()

    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/"
    )

    res = chain.invoke(input={"information": linkedin_data})
    print(res)
